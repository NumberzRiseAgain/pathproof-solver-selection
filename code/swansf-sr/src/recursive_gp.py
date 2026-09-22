"""
Discovery of a recursive update rule, rather than a static formula.

The static search (gp_search.py) looks for f(last, mean, std, slope, ...) - a
closed-form combination of precomputed summary statistics, evaluated once. That
discovers a FORMULA.

This module searches for g in

    s_0 = 0
    s_t = clip( g(s_{t-1}, x_t, dx_t) )      t = 1 .. T
    score = s_T

which discovers a PROCEDURE: something with state that iterates over the
sequence. That is the shape of a filter - the Kalman update is exactly this
form - and it is the category the DARPA topic's examples all live in
(Transformers rediscovering the Kalman filter, GP rediscovering wavelets,
MCTS rediscovering optimisation algorithms).

The static search is kept and becomes the BASELINE. "Does a discovered
procedure buy anything over a discovered formula?" is then a measured question
rather than an assumption, and it is the question the paper turns on.

Two design points that decide whether this works at all:

  * Stability. A fold of an unbounded g diverges to inf or NaN within a few
    timesteps. The grammar therefore carries tanh, and the state is clipped
    every step. Programs that still diverge score zero rather than poisoning
    the run with NaNs.

  * Terminals include dx_t, the first difference. Without it the search cannot
    express an innovation term, which is the core of every filter worth
    discovering.

Written as a small self-contained GP rather than through gplearn, because
gplearn evaluates a program once against a 2-D matrix and cannot fold. The
representation, operators and selection are conventional; nothing here is
novel and nothing here is meant to be.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# grammar
# --------------------------------------------------------------------------


def _pdiv(a, b):
    return np.where(np.abs(b) > 1e-6, a / np.where(np.abs(b) > 1e-6, b, 1.0), 1.0)


def _psqrt(a):
    return np.sqrt(np.abs(a))


FUNCTIONS = {
    "add": (2, np.add),
    "sub": (2, np.subtract),
    "mul": (2, np.multiply),
    "div": (2, _pdiv),
    "min": (2, np.minimum),
    "max": (2, np.maximum),
    "tanh": (1, np.tanh),          # the squashing that keeps a fold bounded
    "abs": (1, np.abs),
    "sqrt": (1, _psqrt),
    "neg": (1, np.negative),
}
FUNC_NAMES = list(FUNCTIONS)

STATE_CLIP = 10.0
CONSTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


class Node:
    __slots__ = ("op", "kids")

    def __init__(self, op, kids=()):
        self.op = op                      # func name, ("x", i), ("d", i), "s", or float
        self.kids = list(kids)

    # -- structure ---------------------------------------------------------
    def size(self):
        return 1 + sum(k.size() for k in self.kids)

    def depth(self):
        return 1 + max((k.depth() for k in self.kids), default=0)

    def copy(self):
        return Node(self.op, [k.copy() for k in self.kids])

    def nodes(self):
        yield self
        for k in self.kids:
            yield from k.nodes()

    # -- evaluation --------------------------------------------------------
    def eval(self, s, x, d):
        """s: [n]   x: [n, p]   d: [n, p]   -> [n]"""
        op = self.op
        if isinstance(op, float):
            return np.full_like(s, op)
        if op == "s":
            return s
        if isinstance(op, tuple):
            src = x if op[0] == "x" else d
            return src[:, op[1]]
        arity, fn = FUNCTIONS[op]
        if arity == 1:
            return fn(self.kids[0].eval(s, x, d))
        return fn(self.kids[0].eval(s, x, d), self.kids[1].eval(s, x, d))

    # -- printing ----------------------------------------------------------
    def text(self, names):
        op = self.op
        if isinstance(op, float):
            return f"{op:g}"
        if op == "s":
            return "s"
        if isinstance(op, tuple):
            return f"{'' if op[0] == 'x' else 'd_'}{names[op[1]]}"
        return f"{op}(" + ", ".join(k.text(names) for k in self.kids) + ")"

    def terminals_used(self, names):
        out = set()
        for nd in self.nodes():
            if nd.op == "s":
                out.add("s")
            elif isinstance(nd.op, tuple):
                out.add(("" if nd.op[0] == "x" else "d_") + names[nd.op[1]])
        return out


# --------------------------------------------------------------------------
# random trees and variation
# --------------------------------------------------------------------------

def _rand_terminal(rng, p, p_state=0.30):
    r = rng.random()
    if r < p_state:
        return Node("s")
    if r < p_state + 0.12:
        # This branch was unreachable when p_state was raised to 0.30 while the
        # constant cutoff stayed at 0.30, so no constant could ever be drawn.
        # Constants are the retention coefficient in a * s_{t-1}: without them
        # a discovered filter cannot express decay, only pure accumulation.
        return Node(float(rng.choice(CONSTS)))
    kind = "x" if rng.random() < 0.6 else "d"
    return Node((kind, int(rng.integers(p))))


def seed_filter(rng, p):
    """
    An affine-recursion template: s_t = a*s_{t-1} + <innovation>.

    This is the general form of a linear filter - exponential smoothing and the
    Kalman update are both instances - and seeding part of the population with
    it supplies the one piece of structure the search reliably fails to
    discover on its own: that the state should be carried forward at all.
    Everything inside the innovation term is still searched.

    Reported as what it is. The filter SHAPE is given; which parameters enter
    the innovation, how they combine, and the retention coefficient are
    discovered.
    """
    innov = random_tree(rng, p, int(rng.integers(2, 4)))
    if rng.random() < 0.5:
        carried = Node("s")
    else:
        carried = Node("mul", [Node(float(rng.choice((0.5, 1.0, 2.0)))), Node("s")])
    return Node("add", [carried, innov])


def random_tree(rng, p, depth, full=False):
    if depth <= 1:
        return _rand_terminal(rng, p)
    if not full and rng.random() < 0.25:
        return _rand_terminal(rng, p)
    name = FUNC_NAMES[int(rng.integers(len(FUNC_NAMES)))]
    arity = FUNCTIONS[name][0]
    return Node(name, [random_tree(rng, p, depth - 1, full) for _ in range(arity)])


def _pick(rng, tree):
    nds = list(tree.nodes())
    return nds[int(rng.integers(len(nds)))]


def _replace(tree, target, new):
    """Return a copy of `tree` with the node identical to `target` swapped."""
    if tree is target:
        return new
    out = Node(tree.op)
    out.kids = [_replace(k, target, new) for k in tree.kids]
    return out


def crossover(rng, a, b, max_depth):
    child = _replace(a.copy(), _pick(rng, a.copy()) if False else _pick(rng, a), _pick(rng, b).copy())
    return child if child.depth() <= max_depth else a.copy()


def subtree_mutate(rng, a, p, max_depth):
    child = _replace(a.copy(), _pick(rng, a), random_tree(rng, p, 3))
    return child if child.depth() <= max_depth else a.copy()


def point_mutate(rng, a, p):
    child = a.copy()
    nd = _pick(rng, child)
    if nd.kids:
        same = [n for n in FUNC_NAMES if FUNCTIONS[n][0] == len(nd.kids)]
        nd.op = same[int(rng.integers(len(same)))]
    else:
        rep = _rand_terminal(rng, p)
        nd.op = rep.op
    return child


def hoist_mutate(rng, a):
    return _pick(rng, a).copy()


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------

def fold_score(tree, X, stride=1):
    """
    X: [n, T, p] standardised. Returns final state s_T as [n], plus a flag for
    whether the recursion stayed finite.
    """
    n, T, p = X.shape
    s = np.zeros(n, dtype="float64")
    prev = X[:, 0, :]
    diverged = False
    for t in range(1, T, stride):
        x = X[:, t, :]
        d = x - prev
        with np.errstate(all="ignore"):
            s_new = tree.eval(s, x, d)
        if not np.all(np.isfinite(s_new)):
            diverged = True
            s_new = np.where(np.isfinite(s_new), s_new, 0.0)
        s = np.clip(s_new, -STATE_CLIP, STATE_CLIP)
        prev = x
    return s, diverged


def gini(y, s):
    """|Gini| rank separability, direction-agnostic, ties handled."""
    y = np.asarray(y).astype(int)
    s = np.where(np.isfinite(s), s, 0.0)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return 0.0
    if np.ptp(s) == 0:
        return 0.0
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype="float64")
    r[order] = np.arange(1, len(s) + 1)
    vals, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    auc = (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(abs(2.0 * auc - 1.0))


# --------------------------------------------------------------------------
# evolution
# --------------------------------------------------------------------------

def evolve(X, y, param_names, seed=0, population=600, generations=25,
           tournament=7, max_depth=6, parsimony=0.0015, stride=1,
           oob_frac=0.15, seed_filter_frac=0.30, verbose=False):
    """Returns (best_tree, log). X: [n, T, p] standardised, train rows only."""
    rng = np.random.default_rng(seed)
    p = X.shape[2]

    n = X.shape[0]
    n_oob = max(20, int(n * oob_frac))
    oob = rng.choice(n, size=n_oob, replace=False)
    inb = np.setdiff1d(np.arange(n), oob)
    X_in, y_in = X[inb], y[inb]
    X_oob, y_oob = X[oob], y[oob]

    n_seeded = int(population * seed_filter_frac)
    pop = [seed_filter(rng, p) for _ in range(n_seeded)]
    pop += [random_tree(rng, p, int(rng.integers(2, 5)), full=rng.random() < 0.5)
            for _ in range(population - n_seeded)]

    def fitness(tree, Xs, ys):
        s, diverged = fold_score(tree, Xs, stride)
        f = gini(ys, s)
        if diverged:
            f *= 0.5                       # penalise, do not disqualify
        return f - parsimony * tree.size()

    log = []
    best, best_oob = None, -np.inf
    for g in range(generations):
        fits = np.array([fitness(t, X_in, y_in) for t in pop])
        order = np.argsort(fits)[::-1]

        # elite judged on out-of-bag rows, which is what we actually keep
        for idx in order[:5]:
            f_oob = fitness(pop[idx], X_oob, y_oob)
            if f_oob > best_oob:
                best_oob, best = f_oob, pop[idx].copy()

        log.append({"gen": g, "best_inbag": float(fits[order[0]]),
                    "best_oob": float(best_oob),
                    "mean_size": float(np.mean([t.size() for t in pop])),
                    "frac_stateful": float(np.mean(
                        [any(nd.op == "s" for nd in t.nodes()) for t in pop]))})
        if verbose:
            print(f"  gen {g:3d}  inbag {fits[order[0]]:.4f}  oob {best_oob:.4f}")

        new = [pop[i].copy() for i in order[:max(2, population // 50)]]   # elitism
        while len(new) < population:
            def tsel():
                c = rng.choice(population, size=tournament, replace=False)
                return pop[c[int(np.argmax(fits[c]))]]
            r = rng.random()
            if r < 0.60:
                new.append(crossover(rng, tsel(), tsel(), max_depth))
            elif r < 0.80:
                new.append(subtree_mutate(rng, tsel(), p, max_depth))
            elif r < 0.92:
                new.append(point_mutate(rng, tsel(), p))
            else:
                new.append(hoist_mutate(rng, tsel()))
        pop = new

    return best, log


def describe(tree, param_names):
    return {
        "rule": "s_t = " + tree.text(param_names),
        "size": tree.size(),
        "depth": tree.depth(),
        "terminals": sorted(tree.terminals_used(param_names)),
    }
