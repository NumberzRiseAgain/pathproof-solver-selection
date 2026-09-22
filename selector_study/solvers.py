"""Classical preconditioners and one preconditioned conjugate gradient.

All six methods share the same PCG so that iteration counts and operator
applications are comparable. Constants are declared in PREREGISTRATION.md.
"""
import time

import numpy as np
import scipy.sparse as sp

TAU = 1e-8
MAXIT = 20000
WALL_CAP = 30.0


class Counter:
    def __init__(self):
        self.matvec = 0
        self.precond = 0


def pcg(A, b, M, counter, tau=TAU, maxit=MAXIT, wall_cap=WALL_CAP, check_every=100):
    """Preconditioned CG, with convergence defined on the TRUE residual.

    The recurrence residual drifts from the true residual on ill-conditioned
    operators, so it is used only to decide when to spend a true-residual check.
    A run is converged only when the true relative residual is at or below tau.
    Returns a dict so that every outcome, including a false claim by the
    recurrence, is recorded rather than collapsed into a number.
    """
    t_start = time.perf_counter()
    n = b.shape[0]
    x = np.zeros(n)
    r = b.copy()
    bnorm = float(np.linalg.norm(b))
    out = dict(iters=0, converged=False, timeout=False, breakdown=False,
               rec_claimed_early=False, true_res=np.inf, rec_res=np.inf)
    if bnorm == 0:
        out.update(converged=True, true_res=0.0, rec_res=0.0)
        return x, out

    def true_res():
        counter.matvec += 1
        return float(np.linalg.norm(b - A @ x)) / bnorm

    z = M(r) if M is not None else r
    if M is not None:
        counter.precond += 1
    p = z.copy()
    rz = float(r @ z)
    it = 0
    while it < maxit:
        Ap = A @ p
        counter.matvec += 1
        pAp = float(p @ Ap)
        if pAp <= 0:
            out.update(iters=it, breakdown=True, rec_res=float(np.linalg.norm(r))/bnorm,
                       true_res=true_res())
            return x, out
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        it += 1
        rn = float(np.linalg.norm(r)) / bnorm
        if rn <= tau or it % check_every == 0:
            tr = true_res()
            if tr <= tau:
                out.update(iters=it, converged=True, rec_res=rn, true_res=tr)
                return x, out
            if rn <= tau:
                out["rec_claimed_early"] = True
        if time.perf_counter() - t_start > wall_cap:
            out.update(iters=it, timeout=True, rec_res=rn, true_res=true_res())
            return x, out
        z = M(r) if M is not None else r
        if M is not None:
            counter.precond += 1
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    out.update(iters=it, rec_res=float(np.linalg.norm(r))/bnorm, true_res=true_res())
    return x, out


# ------------------------------------------------------------------ M0, M1
def make_none(A, probe=None):
    return None


def make_jacobi(A, probe=None):
    dinv = 1.0 / A.diagonal()
    return lambda r: dinv * r


# ------------------------------------------------------------------ M2 SSOR
def make_ssor(A, probe=None, omega=1.0):
    d = A.diagonal()
    L = sp.tril(A, k=-1, format="csr")
    U = sp.triu(A, k=1, format="csr")
    Dw = sp.diags(d / omega)
    Mlo = (Dw + L).tocsr()
    Mhi = (Dw + U).tocsr()
    scale = omega / (2.0 - omega)
    Dm = sp.diags(d / omega)

    from scipy.sparse.linalg import splu
    lo_lu = splu(Mlo.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                 options=dict(SymmetricMode=True))
    hi_lu = splu(Mhi.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                 options=dict(SymmetricMode=True))

    def apply(r):
        y = lo_lu.solve(r)
        y = Dm @ y
        y = hi_lu.solve(y)
        return scale * y

    return apply


# ------------------------------------------------------------------ M3 IC(0)
def ic0_factor(A):
    """Incomplete Cholesky with zero fill on the lower pattern of A.

    Returns L in CSR, or None if a non-positive pivot is met, which is a real
    outcome for an indefinite-looking incomplete factor and is reported.
    """
    Alo = sp.tril(A, format="csr").astype(float)
    indptr, indices, data = Alo.indptr.copy(), Alo.indices.copy(), Alo.data.copy()
    n = A.shape[0]
    scratch = np.zeros(n)
    diag_pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        cols_i = indices[s:e]
        # scatter current row
        scratch[cols_i] = data[s:e]
        for t in range(s, e):
            j = indices[t]
            if j == i:
                break
            sj, ej = indptr[j], indptr[j + 1]
            dj = diag_pos[j]
            cols_j = indices[sj:dj]           # strictly lower part of row j
            acc = float(scratch[cols_j] @ data[sj:dj]) if dj > sj else 0.0
            val = (scratch[j] - acc) / data[dj]
            data[t] = val
            scratch[j] = val
        # diagonal
        dpos = e - 1
        diag_pos[i] = dpos
        cols_l = indices[s:dpos]
        acc = float(data[s:dpos] @ data[s:dpos]) if dpos > s else 0.0
        piv = scratch[i] - acc
        if piv <= 0:
            scratch[cols_i] = 0.0
            return None
        data[dpos] = np.sqrt(piv)
        scratch[cols_i] = 0.0
    return sp.csr_matrix((data, indices, indptr), shape=A.shape)


def make_ic0(A, probe=None):
    L = ic0_factor(A)
    if L is None:
        return None
    from scipy.sparse.linalg import splu
    lu = splu(L.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
              options=dict(SymmetricMode=True))
    lut = splu(L.T.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
               options=dict(SymmetricMode=True))

    def apply(r):
        return lut.solve(lu.solve(r))

    return apply


# ------------------------------------------------------- M4 Neumann degree 4
def make_neumann(A, probe=None, degree=4):
    d = A.diagonal()
    dinv = 1.0 / d
    # scaled operator  N = I - D^-1 A ; series  sum_{k<=degree} N^k D^-1
    def apply(r):
        z = dinv * r
        acc = z.copy()
        term = z
        for _ in range(degree):
            term = term - dinv * (A @ term)
            acc = acc + term
        return acc

    return apply


# ----------------------------------------------- M5 Chebyshev degree 8
def make_chebyshev(A, probe=None, degree=8):
    """Chebyshev polynomial preconditioner on the Jacobi-scaled operator.

    Standard Chebyshev semi-iteration started from zero, so the map r -> x is a
    fixed polynomial in A and the preconditioner is linear and symmetric.
    Interval bounds come from the probe's Ritz values, which is the only place
    in the study where a method consumes the probe.
    """
    dinv = 1.0 / A.diagonal()
    lo, hi = probe["ritz_min"], probe["ritz_max"]
    lo = max(lo, hi * 1e-6)
    d = 0.5 * (hi + lo)
    c = 0.5 * (hi - lo)
    if c <= 0:
        return lambda r: dinv * r
    sigma1 = d / c

    def apply(r):
        z = dinv * r
        x_prev = np.zeros_like(z)
        x = z / d
        rho_prev = 1.0 / sigma1
        for _ in range(1, degree):
            rho = 1.0 / (2.0 * sigma1 - rho_prev)
            zk = dinv * (r - A @ x)
            x_next = x + rho * rho_prev * (x - x_prev) + (2.0 * rho / c) * zk
            x_prev, x = x, x_next
            rho_prev = rho
        return x

    return apply


METHODS = [
    ("M0_none",       make_none,       False),
    ("M1_jacobi",     make_jacobi,     False),
    ("M2_ssor",       make_ssor,       False),
    ("M3_ic0",        make_ic0,        False),
    ("M4_neumann4",   make_neumann,    False),
    ("M5_chebyshev8", make_chebyshev,  True),   # needs the probe
]
