"""Operator families for the selector study.

Every operator is generated from a declared discretisation. Nothing is downloaded.
All matrices are symmetric positive definite in CSR form.

Families and their swept parameter are fixed in PREREGISTRATION.md section 1.
"""
import numpy as np
import scipy.sparse as sp


def _spd_check_cheap(A):
    """Cheap necessary conditions. Full definiteness is not asserted here."""
    d = A.diagonal()
    return bool(np.all(d > 0))


# ---------------------------------------------------------------- F1, F2, F4
def _conduction(nx, ny, nz, kappa_field):
    """Cell-centred finite volume Laplacian with harmonic face conductivity.

    kappa_field: array of shape (nx, ny, nz), strictly positive.
    Dirichlet on all faces, so the operator is SPD.
    """
    n = nx * ny * nz
    idx = np.arange(n).reshape(nx, ny, nz)
    k = kappa_field
    rows, cols, vals = [], [], []
    diag = np.zeros(n)

    def face(a, b, ka, kb):
        # harmonic mean face conductivity
        return 2.0 * ka * kb / (ka + kb)

    for axis in range(3):
        size = (nx, ny, nz)[axis]
        if size < 2:
            continue
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, size - 1)
        sl_hi[axis] = slice(1, size)
        i_lo = idx[tuple(sl_lo)].ravel()
        i_hi = idx[tuple(sl_hi)].ravel()
        kf = face(None, None, k[tuple(sl_lo)].ravel(), k[tuple(sl_hi)].ravel())
        rows.append(i_lo); cols.append(i_hi); vals.append(-kf)
        rows.append(i_hi); cols.append(i_lo); vals.append(-kf)
        np.add.at(diag, i_lo, kf)
        np.add.at(diag, i_hi, kf)

    # boundary faces: Dirichlet, half-cell distance, adds to the diagonal
    for axis in range(3):
        size = (nx, ny, nz)[axis]
        if size < 2:
            continue
        for end in (0, size - 1):
            sl = [slice(None)] * 3
            sl[axis] = end
            ii = idx[tuple(sl)].ravel()
            kb = k[tuple(sl)].ravel()
            np.add.at(diag, ii, 2.0 * kb)

    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def _lognormal_field(shape, contrast, rng, corr=3):
    """Smoothed log-normal conductivity with the stated decadic contrast."""
    g = rng.standard_normal(shape)
    # cheap smoothing by repeated box average along each axis
    for _ in range(corr):
        for ax in range(3):
            if shape[ax] > 2:
                g = (np.roll(g, 1, ax) + g + np.roll(g, -1, ax)) / 3.0
    g -= g.mean()
    s = g.std()
    if s > 0:
        g /= s
    return 10.0 ** (0.5 * np.log10(contrast) * g)


def f1_conduction2d(nx, contrast, seed):
    rng = np.random.default_rng(seed)
    k = _lognormal_field((nx, nx, 1), contrast, rng)
    return _conduction(nx, nx, 1, k)


def f2_conduction3d(nx, contrast, seed):
    rng = np.random.default_rng(seed)
    k = _lognormal_field((nx, nx, nx), contrast, rng)
    return _conduction(nx, nx, nx, k)


def f4_inclusions(nx, jump, seed):
    """Piecewise-constant conductivity on random circular inclusions."""
    rng = np.random.default_rng(seed)
    k = np.ones((nx, nx, 1))
    xs, ys = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    for _ in range(12):
        cx, cy = rng.integers(0, nx, 2)
        r = max(2, int(nx * rng.uniform(0.04, 0.12)))
        m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        k[m, 0] = jump
    return _conduction(nx, nx, 1, k)


# ---------------------------------------------------------------- F3
def f3_anisotropic(nx, ratio, seed):
    """2D anisotropic diffusion, eps in x and 1 in y, 5-point, Dirichlet."""
    n = nx * nx
    idx = np.arange(n).reshape(nx, nx)
    ex = 1.0 / ratio
    ey = 1.0
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for axis, coef in ((0, ex), (1, ey)):
        sl_lo = [slice(None)] * 2
        sl_hi = [slice(None)] * 2
        sl_lo[axis] = slice(0, nx - 1)
        sl_hi[axis] = slice(1, nx)
        i_lo = idx[tuple(sl_lo)].ravel()
        i_hi = idx[tuple(sl_hi)].ravel()
        v = np.full(i_lo.shape, coef)
        rows.append(i_lo); cols.append(i_hi); vals.append(-v)
        rows.append(i_hi); cols.append(i_lo); vals.append(-v)
        np.add.at(diag, i_lo, coef)
        np.add.at(diag, i_hi, coef)
        for end in (0, nx - 1):
            sl = [slice(None)] * 2
            sl[axis] = end
            np.add.at(diag, idx[tuple(sl)].ravel(), 2.0 * coef)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


# ---------------------------------------------------------------- F5
def f5_graph_laplacian(n, sigma, seed):
    """Random geometric graph Laplacian in 2D, shifted by sigma to make it SPD."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    # grid bucketing so the edge list stays linear in n
    cells = max(1, int(np.sqrt(n / 12.0)))
    r = 1.0 / cells
    key = (np.minimum((pts // r).astype(int), cells - 1))
    buckets = {}
    for i, (a, b) in enumerate(key):
        buckets.setdefault((a, b), []).append(i)
    ri, ci = [], []
    for (a, b), members in buckets.items():
        neigh = []
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                neigh.extend(buckets.get((a + da, b + db), []))
        neigh = np.array(neigh)
        for i in members:
            d = np.sum((pts[neigh] - pts[i]) ** 2, axis=1)
            j = neigh[(d < r * r) & (neigh != i)]
            ri.append(np.full(j.shape, i)); ci.append(j)
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    w = np.ones(ri.shape)
    W = sp.coo_matrix((w, (ri, ci)), shape=(n, n)).tocsr()
    W = ((W + W.T) > 0).astype(float)
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(deg) - W
    return (L + sigma * sp.eye(n)).tocsr()


# ---------------------------------------------------------------- F6
def f6_shifted_stiffness(nx, c, seed):
    """K + c*M on a 2D uniform grid: stiffness plus a scaled lumped mass."""
    K = f3_anisotropic(nx, 1.0, seed)          # isotropic 5-point stiffness
    h2 = 1.0 / (nx * nx)
    M = sp.eye(K.shape[0]) * h2                 # lumped mass
    return (K + c * M).tocsr()


# ---------------------------------------------------------------- the grid
FAMILIES = {
    "F1_conduction2d": dict(fn=f1_conduction2d, sizes=[120, 170, 220],
                            params=[1e1, 1e3, 1e5, 1e7]),
    "F2_conduction3d": dict(fn=f2_conduction3d, sizes=[26, 34, 42],
                            params=[1e1, 1e3, 1e5, 1e7]),
    "F3_anisotropic":  dict(fn=f3_anisotropic,  sizes=[120, 170, 220],
                            params=[1.0, 1e2, 1e4, 1e6]),
    "F4_inclusions":   dict(fn=f4_inclusions,   sizes=[120, 170, 220],
                            params=[1e-6, 1e-3, 1e3, 1e6]),
    "F5_graph":        dict(fn=f5_graph_laplacian, sizes=[20000, 60000, 120000],
                            params=[1e-1, 1e-3, 1e-5, 1e-6]),
    "F6_shifted":      dict(fn=f6_shifted_stiffness, sizes=[120, 170, 220],
                            params=[1e-2, 1e0, 1e2, 1e4]),
}


def enumerate_operators():
    """Yield (family, op_id, size_param, sweep_param, seed) over the declared grid."""
    out = []
    for fam, spec in FAMILIES.items():
        for si, s in enumerate(spec["sizes"]):
            for pi, p in enumerate(spec["params"]):
                seed = 1000 * (si + 1) + pi
                out.append((fam, f"{fam}_s{si}_p{pi}", s, p, seed))
    return out


def build(fam, size, param, seed):
    A = FAMILIES[fam]["fn"](size, param, seed)
    A = A.tocsr()
    A.sort_indices()
    assert _spd_check_cheap(A), f"{fam} produced a non-positive diagonal"
    return A
