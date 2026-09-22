"""The spectral probe and the declared feature vector.

k Lanczos steps on the Jacobi-preconditioned operator from a seeded start.
Eleven features, fixed in PREREGISTRATION.md section 4. Nothing else about the
operator is inspected.
"""
import time
import numpy as np

K_STEPS = 30


def lanczos_ritz(A, k=K_STEPS, seed=0):
    """Return the Ritz values of the Jacobi-preconditioned operator."""
    n = A.shape[0]
    dinv = 1.0 / A.diagonal()
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n)
    # M-inner product with M = D, so work in the D^{1/2}-scaled space:
    # the Jacobi-preconditioned operator D^-1 A is similar to D^-1/2 A D^-1/2,
    # which is symmetric, so Lanczos runs on that.
    s = np.sqrt(dinv)
    q /= np.linalg.norm(q)
    alphas, betas = [], []
    q_prev = np.zeros(n)
    beta = 0.0
    Q = [q]
    for j in range(k):
        v = s * (A @ (s * q))
        alpha = float(q @ v)
        v = v - alpha * q - beta * q_prev
        # one full reorthogonalisation pass, declared
        for u in Q:
            v -= (u @ v) * u
        beta = float(np.linalg.norm(v))
        alphas.append(alpha)
        if beta < 1e-13 or j == k - 1:
            break
        betas.append(beta)
        q_prev = q
        q = v / beta
        Q.append(q)
    T = np.diag(alphas)
    if betas:
        T += np.diag(betas, 1) + np.diag(betas, -1)
    w = np.linalg.eigvalsh(T)
    return np.clip(w, 1e-300, None)


FEATURE_NAMES = [
    "log_kappa_ritz", "log_ritz_min", "log_ritz_max",
    "ritz_q25", "ritz_q50", "ritz_q75",
    "max_rel_gap", "log_ritz_mean", "log_ritz_std",
    "log_n", "nnz_per_row", "min_diag_dominance",
]


def features(A, seed=0):
    """Return (feature dict, probe wall-clock seconds, ritz array)."""
    t0 = time.perf_counter()
    w = lanczos_ritz(A, seed=seed)
    t_probe = time.perf_counter() - t0

    lw = np.log10(w)
    rmin, rmax = float(w[0]), float(w[-1])
    gaps = np.diff(w) / np.maximum(w[:-1], 1e-300)
    n = A.shape[0]

    # one pass over the matrix for the structural features
    t1 = time.perf_counter()
    d = np.abs(A.diagonal())
    absrow = np.abs(A).sum(axis=1).A.ravel()
    offdiag = absrow - d
    dom = d / np.maximum(offdiag, 1e-300)
    t_struct = time.perf_counter() - t1

    f = {
        "log_kappa_ritz": float(np.log10(rmax / max(rmin, 1e-300))),
        "log_ritz_min": float(np.log10(max(rmin, 1e-300))),
        "log_ritz_max": float(np.log10(rmax)),
        "ritz_q25": float(np.quantile(w, 0.25) / rmax),
        "ritz_q50": float(np.quantile(w, 0.50) / rmax),
        "ritz_q75": float(np.quantile(w, 0.75) / rmax),
        "max_rel_gap": float(gaps.max()) if gaps.size else 0.0,
        "log_ritz_mean": float(lw.mean()),
        "log_ritz_std": float(lw.std()),
        "log_n": float(np.log10(n)),
        "nnz_per_row": float(A.nnz / n),
        "min_diag_dominance": float(np.min(dom)),
    }
    return f, t_probe + t_struct, w
