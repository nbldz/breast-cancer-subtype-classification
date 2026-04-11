"""
utils/intrinsic_dim.py
Intrinsic Dimensionality Estimation for guiding dtwin selection.
Section III-G.

Two estimators:
  1. Two-NN estimator [38]: Facco et al., Sci. Rep. 2017
  2. Maximum Likelihood Estimation [39]: Levina & Bickel, NeurIPS 2005

Result from paper: d̂_id = 6.3 ± 0.4 (across 4 seeds)
→ guided selection of dtwin = 7

IMPORTANT: ID estimators are noisy and local; they provide a heuristic
guide rather than a strict theoretical guarantee. dtwin=7 is empirically
guided, not theoretically derived.
"""

import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors


def two_nn_estimator(X: np.ndarray, fraction: float = 0.9) -> float:
    """
    Two-Nearest-Neighbours (TwoNN) intrinsic dimensionality estimator.
    Facco et al., Scientific Reports 2017.

    Algorithm:
      1. For each point, compute r1 (nearest) and r2 (2nd nearest) distances.
      2. μ_i = r2_i / r1_i
      3. Empirical CDF F(μ) fitted to 1 - μ^{-d}
      4. d estimated via linear regression on log scale.

    Args:
        X:        (N, D) data matrix (latent embeddings z_T)
        fraction: fraction of points to use (exclude points with μ > quantile)

    Returns:
        d_hat: estimated intrinsic dimension
    """
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Avoid division by zero
    r1 = distances[:, 0] + 1e-10
    r2 = distances[:, 1] + 1e-10
    mu = r2 / r1                           # (N,) μ_i = r2/r1

    # Sort and compute empirical Pareto CDF
    mu_sorted = np.sort(mu)
    n_use     = int(fraction * N)
    mu_use    = mu_sorted[:n_use]
    i_vals    = np.arange(1, n_use + 1)

    # Empirical CDF: F(μ_i) ≈ i/N
    # 1 - F(μ) = μ^{-d}  → log(1 - F) = -d·log(μ)
    log_mu  = np.log(mu_use)
    log_cdf = np.log(1 - i_vals / (N + 1))

    # Linear regression: log(1-F) = -d * log(μ)  →  slope = -d
    valid   = (log_mu > 0) & np.isfinite(log_cdf)
    if valid.sum() < 5:
        return float("nan")

    slope, _ = np.polyfit(log_mu[valid], log_cdf[valid], 1)
    d_hat    = -slope
    return float(d_hat)


def mle_estimator(X: np.ndarray, k: int = 5) -> float:
    """
    Maximum Likelihood Estimation of intrinsic dimension.
    Levina & Bickel, NeurIPS 2005.

    For each point x_i:
        m̂_k(x_i) = [ (1/k) Σ_{j=1}^{k} log(T_k(x_i) / T_j(x_i)) ]^{-1}
    where T_j(x_i) is the distance to the j-th nearest neighbour.

    Returns mean over all points.
    """
    N = X.shape[0]
    k_actual = min(k, N - 1)
    nbrs     = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances    = distances[:, 1:]    # exclude self (distance=0)

    # T_k = distance to k-th NN
    T_k    = distances[:, -1:]         # (N, 1)
    # Avoid log(0)
    T_k    = np.maximum(T_k, 1e-10)
    ratios = np.log(T_k / np.maximum(distances, 1e-10))   # (N, k)
    m_hat  = 1.0 / (ratios.mean(axis=1) + 1e-10)          # (N,)

    # Trim extreme values
    m_hat = m_hat[np.isfinite(m_hat)]
    m_hat = m_hat[m_hat > 0]
    m_hat = np.clip(m_hat, 1, 50)

    return float(np.mean(m_hat))


def estimate_intrinsic_dim(
    Z: np.ndarray,
    k_mle: int = 5,
    fraction_twonn: float = 0.9,
) -> Tuple[float, float, float]:
    """
    Estimate intrinsic dimensionality using both estimators.
    Returns: (mean_estimate, twonn_estimate, mle_estimate)

    Paper result: d̂_id = 6.3 ± 0.4 (four seeds)
    """
    d_twonn = two_nn_estimator(Z, fraction=fraction_twonn)
    d_mle   = mle_estimator(Z, k=k_mle)

    # Average of both estimators
    valid = [d for d in [d_twonn, d_mle] if np.isfinite(d)]
    mean  = float(np.mean(valid)) if valid else float("nan")

    return mean, d_twonn, d_mle


def select_dtwin(
    Z_per_seed: list,   # list of (N, D) arrays; one per seed
    candidates: list = [5, 6, 7, 8, 10, 12],
) -> Tuple[int, float, float]:
    """
    Estimate intrinsic dimensionality across seeds and recommend dtwin.
    Returns: (recommended_dtwin, mean_d_id, std_d_id)
    """
    estimates = []
    for Z in Z_per_seed:
        d, _, _ = estimate_intrinsic_dim(Z)
        if np.isfinite(d):
            estimates.append(d)

    if not estimates:
        return 7, float("nan"), float("nan")

    mean_d = float(np.mean(estimates))
    std_d  = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0

    # Select smallest candidate ≥ mean_d (rounded up)
    # Table XIX confirms d=7 as smallest stable dimension
    recommended = min(
        (c for c in sorted(candidates) if c >= mean_d),
        default=max(candidates),
    )

    print(f"Intrinsic dimensionality: d̂_id = {mean_d:.1f} ± {std_d:.1f}")
    print(f"Recommended dtwin = {recommended} "
          f"(smallest stable dimension; paper: 7)")

    return recommended, mean_d, std_d
