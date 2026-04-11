"""
utils/latent_geometry.py
Latent Trajectory Geometry Analysis for PPO selection justification.

Section III-O.5 / Table V:
  Three geometry metrics that justify PPO selection over SAC/TD3:

  1. Path KL Divergence (Eq. 21):
     D_KL = (1/C(n,2)) Σ_{i<j} D_KL(N(μ_i, Σ_i) || N(μ_j, Σ_j))
     → Higher = different patients take different paths (patient-specific refinement)
     PPO: 1.73  SAC: 1.14  TD3: 0.97  IGR: 0.31

  2. Latent Space Coverage:
     Fraction of k-means (k=50) cluster centroids occupied by trajectory points
     → Higher = broader manifold exploration
     PPO: 0.71  SAC: 0.58  IGR: 0.34

  3. Action Entropy H(a_τ) = -E[log π_θ(a_τ|z_τ)]:
     → Higher = stochastic diversity of policy exploration
     PPO: 2.47  SAC: 1.89  TD3: 0.00 (deterministic)

PPO vs. SAC accuracy gap: +0.004 (within σ=0.034 — not a reliable discriminator).
PPO is selected EXCLUSIVELY for trajectory geometry properties.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


def path_kl_divergence(
    trajectories: List[torch.Tensor],   # List of (T+1, dtwin) patient trajectories
    policy:        nn.Module,            # PPO policy to extract trajectory distributions
    device:        torch.device,
) -> float:
    """
    Path KL divergence (Eq. 21):
        D_KL = (1/C(n,2)) Σ_{i<j} D_KL( N(μ_i, Σ_i) || N(μ_j, Σ_j) )

    For each patient trajectory, we fit a Gaussian to the trajectory points.
    Higher KL indicates different patients take qualitatively different paths.

    Args:
        trajectories: List of per-patient trajectory tensors (T+1, dtwin)
        policy:        PPO policy (to get trajectory distributions)

    Returns:
        Mean pairwise KL divergence
    """
    # Fit Gaussian per patient trajectory
    mus  = []
    vars = []

    for traj in trajectories:
        # traj: (T+1, dtwin) — latent states z_0 to z_T
        mu  = traj.mean(dim=0)           # (dtwin,)
        var = traj.var(dim=0) + 1e-8     # (dtwin,) — diagonal covariance
        mus.append(mu)
        vars.append(var)

    n     = len(mus)
    if n < 2:
        return 0.0

    # Pairwise KL between diagonal Gaussians
    # KL(N(μ1, Σ1) || N(μ2, Σ2)) = 0.5 * [tr(Σ2^{-1}Σ1) + (μ2-μ1)^T Σ2^{-1}(μ2-μ1)
    #                                         - d + ln(|Σ2|/|Σ1|)]
    total_kl = 0.0
    count    = 0

    for i in range(n):
        for j in range(i + 1, n):
            mu1, var1 = mus[i], vars[i]
            mu2, var2 = mus[j], vars[j]
            d = mu1.shape[0]

            kl_ij = 0.5 * (
                (var1 / var2).sum()
                + ((mu2 - mu1) ** 2 / var2).sum()
                - d
                + (torch.log(var2).sum() - torch.log(var1).sum())
            )
            total_kl += kl_ij.item()
            count    += 1

    return total_kl / count


def latent_coverage(
    trajectories: List[torch.Tensor],   # List of (T+1, dtwin)
    k_clusters:   int = 50,
    seed:         int = 789,
) -> float:
    """
    Latent space coverage: fraction of k-means cluster centroids
    (fit on training data) occupied by at least one trajectory point.

    Args:
        trajectories: per-patient trajectory tensors
        k_clusters:   number of centroids (paper: k=50)

    Returns:
        Coverage fraction ∈ [0, 1]
    """
    from sklearn.cluster import KMeans

    # Stack all trajectory points
    all_points = torch.cat(trajectories, dim=0).numpy()   # (N*T, dtwin)

    if len(all_points) < k_clusters:
        return float("nan")

    km = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10)
    km.fit(all_points)
    centroids = km.cluster_centers_   # (k, dtwin)

    # For each centroid, check if any trajectory point is within radius r
    # Use the mean nearest-centroid distance as adaptive radius
    from sklearn.metrics import pairwise_distances_argmin_min
    _, min_dists = pairwise_distances_argmin_min(centroids, all_points)
    radius = np.mean(min_dists) * 2

    occupied = 0
    for c in centroids:
        dists = np.linalg.norm(all_points - c, axis=1)
        if dists.min() <= radius:
            occupied += 1

    return occupied / k_clusters


def action_entropy(
    z_trajectories: List[torch.Tensor],   # (T, dtwin) per patient
    policy:          nn.Module,
    device:          torch.device,
) -> float:
    """
    Mean action entropy H(a_τ) = -E[log π_θ(a_τ | z_τ)] averaged over τ and patients.
    Deterministic methods (IGR, TD3) have entropy=0.
    """
    from torch.distributions import Normal

    all_entropies = []
    for traj in z_trajectories:
        # traj: (T, dtwin) — states z_0 to z_{T-1}
        traj_dev = traj.to(device)
        for t in range(len(traj)):
            z   = traj_dev[t:t+1]
            mu, std = policy(z)
            ent = Normal(mu, std).entropy().sum(dim=-1).mean().item()
            all_entropies.append(ent)

    return float(np.mean(all_entropies)) if all_entropies else 0.0


def compute_geometry_metrics(
    model:         nn.Module,
    test_loader,
    device:        torch.device,
    k_clusters:    int = 50,
    max_patients:  int = 165,   # paper: n=165 test patients
) -> Dict[str, float]:
    """
    Compute all three geometry metrics for Table V.

    Paper results (Seed 789, n=165):
        PPO: Path KL=1.73, Coverage=0.71, Entropy=2.47, Acc=0.922
    """
    model.eval()
    trajectories = []
    all_preds, all_labels = [], []
    n = 0

    with torch.no_grad():
        for batch in test_loader:
            if n >= max_patients:
                break
            b_size = batch["subtype"].shape[0]
            n_take = min(b_size, max_patients - n)

            out = model.forward(batch, return_trajectory=True)
            traj = out["trajectory"]   # list of (B, 7) tensors

            for i in range(n_take):
                # Extract trajectory for patient i: (T+1, 7)
                pt_traj = torch.stack([z[i] for z in traj], dim=0)
                trajectories.append(pt_traj)

            proba  = torch.softmax(out["subtype_logits"][:n_take], dim=-1).cpu().numpy()
            labels = batch["subtype"][:n_take].numpy()
            all_preds.append(proba.argmax(axis=1))
            all_labels.append(labels)
            n += n_take

    # Accuracy
    y_pred  = np.concatenate(all_preds)
    y_true  = np.concatenate(all_labels)
    acc     = float((y_pred == y_true).mean())

    # Path KL
    path_kl = path_kl_divergence(trajectories, model.policy, device)

    # Coverage
    coverage = latent_coverage(trajectories, k_clusters=k_clusters)

    # Action entropy
    # For trajectory: exclude z_T (last point), use z_0 to z_{T-1}
    traj_for_ent = [t[:-1] for t in trajectories]  # (T, 7) per patient
    ent = action_entropy(traj_for_ent, model.policy, device)

    metrics = {
        "path_kl":  path_kl,
        "coverage": coverage,
        "entropy":  ent,
        "accuracy": acc,
    }

    print("\nLatent Trajectory Geometry Metrics:")
    print(f"  Path KL Divergence:  {path_kl:.2f}  (PPO paper: 1.73)")
    print(f"  Latent Coverage:     {coverage:.2f}  (PPO paper: 0.71)")
    print(f"  Action Entropy:      {ent:.2f}  (PPO paper: 2.47)")
    print(f"  Accuracy:            {acc:.3f}  (PPO paper: 0.922)")
    print("\nNOTE: PPO selected for geometry properties, not for")
    print("      marginal accuracy gain over SAC (+0.004 < σ=0.034).")

    return metrics
