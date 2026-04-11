"""
utils/calibration.py
Calibration analysis for CDLS outputs.

Results from paper:
  ECE   = 0.031 ± 0.005  (Table VI)
  Brier = 0.045 ± 0.003  (Table VI)

Fig. 4: Reliability diagrams per PAM50 subtype (Seed 789).
Close diagonal tracking for Basal and LumA.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# PAM50 subtype names
SUBTYPES = ["Basal", "HER2", "LumA", "LumB", "Normal"]


def reliability_diagram(
    y_true:    np.ndarray,    # (N,)
    y_proba:   np.ndarray,    # (N, 5)
    n_bins:    int = 10,
    save_path: Optional[str] = None,
    seed_label: str = "",
) -> plt.Figure:
    """
    Generate per-class reliability diagrams (Fig. 4).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for c, (ax, name) in enumerate(zip(axes[:5], SUBTYPES)):
        y_bin   = (y_true == c).astype(float)
        prob_c  = y_proba[:, c]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_conf = []
        frac_pos  = []

        for i in range(n_bins):
            mask = (prob_c > bin_edges[i]) & (prob_c <= bin_edges[i + 1])
            if mask.sum() > 0:
                mean_conf.append(prob_c[mask].mean())
                frac_pos.append(y_bin[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
        ax.plot(mean_conf, frac_pos, "s-", color="steelblue", label=name, markersize=6)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives",      fontsize=10)
        ax.set_title(f"({chr(97+c)}) {name}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)

    # Compute overall ECE for title
    y_pred   = np.argmax(y_proba, axis=1)
    confs    = np.max(y_proba, axis=1)
    accs     = (y_pred == y_true).astype(float)
    ece_val  = _ece_from_arrays(confs, accs, n_bins=15)

    axes[5].axis("off")
    fig.suptitle(
        f"Reliability Diagrams per PAM50 Subtype {seed_label} — ECE = {ece_val:.3f}",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _ece_from_arrays(confidences, accuracies, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n         = len(confidences)
    ece       = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(confidences[mask].mean() - accuracies[mask].mean())
    return float(ece)


def temperature_scaling_calibrate(
    logits: np.ndarray,    # (N, 5) — validation set logits
    labels: np.ndarray,    # (N,)
    lr:     float = 0.01,
    max_iter: int = 200,
) -> float:
    """
    Find optimal temperature T* to minimise NLL on validation set.
    Used to initialise / fine-tune the temperature parameter in PAM50Classifier.

    Returns: optimal temperature value
    """
    import torch
    import torch.nn.functional as F

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1))
    optimiser   = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval_():
        optimiser.zero_grad()
        scaled = logits_t / temperature.clamp(min=0.01)
        loss   = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimiser.step(eval_)
    return float(temperature.item())
