"""
utils/metrics.py
Evaluation metrics for CDLS.

Primary metrics (Section IV-C):
  - Balanced Accuracy
  - MCC (Matthews Correlation Coefficient) [31]

Secondary metrics:
  - Accuracy, Macro-F1, Weighted-F1, Cohen's κ, Macro-AUC
  - Per-class: Precision (PPV), Recall (Sensitivity), F1, NPV
  - ECE (Expected Calibration Error), Brier Score
  - C-index (auxiliary survival; not primary), IBS

All results reported as mean ± std (n=4 seeds).
Differences < σ = 0.034 are not interpretable as reliable discriminators.

Bootstrap CIs: 5,000 resamples (95% CI).
"""

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
)
from typing import Dict, List, Optional, Tuple


SIGMA_THRESHOLD = 0.034   # Differences below this are not reliable discriminators
N_BOOTSTRAP     = 5000


# ─── Primary Metrics ──────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true:    np.ndarray,            # (N,)   ground-truth subtype indices
    y_pred:    np.ndarray,            # (N,)   predicted subtype indices
    y_proba:   Optional[np.ndarray] = None,   # (N, 5) softmax probabilities
    surv_times: Optional[np.ndarray] = None,
    events:     Optional[np.ndarray] = None,
    risk_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the full metric suite for one seed / fold evaluation.
    """
    metrics = {}

    # ── Primary ───────────────────────────────────────────────────────────
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"]               = matthews_corrcoef(y_true, y_pred)

    # ── Secondary ─────────────────────────────────────────────────────────
    metrics["accuracy"]          = accuracy_score(y_true, y_pred)
    metrics["macro_f1"]          = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    metrics["weighted_f1"]       = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["cohen_kappa"]       = cohen_kappa_score(y_true, y_pred)

    if y_proba is not None:
        try:
            metrics["macro_auc"] = roc_auc_score(
                y_true, y_proba, average="macro", multi_class="ovr"
            )
        except ValueError:
            metrics["macro_auc"] = float("nan")

        # ECE and Brier (macro averaged over classes)
        metrics["ece"]         = compute_ece(y_true, y_proba)
        metrics["brier_score"] = compute_multiclass_brier(y_true, y_proba)

        # Per-class AUC
        n_classes = y_proba.shape[1]
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            try:
                metrics[f"auc_class_{c}"] = roc_auc_score(y_bin, y_proba[:, c])
            except ValueError:
                metrics[f"auc_class_{c}"] = float("nan")

    # ── Per-class precision / recall / F1 ─────────────────────────────────
    for c in range(5):
        y_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        tp = ((y_pred_bin == 1) & (y_bin == 1)).sum()
        fp = ((y_pred_bin == 1) & (y_bin == 0)).sum()
        fn = ((y_pred_bin == 0) & (y_bin == 1)).sum()
        tn = ((y_pred_bin == 0) & (y_bin == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        npv  = tn / (tn + fn + 1e-8)
        metrics[f"class_{c}_ppv"]    = float(prec)
        metrics[f"class_{c}_recall"] = float(rec)
        metrics[f"class_{c}_f1"]     = float(f1)
        metrics[f"class_{c}_npv"]    = float(npv)
        metrics[f"class_{c}_n"]      = int(y_bin.sum())

    # ── Survival (auxiliary) ──────────────────────────────────────────────
    if risk_scores is not None and surv_times is not None and events is not None:
        try:
            from lifelines.utils import concordance_index
            metrics["c_index"] = concordance_index(surv_times, -risk_scores, events)
        except ImportError:
            metrics["c_index"] = _concordance_index_numpy(surv_times, risk_scores, events)

    return metrics


# ─── ECE ──────────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    ECE = 0.031 ± 0.005 reported in paper (Table VI).
    """
    y_pred       = np.argmax(y_proba, axis=1)
    confidences  = np.max(y_proba, axis=1)
    accuracies   = (y_pred == y_true).astype(float)

    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    n          = len(y_true)

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc  = accuracies[mask].mean()
            ece     += (mask.sum() / n) * abs(avg_conf - avg_acc)

    return float(ece)


# ─── Multiclass Brier Score ───────────────────────────────────────────────────

def compute_multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Macro-averaged multiclass Brier score.
    Brier = 0.045 ± 0.003 reported in paper (Table VI).
    """
    n_classes = y_proba.shape[1]
    score     = 0.0
    for c in range(n_classes):
        y_bin  = (y_true == c).astype(float)
        score += np.mean((y_proba[:, c] - y_bin) ** 2)
    return float(score / n_classes)


# ─── Concordance Index ────────────────────────────────────────────────────────

def _concordance_index_numpy(
    times: np.ndarray,
    risk:  np.ndarray,
    events: np.ndarray,
) -> float:
    """Harrell's C-index (numpy fallback when lifelines is absent)."""
    concordant = discordant = 0
    for i in range(len(times)):
        if events[i] == 0:
            continue
        for j in range(len(times)):
            if times[j] > times[i]:
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


# ─── Bootstrap Confidence Intervals ──────────────────────────────────────────

def bootstrap_ci(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metric:  str  = "balanced_accuracy",
    n:       int  = N_BOOTSTRAP,
    alpha:   float = 0.05,
    seed:    int  = 0,
) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for a single metric.
    Used for per-class metrics (5,000 resamples, Table IX).
    """
    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt  = y_true[idx]
        yp  = y_pred[idx]
        ypr = y_proba[idx] if y_proba is not None else None

        try:
            m = compute_all_metrics(yt, yp, ypr)
            estimates.append(m.get(metric, float("nan")))
        except Exception:
            estimates.append(float("nan"))

    estimates = np.array([e for e in estimates if not np.isnan(e)])
    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return lo, hi


# ─── Multi-Seed Aggregation ───────────────────────────────────────────────────

def aggregate_seeds(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict]:
    """
    Aggregate metrics across n seeds.
    Returns mean, std, 95% CI (via Eq. 22: CI = x̄ ± t_{α/2, n-1} · s/√n).
    """
    import scipy.stats as stats

    all_keys = set().union(*[m.keys() for m in metrics_list])
    out      = {}
    n        = len(metrics_list)

    for k in all_keys:
        vals = np.array([m.get(k, float("nan")) for m in metrics_list])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # t-distribution CI (n=4 → df=3, t_0.025,3 = 3.182; CI width ≈ 11%)
        t_crit = stats.t.ppf(0.975, df=max(len(vals) - 1, 1))
        ci_lo  = mean - t_crit * std / np.sqrt(len(vals))
        ci_hi  = mean + t_crit * std / np.sqrt(len(vals))
        out[k] = {
            "mean":  mean,
            "std":   std,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n":     len(vals),
        }

    return out


def format_results_table(aggregated: Dict) -> str:
    """Pretty-print aggregated results."""
    lines = [
        f"{'Metric':<30} {'Mean':>8} {'±Std':>8} {'95% CI':>22}",
        "-" * 72,
    ]
    primary = ["balanced_accuracy", "mcc", "accuracy", "macro_f1", "weighted_f1",
               "macro_auc", "ece", "brier_score", "c_index"]
    for k in primary:
        if k in aggregated:
            v = aggregated[k]
            ci_str = f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
            lines.append(
                f"{k:<30} {v['mean']:>8.3f} {v['std']:>8.3f} {ci_str:>22}"
            )
    return "\n".join(lines)
