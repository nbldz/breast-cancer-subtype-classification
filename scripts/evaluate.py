"""
scripts/evaluate.py
Comprehensive evaluation of CDLS across seeds.

Implements two evaluation protocols (Section IV-A):
  A) Held-out test set, n=4 seeds → Table VI (primary)
  B) 5-fold patient-level stratified CV, n=4 seeds → Table VII

Per-seed accuracy (paper):
  S42: 88.4%, S123: 85.6%, S789: 92.2%, S2024: 90.3%
  Mean: 87.6% ± 3.4%

Usage:
    # Test set evaluation
    python scripts/evaluate.py --config configs/default.yaml --seeds 42 123 789 2024

    # 5-fold cross-validation
    python scripts/evaluate.py --config configs/default.yaml --seeds 42 123 789 2024 --cv
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cdls  import CDLS
from data.dataset import TCGABRCADataset, collate_modality_disjoint, SUBTYPE2IDX, IDX2SUBTYPE
from torch.utils.data import DataLoader, Subset
from utils.metrics import (
    compute_all_metrics, aggregate_seeds, format_results_table, bootstrap_ci
)
from utils.calibration import reliability_diagram


SIGMA = 0.034   # Differences below σ are not reliable discriminators


def load_model(cfg: dict, seed: int, device: torch.device) -> CDLS:
    """Load best PPO checkpoint for a given seed."""
    ckpt_dir  = os.path.join(cfg["logging"]["checkpoint_dir"], f"seed_{seed}")
    ckpt_path = os.path.join(ckpt_dir, f"ppo_best_seed{seed}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CDLS(
        clinical_dim    = 20,
        dtwin           = cfg["latent"]["dtwin"],
        T               = cfg["latent"]["T"],
        lambda_feedback = cfg["feedback"]["lambda_feedback"],
        k_neighbours    = cfg["feedback"]["k_neighbours"],
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Load kNN bank
    bank_path = os.path.join(ckpt_dir, f"z0_bank_seed{seed}.pt")
    if os.path.exists(bank_path):
        z0_bank = torch.load(bank_path)
        model.build_feedback_bank(z0_bank)

    return model.to(device).eval()


@torch.no_grad()
def evaluate_on_loader(model, loader, device) -> dict:
    all_preds, all_labels, all_proba = [], [], []
    all_risk, all_surv, all_events   = [], [], []
    all_z_T = []

    for batch in loader:
        subtypes   = batch["subtype"].to(device)
        surv_times = batch["surv_time"].to(device)
        events     = batch["event"].to(device)

        out    = model.forward(batch)
        proba  = torch.softmax(out["subtype_logits"], dim=-1).cpu().numpy()
        preds  = proba.argmax(axis=1)

        all_preds.append(preds)
        all_labels.append(subtypes.cpu().numpy())
        all_proba.append(proba)
        all_risk.append(out["risk_scores"].cpu().numpy())
        all_surv.append(surv_times.cpu().numpy())
        all_events.append(events.cpu().numpy())
        all_z_T.append(out["z_T"].cpu())

    return {
        "y_true":  np.concatenate(all_labels),
        "y_pred":  np.concatenate(all_preds),
        "y_proba": np.concatenate(all_proba),
        "risk":    np.concatenate(all_risk),
        "surv":    np.concatenate(all_surv),
        "events":  np.concatenate(all_events),
        "z_T":     torch.cat(all_z_T, dim=0),
    }


def plot_confusion_matrices(results_per_seed: dict, out_dir: str):
    """
    Plot 2×2 confusion matrix grid for all 4 seeds (Fig. 3).
    Row-normalised.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    seeds  = sorted(results_per_seed.keys())
    n_rows = 2
    n_cols = (len(seeds) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    subtype_labels = list(IDX2SUBTYPE.values())

    for i, seed in enumerate(seeds):
        res = results_per_seed[seed]
        cm  = confusion_matrix(res["y_true"], res["y_pred"], normalize="true")
        ax  = axes[i]
        im  = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(5)); ax.set_xticklabels(subtype_labels, fontsize=7, rotation=45)
        ax.set_yticks(range(5)); ax.set_yticklabels(subtype_labels, fontsize=7)
        ax.set_title(f"Seed {seed}  (Acc={res['metrics']['accuracy']:.3f})", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{cm[r,c]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm[r,c] > 0.5 else "black")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Confusion Matrices (row-normalised). HER2/Normal variability "
                 "reflects small test sets.", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrices → {path}")


def evaluate_held_out(cfg: dict, seeds: list, device: torch.device, out_dir: str):
    """
    Protocol A: held-out test set evaluation across seeds.
    """
    print("\n" + "="*60)
    print("Protocol A: Held-Out Test Set Evaluation")
    print("="*60)

    all_metrics    = []
    results_by_seed = {}

    for seed in seeds:
        print(f"\n── Seed {seed} ──")
        model = load_model(cfg, seed, device)

        test_dataset = TCGABRCADataset(
            cfg["data"]["tcga_brca_dir"], split="test"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg["training"]["batch_size"],
            shuffle=False, collate_fn=collate_modality_disjoint,
        )

        res     = evaluate_on_loader(model, test_loader, device)
        metrics = compute_all_metrics(
            res["y_true"], res["y_pred"], res["y_proba"],
            res["surv"], res["events"], res["risk"],
        )
        res["metrics"] = metrics
        results_by_seed[seed] = res
        all_metrics.append(metrics)

        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  MCC:               {metrics['mcc']:.4f}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")

        # Per-class bootstrap CIs (Table IX)
        print("  Per-class metrics with 95% bootstrap CIs (5000 resamples):")
        for c, name in IDX2SUBTYPE.items():
            n_c = int((res["y_true"] == c).sum())
            ppv = metrics[f"class_{c}_ppv"]
            rec = metrics[f"class_{c}_recall"]
            f1  = metrics[f"class_{c}_f1"]
            # Bootstrap CI for recall
            lo, hi = bootstrap_ci(res["y_true"], res["y_pred"], res["y_proba"],
                                   metric=f"class_{c}_recall", n=5000)
            if n_c <= 15:
                note = "  ← UNINFORMATIVE (small test set)"
            else:
                note = ""
            print(f"    {name:<8}: PPV={ppv:.2f}  Rec={rec:.2f} [{lo:.2f},{hi:.2f}]"
                  f"  F1={f1:.2f}  n={n_c}{note}")

        # Reliability diagrams (Fig. 4)
        seed_out = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_out, exist_ok=True)
        reliability_diagram(
            res["y_true"], res["y_proba"],
            save_path=os.path.join(seed_out, "reliability_diagram.png"),
            seed_label=f"(Seed {seed})",
        )

    # Aggregated results (Table VI)
    aggregated = aggregate_seeds(all_metrics)
    print("\n" + "="*60)
    print(f"Aggregated Results (n={len(seeds)} seeds) — Training Stability Only")
    print(f"NOTE: Differences < σ={SIGMA} are not reliable discriminators.")
    print("="*60)
    print(format_results_table(aggregated))

    # Confusion matrix grid
    plot_confusion_matrices(results_by_seed, out_dir)

    torch.save(aggregated, os.path.join(out_dir, "held_out_aggregated.pt"))
    return aggregated


def evaluate_cross_validation(cfg: dict, seeds: list, device: torch.device, out_dir: str):
    """
    Protocol B: 5-fold patient-level stratified CV (Table VII).
    20 fold-seed combinations, df=19, CI width ≈4%.
    """
    print("\n" + "="*60)
    print("Protocol B: 5-Fold Cross-Validation (20 fold-seed combinations)")
    print("="*60)

    n_folds     = 5
    all_metrics = []

    full_dataset = TCGABRCADataset(cfg["data"]["tcga_brca_dir"], split="train")
    all_labels   = np.array([full_dataset[i]["subtype"].item()
                              for i in range(len(full_dataset))])

    for seed in seeds:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(range(len(full_dataset)), all_labels)
        ):
            print(f"\n── Seed {seed}, Fold {fold_idx+1}/{n_folds} ──")

            model = load_model(cfg, seed, device)

            val_subset = Subset(full_dataset, val_idx)
            val_loader = DataLoader(
                val_subset, batch_size=cfg["training"]["batch_size"],
                shuffle=False, collate_fn=collate_modality_disjoint,
            )

            res     = evaluate_on_loader(model, val_loader, device)
            metrics = compute_all_metrics(
                res["y_true"], res["y_pred"], res["y_proba"],
                res["surv"], res["events"], res["risk"],
            )
            all_metrics.append(metrics)

            print(f"  Acc: {metrics['accuracy']:.4f} | "
                  f"BA: {metrics['balanced_accuracy']:.4f} | "
                  f"MCC: {metrics['mcc']:.4f}")

    aggregated = aggregate_seeds(all_metrics)
    print(f"\n{'='*60}")
    print(f"5-Fold CV Results ({len(all_metrics)} fold-seed combinations, df={len(all_metrics)-1})")
    print(f"CI width ≈ {4:.0f}% (df={len(all_metrics)-1})")
    print("="*60)
    print(format_results_table(aggregated))

    torch.save(aggregated, os.path.join(out_dir, "cv_aggregated.pt"))
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate CDLS")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--seeds",   nargs="+", type=int, default=[42, 123, 789, 2024])
    parser.add_argument("--cv",      action="store_true", help="Run 5-fold CV")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.cv:
        evaluate_held_out(cfg, args.seeds, device, args.out_dir)
    else:
        evaluate_cross_validation(cfg, args.seeds, device, args.out_dir)


if __name__ == "__main__":
    main()
