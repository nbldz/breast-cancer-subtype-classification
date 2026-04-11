"""
scripts/train.py
Main training script for CDLS.

Usage:
    # Stage 1 + 2 for a single seed:
    python scripts/train.py --config configs/default.yaml --seed 42

    # All 4 seeds:
    for seed in 42 123 789 2024; do
        python scripts/train.py --config configs/default.yaml --seed $seed
    done

    # Deterministic mode (exact reproducibility):
    python scripts/train.py --config configs/default.yaml --seed 42 --deterministic

Reproducibility:
    Seeds {42, 123, 789, 2024} control all stochastic components.
    Full reproducibility instructions: see README.md.
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset   import (
    TCGABRCADataset, BCSCDataset, ConcatDataset,
    collate_modality_disjoint, build_datasets
)
from models.cdls    import CDLS
from trainers.pretrain   import PreTrainer
from trainers.ppo_trainer import PPOTrainer
from utils.leakage_check  import verify_no_leakage
from utils.metrics        import aggregate_seeds, format_results_table


def set_seed(seed: int, deterministic: bool = False):
    """
    Set all random seeds for reproducibility.
    Seeds {42, 123, 789, 2024} used in paper.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, seed: int):
    """Build train/val/test DataLoaders."""
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_data = build_datasets(cfg, split="train")
    val_data   = build_datasets(cfg, split="val")
    test_data  = build_datasets(cfg, split="test")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_modality_disjoint,
        generator=g, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_modality_disjoint,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_modality_disjoint,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def infer_clinical_dim(cfg: dict) -> int:
    """
    Infer clinical feature dimensionality from TCGA-BRCA clinical.csv.
    Falls back to 20 if file not found.
    """
    import pandas as pd
    clin_path = os.path.join(cfg["data"]["tcga_brca_dir"], "clinical.csv")
    if os.path.exists(clin_path):
        df  = pd.read_csv(clin_path)
        exc = {"patient_id", "subtype", "split", "survival_time", "event"}
        return len([c for c in df.columns if c not in exc])
    return 20


def train_single_seed(cfg: dict, seed: int, args) -> dict:
    """Train CDLS for a single seed. Returns test metrics."""
    set_seed(seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'#'*60}")
    print(f"# Training CDLS  |  Seed: {seed}  |  Device: {device}")
    print(f"{'#'*60}")

    output_dir   = os.path.join(cfg["logging"]["checkpoint_dir"], f"seed_{seed}")
    clinical_dim = infer_clinical_dim(cfg)

    # ── Build model ────────────────────────────────────────────────────────
    model = CDLS(
        clinical_dim    = clinical_dim,
        dtwin           = cfg["latent"]["dtwin"],
        T               = cfg["latent"]["T"],
        lambda_feedback = cfg["feedback"]["lambda_feedback"],
        k_neighbours    = cfg["feedback"]["k_neighbours"],
    )
    param_counts = model.parameter_count()
    print(f"\nModel parameters:")
    for k, v in param_counts.items():
        print(f"  {k:<20}: {v:>12,}")

    # ── Build data loaders ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed)

    # ── Stage 1: Pre-training ──────────────────────────────────────────────
    if args.stage in ("pretrain", "both", None):
        pretrain_ckpt = os.path.join(output_dir, f"pretrain_best_seed{seed}.pt")
        if os.path.exists(pretrain_ckpt) and not args.force_retrain:
            print(f"\nLoading pretrain checkpoint: {pretrain_ckpt}")
            ckpt = torch.load(pretrain_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        else:
            pretrainer = PreTrainer(
                model, train_loader, val_loader, cfg, device, output_dir, seed
            )
            pretrain_ckpt = pretrainer.train()
            ckpt = torch.load(pretrain_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])

        # Build kNN bank from training embeddings (Algorithm 1, line 3)
        pretrainer_for_bank = PreTrainer(
            model, train_loader, val_loader, cfg, device, output_dir, seed
        )
        z0_bank = pretrainer_for_bank.build_knn_bank(train_loader)

        # Save bank
        bank_path = os.path.join(output_dir, f"z0_bank_seed{seed}.pt")
        torch.save(z0_bank, bank_path)

    # ── Stage 2: PPO training ──────────────────────────────────────────────
    if args.stage in ("ppo", "both", None):
        # Load kNN bank if not already built
        bank_path = os.path.join(output_dir, f"z0_bank_seed{seed}.pt")
        if os.path.exists(bank_path):
            z0_bank = torch.load(bank_path)
            model.build_feedback_bank(z0_bank)

        ppo_trainer = PPOTrainer(
            model, train_loader, val_loader, cfg, device, output_dir, seed
        )
        best_ckpt = ppo_trainer.train()

        # ── Final test evaluation ──────────────────────────────────────────
        test_metrics, y_true, y_pred, y_proba, z_T = ppo_trainer.evaluate_test(
            test_loader, checkpoint_path=best_ckpt
        )

        # Save results
        results_path = os.path.join(output_dir, f"test_metrics_seed{seed}.pt")
        torch.save({
            "metrics":   test_metrics,
            "y_true":    y_true,
            "y_pred":    y_pred,
            "y_proba":   y_proba,
            "z_T":       z_T,
            "seed":      seed,
        }, results_path)

        return test_metrics

    return {}


def train_all_seeds(cfg: dict, args) -> None:
    """
    Train CDLS on all 4 seeds and report aggregated results.
    n=4 seeds used for training stability assessment only
    (not statistical population inference).
    """
    seeds = cfg["seeds"]
    if args.seed is not None:
        seeds = [args.seed]

    all_metrics = []
    for seed in seeds:
        metrics = train_single_seed(cfg, seed, args)
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        print("\n" + "="*60)
        print("Aggregated Results (n={} seeds)".format(len(all_metrics)))
        print("NOTE: n={} reflects training stability only, "
              "not population inference.".format(len(all_metrics)))
        print("Differences < σ=0.034 are not reliable discriminators.")
        print("="*60)
        aggregated = aggregate_seeds(all_metrics)
        print(format_results_table(aggregated))

        # Save aggregated results
        out_dir = cfg["logging"]["checkpoint_dir"]
        torch.save(aggregated, os.path.join(out_dir, "aggregated_results.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train CDLS")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Single seed to train (default: all seeds in config)")
    parser.add_argument("--stage",        choices=["pretrain", "ppo", "both"],
                        default="both", help="Training stage")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic CUDA operations")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain even if checkpoint exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["log_dir"],        exist_ok=True)

    train_all_seeds(cfg, args)


if __name__ == "__main__":
    main()
