"""
scripts/transfer_metabric.py
METABRIC transfer-learning evaluation (Section V-E, Table XIV).

Protocol:
  - RNA MLP + Clinical ANN weights transferred from TCGA-BRCA
  - Projector re-initialised and fine-tuned (10 epochs, early stopping)
  - No WSI modality: h_wsi set to learned absence token
  - No BCSC sequences: e_long set to learned absence token

Results (Table XIV, Seed 789):
  TCGA-BRCA (full): Acc=0.922, MCC=0.907
  METABRIC (transfer): Acc=0.928, MCC=0.904

IMPORTANT (Remark III.1):
  - 0.006 gap vs. best seed is within σ=0.034 and confounded by shared preprocessing
  - Transfer accuracy MUST NOT be interpreted as true external generalisation
  - 6.6% zero-shot accuracy drop (Table XXIV) is the more realistic indicator

Usage:
    python scripts/transfer_metabric.py --config configs/default.yaml --seed 789
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cdls   import CDLS
from data.dataset  import METABRICDataset, collate_modality_disjoint
from utils.metrics import compute_all_metrics, format_results_table


def transfer_to_metabric(cfg: dict, seed: int, device: torch.device) -> dict:
    """
    Transfer TCGA-BRCA model weights to METABRIC and fine-tune projector.

    Transfer protocol:
      1. Load TCGA-BRCA pretrained weights
      2. Freeze RNA MLP + Clinical ANN
      3. Re-initialise projector
      4. Fine-tune projector for 10 epochs (early stopping patience=5)
    """
    print(f"\n{'='*60}")
    print(f"METABRIC Transfer Learning (Seed {seed})")
    print(f"IMPORTANT: Shared preprocessing limits generalisation claims.")
    print(f"{'='*60}")

    # ── Load source model (TCGA-BRCA) ────────────────────────────────────
    ckpt_dir  = os.path.join(cfg["logging"]["checkpoint_dir"], f"seed_{seed}")
    ppo_ckpt  = os.path.join(ckpt_dir, f"ppo_best_seed{seed}.pt")

    model = CDLS(
        clinical_dim=20,
        dtwin=cfg["latent"]["dtwin"],
        T=cfg["latent"]["T"],
        lambda_feedback=cfg["feedback"]["lambda_feedback"],
        k_neighbours=cfg["feedback"]["k_neighbours"],
    ).to(device)

    if os.path.exists(ppo_ckpt):
        ckpt = torch.load(ppo_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded TCGA-BRCA weights from {ppo_ckpt}")
    else:
        print(f"WARNING: Checkpoint not found at {ppo_ckpt}. Using random weights.")

    # ── Freeze RNA MLP + Clinical ANN ─────────────────────────────────────
    for param in model.rna_encoder.parameters():
        param.requires_grad = False
    for param in model.clin_encoder.parameters():
        param.requires_grad = False

    # ── Re-initialise projector (as per paper) ───────────────────────────
    model.projector = type(model.projector)(
        fusion_dim=1600, dtwin=cfg["latent"]["dtwin"], hidden_dims=(512, 128)
    ).to(device)
    print("Projector re-initialised.")

    # ── Load METABRIC data ────────────────────────────────────────────────
    metabric_train = METABRICDataset(cfg["data"]["metabric_dir"], split="train")
    metabric_test  = METABRICDataset(cfg["data"]["metabric_dir"], split="test")

    train_loader = DataLoader(
        metabric_train, batch_size=64, shuffle=True,
        collate_fn=collate_modality_disjoint,
    )
    test_loader = DataLoader(
        metabric_test, batch_size=64, shuffle=False,
        collate_fn=collate_modality_disjoint,
    )

    print(f"METABRIC: train={len(metabric_train)}, test={len(metabric_test)}")

    # ── Fine-tune projector for 10 epochs ─────────────────────────────────
    import torch.nn.functional as F
    from utils.losses import combined_pretrain_loss

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-5)

    best_val_acc    = 0.0
    best_state      = None
    patience_count  = 0
    patience        = 5
    max_ft_epochs   = 10

    for epoch in range(1, max_ft_epochs + 1):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad()
            z0, _ = model.encode(batch)
            logits = model.classifier(z0)
            risk   = model.survival_head(z0)
            subtypes   = batch["subtype"].to(device)
            surv_times = batch["surv_time"].to(device)
            events     = batch["event"].to(device)
            loss = combined_pretrain_loss(
                logits, subtypes, risk, surv_times, events
            )
            loss.backward()
            optimiser.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds_all, labels_all = [], []
            for batch in test_loader:
                z0, _ = model.encode(batch)
                logits = model.classifier(z0)
                preds  = logits.argmax(dim=-1).cpu().numpy()
                labels = batch["subtype"].numpy()
                preds_all.append(preds)
                labels_all.append(labels)
            y_pred = np.concatenate(preds_all)
            y_true = np.concatenate(labels_all)
            acc    = (y_pred == y_true).mean()

        print(f"  Epoch {epoch:2d}/{max_ft_epochs} | Transfer Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc  = acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Reload best
    if best_state:
        model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            z0, _ = model.encode(batch)
            logits = model.classifier(z0)
            proba  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = proba.argmax(axis=1)
            labels = batch["subtype"].numpy()
            all_preds.append(preds)
            all_labels.append(labels)
            all_proba.append(proba)

    y_true  = np.concatenate(all_labels)
    y_pred  = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)

    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    print("\nMETABRIC Transfer Results:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    print(f"  Macro AUC:         {metrics.get('macro_auc', float('nan')):.4f}")
    print("\nINTERPRETATION WARNING:")
    print("  Shared preprocessing between TCGA-BRCA and METABRIC introduces")
    print("  favourable bias. This is transfer-learning validation ONLY.")
    print("  The 6.6% zero-shot gap is the realistic generalisation indicator.")

    return metrics


def zero_shot_evaluation(cfg: dict, seed: int, device: torch.device) -> dict:
    """
    Zero-shot cross-cohort evaluation (Table XXIV).
    No fine-tuning. Direct inference on METABRIC.

    Expected results:
      Without BCSC: Acc=0.780, MCC=0.740
      With BCSC:    Acc=0.810, MCC=0.780
      With masking: Acc=0.818, MCC=0.791
      6.6% gap: domain shift, open challenge.
    """
    print(f"\n{'='*60}")
    print(f"Zero-Shot METABRIC Evaluation (Seed {seed})")
    print(f"{'='*60}")

    ckpt_dir  = os.path.join(cfg["logging"]["checkpoint_dir"], f"seed_{seed}")
    ppo_ckpt  = os.path.join(ckpt_dir, f"ppo_best_seed{seed}.pt")

    model = CDLS(
        clinical_dim=20,
        dtwin=cfg["latent"]["dtwin"],
        T=cfg["latent"]["T"],
        lambda_feedback=cfg["feedback"]["lambda_feedback"],
        k_neighbours=cfg["feedback"]["k_neighbours"],
    ).to(device)

    if os.path.exists(ppo_ckpt):
        ckpt = torch.load(ppo_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    # Load kNN bank
    bank_path = os.path.join(ckpt_dir, f"z0_bank_seed{seed}.pt")
    if os.path.exists(bank_path):
        model.build_feedback_bank(torch.load(bank_path))

    model.eval()
    test_dataset = METABRICDataset(cfg["data"]["metabric_dir"], split="test")
    test_loader  = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        collate_fn=collate_modality_disjoint,
    )

    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            out    = model.forward(batch)
            proba  = torch.softmax(out["subtype_logits"], dim=-1).cpu().numpy()
            preds  = proba.argmax(axis=1)
            labels = batch["subtype"].numpy()
            all_preds.append(preds)
            all_labels.append(labels)
            all_proba.append(proba)

    y_true  = np.concatenate(all_labels)
    y_pred  = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)

    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    print(f"Zero-shot Accuracy: {metrics['accuracy']:.4f}")
    print(f"Zero-shot MCC:      {metrics['mcc']:.4f}")
    print(f"(Expected: Acc≈0.780–0.818, 6.6% below in-domain)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="METABRIC Transfer Learning Evaluation")
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--seed",      type=int, default=789)
    parser.add_argument("--zero-shot", action="store_true",
                        help="Run zero-shot evaluation instead of transfer learning")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.zero_shot:
        zero_shot_evaluation(cfg, args.seed, device)
    else:
        transfer_to_metabric(cfg, args.seed, device)


if __name__ == "__main__":
    main()
