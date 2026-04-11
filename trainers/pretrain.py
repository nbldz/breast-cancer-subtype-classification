"""
trainers/pretrain.py
Stage 1: Pre-train encoders + projector with supervised CE + Cox loss.

Algorithm 1, line 2:
    "Pre-train projector + Twin-GRU: α·L_CE + β·L_Cox;
     event-enforced batches (≥10 uncensored events per batch)"

After pretraining, the offline kNN index is built from training z0 embeddings
(Algorithm 1, line 3) before PPO training begins.

Training details (Section IV-B):
  - AdamW, weight_decay: 1e-4 (ConvNeXt-Base), 1e-5 (others)
  - Up to 50 epochs, patience=10
  - Batch size: 256, event-enforced (≥10 uncensored)
  - α=1.0 (CE), β=0.5 (Cox)
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, Optional

from models.cdls  import CDLS
from utils.losses import combined_pretrain_loss
from utils.metrics import compute_all_metrics, format_results_table


class PreTrainer:
    """
    Stage-1 trainer: jointly optimise modality encoders + fusion projector
    using supervised classification and auxiliary Cox survival loss.
    """

    def __init__(
        self,
        model:       CDLS,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Dict,
        device:       torch.device,
        output_dir:   str,
        seed:         int,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.output_dir   = output_dir
        self.seed         = seed

        os.makedirs(output_dir, exist_ok=True)

        # ── Optimiser ──────────────────────────────────────────────────────
        # ConvNeXt-Base uses higher weight decay (Table IV-B)
        wsi_params   = list(model.wsi_encoder.parameters())
        other_params = [
            p for n, p in model.named_parameters()
            if not any(id(p) == id(wp) for wp in wsi_params)
        ]
        self.optimiser = torch.optim.AdamW([
            {"params": wsi_params,   "weight_decay": cfg["training"]["weight_decay_convnext"]},
            {"params": other_params, "weight_decay": cfg["training"]["weight_decay_other"]},
        ], lr=1e-4)

        self.alpha = cfg["training"]["alpha_obj"]   # 1.0
        self.beta  = cfg["training"]["beta_obj"]    # 0.5
        self.max_epochs = cfg["training"]["max_epochs"]
        self.patience   = cfg["training"]["patience"]

        self.best_val_loss  = float("inf")
        self.patience_count = 0
        self.best_epoch     = 0
        self.train_losses   = []
        self.val_losses     = []

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            self.optimiser.zero_grad()

            # Forward pass (no PPO rollout in pretrain — only encode + static heads)
            z0, _ = self.model.encode(batch)

            # Use z0 directly for pretraining (before trajectory refinement)
            subtype_logits = self.model.classifier(z0)
            risk_scores    = self.model.survival_head(z0)

            subtypes   = batch["subtype"].to(self.device)
            surv_times = batch["surv_time"].to(self.device)
            events     = batch["event"].to(self.device)

            loss = combined_pretrain_loss(
                subtype_logits, subtypes,
                risk_scores, surv_times, events,
                alpha=self.alpha, beta=self.beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        all_preds, all_labels, all_proba = [], [], []
        total_loss = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            z0, _ = self.model.encode(batch)
            logits = self.model.classifier(z0)
            risk   = self.model.survival_head(z0)

            subtypes   = batch["subtype"].to(self.device)
            surv_times = batch["surv_time"].to(self.device)
            events     = batch["event"].to(self.device)

            loss = combined_pretrain_loss(
                logits, subtypes, risk, surv_times, events,
                alpha=self.alpha, beta=self.beta,
            )
            total_loss += loss.item()
            n_batches  += 1

            proba  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = proba.argmax(axis=1)
            labels = subtypes.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
            all_proba.append(proba)

        y_true  = np.concatenate(all_labels)
        y_pred  = np.concatenate(all_preds)
        y_proba = np.concatenate(all_proba)

        metrics = compute_all_metrics(y_true, y_pred, y_proba)
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics

    def train(self) -> str:
        """
        Full pretraining loop with early stopping.
        Returns path to best checkpoint.
        """
        print(f"\n{'='*60}")
        print(f"Stage 1 Pretraining (Seed {self.seed})")
        print(f"{'='*60}")

        best_ckpt = os.path.join(self.output_dir, f"pretrain_best_seed{self.seed}.pt")

        for epoch in range(1, self.max_epochs + 1):
            t0         = time.time()
            train_loss = self.train_one_epoch()
            val_metrics = self.evaluate()
            val_loss    = val_metrics["val_loss"]
            val_acc     = val_metrics["accuracy"]
            val_ba      = val_metrics["balanced_accuracy"]

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"  Epoch {epoch:3d}/{self.max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val BA: {val_ba:.4f} | "
                  f"Time: {time.time()-t0:.1f}s")

            if val_loss < self.best_val_loss:
                self.best_val_loss  = val_loss
                self.best_epoch     = epoch
                self.patience_count = 0
                torch.save({
                    "epoch":          epoch,
                    "model_state":    self.model.state_dict(),
                    "optimiser_state": self.optimiser.state_dict(),
                    "val_metrics":    val_metrics,
                    "seed":           self.seed,
                }, best_ckpt)
                print(f"    → Saved best checkpoint (val_loss={val_loss:.4f})")
            else:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best: epoch {self.best_epoch})")
                    break

        print(f"Pretraining complete. Best epoch: {self.best_epoch}, "
              f"Best val loss: {self.best_val_loss:.4f}")
        return best_ckpt

    @torch.no_grad()
    def build_knn_bank(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Build offline kNN index from training-set z0 embeddings.
        Algorithm 1, line 3.
        """
        self.model.eval()
        all_z0 = []
        for batch in train_loader:
            z0, _ = self.model.encode(batch)
            all_z0.append(z0.cpu())
        z0_bank = torch.cat(all_z0, dim=0)
        self.model.build_feedback_bank(z0_bank)
        print(f"kNN bank built: {z0_bank.shape} training embeddings")
        return z0_bank
