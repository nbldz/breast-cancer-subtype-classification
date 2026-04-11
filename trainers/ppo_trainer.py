"""
trainers/ppo_trainer.py
Stage 2: PPO trajectory optimisation for latent refinement.

Implements Algorithm 1 (full training loop) from the paper.
PPO is selected for trajectory geometry properties (Section III-O.5):
  - Path KL divergence: 1.73 (PPO) vs 1.14 (SAC)
  - Latent coverage:    0.71  (PPO) vs 0.58 (SAC)
  - Action entropy:     2.47  (PPO) vs 1.89 (SAC)

PPO hyperparameters (Section IV-B):
  - Policy LR:       3e-4
  - Clip ε:          0.2  (observed rate: 18.3%)
  - GAE λ:           0.95
  - K update epochs: 10 per rollout
  - Empirical KL:    0.015

Training cost: +4.8 h/seed overhead vs. IGR (+1.1 h); justified by
geometry, not marginal accuracy gain (+0.004 over SAC, within σ=0.034).
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from models.cdls   import CDLS
from models.ppo    import ppo_clip_loss, GAEComputer
from utils.losses  import ppo_terminal_reward, cox_partial_likelihood_loss, cross_entropy_loss
from utils.metrics import compute_all_metrics, aggregate_seeds, format_results_table


class PPOTrainer:
    """
    PPO-based latent trajectory trainer.

    Per-patient rollout:
      z_0 ← encode(x)
      for τ = 0 to T-1:
          a_τ ~ π_θ(z_τ)
          ẑ_{τ+1} ← Twin-GRU(z_τ, a_τ)       [Eq. 20]
          z_{τ+1} ← ẑ_{τ+1} + feedback(z_τ)   [Eq. 2]
      R_{T-1} ← terminal reward [Eq. 16]
      Update θ, ϕ via PPO [Eq. 15]
    """

    def __init__(
        self,
        model:        CDLS,
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

        ppo_cfg = cfg["ppo"]

        # Only PPO policy + value + Twin-GRU are updated in Stage 2
        # Encoders and projector keep weights from pretrain
        self.optimiser = torch.optim.AdamW(
            list(model.policy.parameters()) +
            list(model.value.parameters()) +
            list(model.transition.parameters()) +
            list(model.classifier.parameters()) +
            list(model.survival_head.parameters()),
            lr=ppo_cfg["policy_lr"],
            weight_decay=cfg["training"]["weight_decay_other"],
        )

        self.clip_eps      = ppo_cfg["clip_eps"]
        self.k_epochs      = ppo_cfg["update_epochs"]
        self.alpha_rew     = ppo_cfg["alpha_rew"]
        self.beta_rew      = ppo_cfg["beta_rew"]
        self.gamma_rew     = ppo_cfg["gamma_rew"]
        self.T             = cfg["latent"]["T"]

        self.gae = GAEComputer(gamma=0.99, gae_lambda=ppo_cfg["gae_lambda"])

        self.max_epochs     = cfg["training"]["max_epochs"]
        self.patience       = cfg["training"]["patience"]
        self.best_val_ba    = 0.0
        self.patience_count = 0
        self.best_epoch     = 0

        # Statistics tracking
        self.clip_rates    = []
        self.kl_estimates  = []
        self.entropy_steps = []

    # ─────────────────────────────────────────────────────────────────────────

    def collect_rollouts(self) -> Dict:
        """
        Collect PPO rollout data from one pass over the training set.
        Returns buffers: states, actions, log_probs_old, advantages, returns, etc.
        """
        self.model.eval()   # Collect with frozen BN stats

        buf_z0          = []
        buf_actions     = []        # (B, T, dtwin) list → stacked
        buf_log_probs   = []        # (B, T)
        buf_values      = []        # (B, T+1)
        buf_rewards     = []        # (B, T) — only last step non-zero
        buf_subtypes    = []
        buf_surv_times  = []
        buf_events      = []

        with torch.no_grad():
            for batch in self.train_loader:
                subtypes   = batch["subtype"].to(self.device)
                surv_times = batch["surv_time"].to(self.device)
                events     = batch["event"].to(self.device)

                z0, _ = self.model.encode(batch)
                buf_z0.append(z0)

                # Run T-step trajectory
                z        = z0
                h_gru    = None
                actions_ep   = []
                log_probs_ep = []
                values_ep    = []

                for tau in range(self.T):
                    action, log_prob = self.model.policy.sample(z)
                    value            = self.model.value(z).squeeze(-1)

                    z_hat, h_gru = self.model.transition(z, action, h_gru)
                    z_next       = self.model.feedback.apply(z_hat)

                    actions_ep.append(action)
                    log_probs_ep.append(log_prob)
                    values_ep.append(value)
                    z = z_next

                # Bootstrap value at z_T
                v_T = self.model.value(z).squeeze(-1)
                values_ep.append(v_T)

                # Terminal reward (only at τ = T-1; intermediates are zero)
                logits     = self.model.classifier(z)
                risk       = self.model.survival_head(z)
                reward_val = ppo_terminal_reward(
                    logits, subtypes, risk, surv_times, events,
                    actions_ep[-1],
                    alpha_rew=self.alpha_rew,
                    beta_rew=self.beta_rew,
                    gamma_rew=self.gamma_rew,
                )
                # Broadcast scalar reward to all batch elements
                rewards_ep = torch.zeros(len(subtypes), self.T, device=self.device)
                rewards_ep[:, -1] = reward_val.detach()

                buf_actions.append(torch.stack(actions_ep, dim=1))    # (B, T, d)
                buf_log_probs.append(torch.stack(log_probs_ep, dim=1))# (B, T)
                buf_values.append(torch.stack(values_ep, dim=1))      # (B, T+1)
                buf_rewards.append(rewards_ep)
                buf_subtypes.append(subtypes)
                buf_surv_times.append(surv_times)
                buf_events.append(events)

        # Concatenate across batches
        z0_all       = torch.cat(buf_z0,        dim=0)
        actions_all  = torch.cat(buf_actions,   dim=0)
        lp_old_all   = torch.cat(buf_log_probs, dim=0)
        values_all   = torch.cat(buf_values,    dim=0)
        rewards_all  = torch.cat(buf_rewards,   dim=0)
        subtypes_all = torch.cat(buf_subtypes,  dim=0)
        st_all       = torch.cat(buf_surv_times, dim=0)
        ev_all       = torch.cat(buf_events,    dim=0)

        # Compute GAE advantages
        dones = torch.zeros_like(rewards_all)   # no episode termination
        advantages, returns = self.gae.compute(rewards_all, values_all, dones)

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return {
            "z0":          z0_all,
            "actions":     actions_all,
            "log_probs_old": lp_old_all,
            "advantages":  advantages,
            "returns":     returns,
            "subtypes":    subtypes_all,
            "surv_times":  st_all,
            "events":      ev_all,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def update_policy(self, rollouts: Dict) -> Dict:
        """
        K-epoch PPO update on collected rollouts.
        Returns statistics (clip rate, KL, entropy).
        """
        self.model.train()

        z0          = rollouts["z0"]
        actions_old = rollouts["actions"]           # (N, T, d)
        lp_old      = rollouts["log_probs_old"]     # (N, T)
        advantages  = rollouts["advantages"]         # (N, T)
        returns     = rollouts["returns"]            # (N, T)
        subtypes    = rollouts["subtypes"]
        surv_times  = rollouts["surv_times"]
        events      = rollouts["events"]

        N = z0.shape[0]
        clip_rates_k = []
        kl_k         = []

        for k_epoch in range(self.k_epochs):
            # Re-run trajectory to get updated log-probs and values
            z = z0.detach()
            h_gru = None
            log_probs_new = []
            values_new    = []
            entropies     = []
            z_T           = None

            for tau in range(self.T):
                action_old  = actions_old[:, tau, :].detach()
                lp_new, ent = self.model.policy.evaluate(z, action_old)
                v_new       = self.model.value(z).squeeze(-1)

                z_hat, h_gru = self.model.transition(z, action_old, h_gru)
                z_next       = self.model.feedback.apply(z_hat)

                log_probs_new.append(lp_new)
                values_new.append(v_new)
                entropies.append(ent)
                z = z_next

            v_T_new = self.model.value(z).squeeze(-1)
            values_new.append(v_T_new)
            z_T = z

            log_probs_new = torch.stack(log_probs_new, dim=1)  # (N, T)
            values_new    = torch.stack(values_new,    dim=1)  # (N, T+1)
            entropies     = torch.stack(entropies,     dim=1)  # (N, T)

            # ── PPO Clip Loss (Eq. 15) ────────────────────────────────────
            lp_new_flat  = log_probs_new.reshape(-1)
            lp_old_flat  = lp_old.reshape(-1).detach()
            adv_flat     = advantages.reshape(-1).detach()

            ppo_loss, clip_rate = ppo_clip_loss(lp_new_flat, lp_old_flat, adv_flat, self.clip_eps)

            # ── Value Loss ────────────────────────────────────────────────
            val_loss = 0.5 * ((values_new[:, :-1] - returns.detach()) ** 2).mean()

            # ── Entropy Bonus ─────────────────────────────────────────────
            entropy_loss = -0.01 * entropies.mean()

            # ── Supervised auxiliary (on z_T) ─────────────────────────────
            logits_T     = self.model.classifier(z_T)
            sup_loss     = cross_entropy_loss(logits_T, subtypes)
            risk_T       = self.model.survival_head(z_T)
            cox_loss     = cox_partial_likelihood_loss(risk_T, surv_times, events)
            aux_loss     = self.alpha_rew * sup_loss + self.beta_rew * cox_loss

            # ── Total Loss ────────────────────────────────────────────────
            total_loss = ppo_loss + 0.5 * val_loss + entropy_loss + 0.1 * aux_loss

            self.optimiser.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimiser.step()

            # ── Track KL divergence ───────────────────────────────────────
            with torch.no_grad():
                kl = (lp_old_flat - lp_new_flat).mean().item()
                kl_k.append(abs(kl))

            clip_rates_k.append(clip_rate.item())

        return {
            "clip_rate":      float(np.mean(clip_rates_k)),
            "approx_kl":      float(np.mean(kl_k)),
            "entropy":        float(entropies.mean().item()),
        }

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set using full T-step trajectory."""
        self.model.eval()
        all_preds, all_labels, all_proba = [], [], []
        all_risk, all_surv, all_events   = [], [], []

        for batch in self.val_loader:
            subtypes   = batch["subtype"].to(self.device)
            surv_times = batch["surv_time"].to(self.device)
            events     = batch["event"].to(self.device)

            out    = self.model.forward(batch)
            proba  = torch.softmax(out["subtype_logits"], dim=-1).cpu().numpy()
            preds  = proba.argmax(axis=1)
            labels = subtypes.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
            all_proba.append(proba)
            all_risk.append(out["risk_scores"].cpu().numpy())
            all_surv.append(surv_times.cpu().numpy())
            all_events.append(events.cpu().numpy())

        y_true   = np.concatenate(all_labels)
        y_pred   = np.concatenate(all_preds)
        y_proba  = np.concatenate(all_proba)
        risk     = np.concatenate(all_risk)
        surv     = np.concatenate(all_surv)
        evs      = np.concatenate(all_events)

        metrics = compute_all_metrics(y_true, y_pred, y_proba, surv, evs, risk)
        return metrics

    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> str:
        """
        Full PPO training loop.
        Returns path to best checkpoint.
        """
        print(f"\n{'='*60}")
        print(f"Stage 2 PPO Training (Seed {self.seed})")
        print(f"{'='*60}")

        best_ckpt = os.path.join(
            self.output_dir, f"ppo_best_seed{self.seed}.pt"
        )

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            # Collect rollouts
            rollouts = self.collect_rollouts()

            # PPO update
            ppo_stats = self.update_policy(rollouts)

            # Evaluate
            val_metrics = self.evaluate()
            val_ba  = val_metrics["balanced_accuracy"]
            val_mcc = val_metrics["mcc"]
            val_acc = val_metrics["accuracy"]

            self.clip_rates.append(ppo_stats["clip_rate"])
            self.kl_estimates.append(ppo_stats["approx_kl"])

            print(
                f"  Epoch {epoch:3d}/{self.max_epochs} | "
                f"Val BA: {val_ba:.4f} | MCC: {val_mcc:.4f} | Acc: {val_acc:.4f} | "
                f"Clip: {ppo_stats['clip_rate']*100:.1f}% | "
                f"KL: {ppo_stats['approx_kl']:.4f} | "
                f"H: {ppo_stats['entropy']:.3f} | "
                f"Time: {time.time()-t0:.1f}s"
            )

            if val_ba > self.best_val_ba:
                self.best_val_ba    = val_ba
                self.best_epoch     = epoch
                self.patience_count = 0
                torch.save({
                    "epoch":        epoch,
                    "model_state":  self.model.state_dict(),
                    "opt_state":    self.optimiser.state_dict(),
                    "val_metrics":  val_metrics,
                    "ppo_stats":    ppo_stats,
                    "seed":         self.seed,
                    "clip_rates":   self.clip_rates,
                    "kl_estimates": self.kl_estimates,
                }, best_ckpt)
                print(f"    → Saved best (val_BA={val_ba:.4f})")
            else:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch: {self.best_epoch})")
                    break

            # Warn if KL divergence exceeds threshold (paper: empirical KL = 0.015)
            if ppo_stats["approx_kl"] > 0.05:
                print(f"    WARNING: high KL={ppo_stats['approx_kl']:.4f} "
                      f"(paper target: ~0.015)")

        mean_clip = np.mean(self.clip_rates) * 100
        print(f"\nPPO Training complete.")
        print(f"  Best epoch:   {self.best_epoch}")
        print(f"  Best val BA:  {self.best_val_ba:.4f}")
        print(f"  Mean clip %:  {mean_clip:.1f}% (paper: 18.3%)")
        return best_ckpt

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_test(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        """
        Final evaluation on held-out test set.
        Optionally loads from checkpoint.
        """
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])

        self.model.eval()
        all_preds, all_labels, all_proba = [], [], []
        all_risk, all_surv, all_events   = [], [], []
        all_z_T = []

        for batch in test_loader:
            subtypes   = batch["subtype"].to(self.device)
            surv_times = batch["surv_time"].to(self.device)
            events     = batch["event"].to(self.device)

            out = self.model.forward(batch)

            proba  = torch.softmax(out["subtype_logits"], dim=-1).cpu().numpy()
            preds  = proba.argmax(axis=1)
            labels = subtypes.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
            all_proba.append(proba)
            all_risk.append(out["risk_scores"].cpu().numpy())
            all_surv.append(surv_times.cpu().numpy())
            all_events.append(events.cpu().numpy())
            all_z_T.append(out["z_T"].cpu())

        y_true  = np.concatenate(all_labels)
        y_pred  = np.concatenate(all_preds)
        y_proba = np.concatenate(all_proba)
        risk    = np.concatenate(all_risk)
        surv    = np.concatenate(all_surv)
        evs     = np.concatenate(all_events)
        z_T_all = torch.cat(all_z_T, dim=0)

        metrics = compute_all_metrics(y_true, y_pred, y_proba, surv, evs, risk)

        print("\n" + "="*60)
        print(f"Test Set Results (Seed {self.seed})")
        print("="*60)
        for key in ["balanced_accuracy", "mcc", "accuracy", "macro_f1",
                    "weighted_f1", "macro_auc", "ece", "brier_score", "c_index"]:
            if key in metrics:
                print(f"  {key:<25}: {metrics[key]:.4f}")

        return metrics, y_true, y_pred, y_proba, z_T_all
