"""
models/cdls.py
Full CDLS model — Cross-Cohort Modality-Disjoint Latent Simulation.

Integrates all components:
  1. Modality encoders       (WSI, RNA, BCSC, Clinical)
  2. Modality absence tokens (Section III-I)
  3. Fusion projector        → z0 ∈ R^dtwin (Section III-L)
  4. Scenario conditioner    (optional; Section III-M)
  5. PPO policy + value networks (Section III-O)
  6. Twin-GRU transition model  (Section III-O.4)
  7. Closed-loop kNN feedback   (Section III-E, Eq. 2)
  8. PAM50 classifier           (Section IV-C)
  9. Cox survival head          (Section III-N; auxiliary only)

Full forward pass implements Algorithm 1 (inference mode):
  z0 ← LayerNorm(π(x_fusion))
  for τ = 0 to T-1:
      a_τ ~ π_θ(z_τ, t)
      ẑ_{τ+1} ← Twin-GRU(z_τ, a_τ)
      z_{τ+1} ← ẑ_{τ+1} + λ · (1/k) Σ_{i∈N(z_τ)} (z_i − z_τ)
  return z_T
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .encoders  import (
    WSIHierarchicalAttentionEncoder,
    RNAMLPEncoder,
    BCSCGRUEncoder,
    ClinicalANN,
)
from .projector import FusionProjector, ModalityAbsenceEncoder
from .twin_gru  import TwinGRUTransition
from .ppo       import PPOPolicy, PPOValueNetwork, ScenarioConditioner
from .feedback  import ClosedLoopFeedback
from .heads     import PAM50Classifier, CoxSurvivalHead


class CDLS(nn.Module):
    """
    Cross-Cohort Modality-Disjoint Latent Simulation framework.

    Args:
        clinical_dim:    Dimensionality of clinical feature vector (cohort-specific).
        dtwin:           Latent space dimension (default: 7, guided by d_id=6.3±0.4).
        T:               Refinement horizon (default: 5 optimisation steps).
        lambda_feedback: Closed-loop feedback strength (default: 0.1).
        k_neighbours:    kNN neighbours for feedback (default: 5).
        n_scenarios:     Number of scenario embeddings (default: 3).
    """

    def __init__(
        self,
        clinical_dim:    int   = 20,
        dtwin:           int   = 7,
        T:               int   = 5,
        lambda_feedback: float = 0.1,
        k_neighbours:    int   = 5,
        n_scenarios:     int   = 3,
    ):
        super().__init__()
        self.dtwin = dtwin
        self.T     = T

        # ── Modality Encoders ──────────────────────────────────────────────
        self.wsi_encoder  = WSIHierarchicalAttentionEncoder(input_dim=1024, d_model=768)
        self.rna_encoder  = RNAMLPEncoder(input_dim=20481, output_dim=512)
        self.bcsc_encoder = BCSCGRUEncoder(input_dim=7, hidden_dim=64)
        self.clin_encoder = ClinicalANN(input_dim=clinical_dim, hidden_dim=256)

        # ── Absence Tokens ─────────────────────────────────────────────────
        self.absence_encoder = ModalityAbsenceEncoder(wsi_dim=768, rna_dim=512)

        # ── Fusion Projector → z0 ─────────────────────────────────────────
        # Fusion dim: 768 (WSI) + 512 (RNA) + 64 (BCSC) + 256 (Clin) = 1600
        self.projector = FusionProjector(
            fusion_dim=1600, dtwin=dtwin, hidden_dims=(512, 128)
        )

        # ── Scenario Conditioner ──────────────────────────────────────────
        self.scenario_conditioner = ScenarioConditioner(
            dtwin=dtwin, n_scenarios=n_scenarios
        )

        # ── PPO Policy + Value ────────────────────────────────────────────
        self.policy = PPOPolicy(
            dtwin=dtwin, hidden_dims=(128, 64), scenario_dim=0
        )
        self.value  = PPOValueNetwork(
            dtwin=dtwin, hidden_dims=(128, 64), scenario_dim=0
        )

        # ── Twin-GRU Transition ───────────────────────────────────────────
        self.transition = TwinGRUTransition(dtwin=dtwin, hidden_size=128)

        # ── Closed-Loop Feedback ──────────────────────────────────────────
        self.feedback = ClosedLoopFeedback(
            lambda_feedback=lambda_feedback,
            k_neighbours=k_neighbours,
        )

        # ── Output Heads ──────────────────────────────────────────────────
        self.classifier   = PAM50Classifier(dtwin=dtwin, n_classes=5)
        self.survival_head = CoxSurvivalHead(dtwin=dtwin, hidden_dim=32)

    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, batch: Dict) -> torch.Tensor:
        """
        Encode all available modalities, apply absence tokens, fuse,
        and project to initial latent state z0.

        Returns: z0 ∈ R^{B × dtwin}
        """
        device = next(self.parameters()).device

        # ── WSI ──────────────────────────────────────────────────────────
        wsi_patches = batch["wsi"].to(device)         # (B, N, 1024)
        wsi_present = batch["wsi_present"].to(device)

        # Build attention mask: True for valid patches
        B, N, _ = wsi_patches.shape
        patch_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        # (For absent patients, entire patch tensor is zeros; mask is all-True
        #  but absence token will replace the output anyway.)

        h_wsi_raw, attn_weights = self.wsi_encoder(wsi_patches, patch_mask)
        # Shape: (B, 768)

        # ── RNA ───────────────────────────────────────────────────────────
        rna_present = batch["rna_present"].to(device)
        rna_input   = batch["rna"].to(device)              # (B, 20481)
        psi_rna_raw = self.rna_encoder(rna_input)          # (B, 512)

        # ── Apply absence tokens ──────────────────────────────────────────
        h_wsi, psi_rna = self.absence_encoder.apply_tokens(
            h_wsi_raw, wsi_present, psi_rna_raw, rna_present
        )

        # ── BCSC GRU ─────────────────────────────────────────────────────
        bcsc_present = batch["bcsc_present"].to(device)
        bcsc_seq     = batch["bcsc"].to(device)            # (B, T_visits, 7)
        bcsc_lens    = batch["bcsc_len"].to(device)
        e_long = self.bcsc_encoder(bcsc_seq, bcsc_lens, bcsc_present)  # (B, 64)

        # ── Clinical ANN ──────────────────────────────────────────────────
        clin_input = batch["clinical"].to(device)          # (B, C)
        h_clin     = self.clin_encoder(clin_input)         # (B, 256)

        # ── Fusion ────────────────────────────────────────────────────────
        x_fusion = torch.cat([h_wsi, psi_rna, e_long, h_clin], dim=-1)  # (B, 1600)
        z0       = self.projector(x_fusion)                               # (B, dtwin)

        return z0, attn_weights

    # ─────────────────────────────────────────────────────────────────────────

    def rollout(
        self,
        z0:           torch.Tensor,             # (B, dtwin)
        scenario_idx: Optional[int] = None,
        t_custom:     Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict:
        """
        Execute T-step latent refinement trajectory (Algorithm 1, lines 4–12).

        Args:
            z0:                Initial latent state
            scenario_idx:      Optional scenario conditioning index
            t_custom:          Optional custom scenario embedding
            return_trajectory: If True, return all intermediate states

        Returns dict with:
            z_T:         Final latent state (B, dtwin)
            actions:     List of actions per step
            log_probs:   List of log-probabilities per step
            values:      List of value estimates per step
            trajectory:  List of z_τ (if return_trajectory=True)
            attn_weights_per_step: attention weights are encoder-level only
        """
        # Optional scenario conditioning on z0
        z = self.scenario_conditioner.condition_z0(z0, scenario_idx, t_custom)

        h_gru    = None
        actions_list   = []
        log_probs_list = []
        values_list    = []
        traj           = [z.detach()] if return_trajectory else []

        for tau in range(self.T):
            # PPO policy: sample action a_τ ~ π_θ(z_τ)
            action, log_prob = self.policy.sample(z)
            value            = self.value(z)

            # Twin-GRU transition (Eq. 20)
            z_hat, h_gru = self.transition(z, action, h_gru)

            # Closed-loop latent feedback (Eq. 2)
            z_next = self.feedback.apply(z_hat)

            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)

            if return_trajectory:
                traj.append(z_next.detach())

            z = z_next

        # Bootstrap value for GAE
        v_terminal = self.value(z)
        values_list.append(v_terminal)

        out = {
            "z_T":         z,
            "actions":     actions_list,
            "log_probs":   log_probs_list,
            "values":      values_list,
            "h_gru_final": h_gru,
        }
        if return_trajectory:
            out["trajectory"] = traj
        return out

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        batch:        Dict,
        scenario_idx: Optional[int] = None,
        t_custom:     Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict:
        """
        Full forward pass: encode → rollout → classify → survival.

        Returns:
            subtype_logits:  (B, 5)
            risk_scores:     (B,)
            z_0:             (B, dtwin)
            z_T:             (B, dtwin)
            attn_weights:    (B, N_patches)   WSI attention for visualisation
            trajectory:      list of tensors (if requested)
            actions/log_probs/values: PPO rollout data
        """
        z0, attn_weights = self.encode(batch)
        rollout_out      = self.rollout(z0, scenario_idx, t_custom, return_trajectory)

        z_T            = rollout_out["z_T"]
        subtype_logits = self.classifier(z_T)
        risk_scores    = self.survival_head(z_T)

        return {
            "subtype_logits": subtype_logits,
            "risk_scores":    risk_scores,
            "z_0":            z0,
            "z_T":            z_T,
            "attn_weights":   attn_weights,
            **rollout_out,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        z_T:         torch.Tensor,    # (B, dtwin)
        subtypes:    torch.Tensor,    # (B,)
        surv_times:  torch.Tensor,    # (B,)
        events:      torch.Tensor,    # (B,)
        last_action: torch.Tensor,    # (B, dtwin)
        alpha_rew:   float = 1.0,
        beta_rew:    float = 0.5,
        gamma_rew:   float = 1e-3,
    ) -> torch.Tensor:
        """
        Terminal reward function (Eq. 16).
        R = − ( α·L_CE(z_T) + β·L_Cox(z_T) + γ·||a_{T-1}||^2 )

        Only applied at terminal step τ = T-1;
        intermediate rewards are identically zero.
        """
        import torch.nn.functional as F
        from .heads import cox_partial_likelihood_loss

        logits    = self.classifier(z_T)
        L_CE      = F.cross_entropy(logits, subtypes)
        risk      = self.survival_head(z_T)
        L_Cox     = cox_partial_likelihood_loss(risk, surv_times, events)
        action_norm = (last_action ** 2).sum(dim=-1).mean()

        reward = -(alpha_rew * L_CE + beta_rew * L_Cox + gamma_rew * action_norm)
        return reward

    # ─────────────────────────────────────────────────────────────────────────

    def build_feedback_bank(self, train_z0: torch.Tensor):
        """
        Build offline kNN index from training-set z0 embeddings.
        Algorithm 1, line 3.
        """
        self.feedback.build_bank(train_z0)

    def get_per_step_entropy(self, z_trajectory: List[torch.Tensor]) -> List[float]:
        """
        Compute per-step action entropy for uncertainty tracking (Panel P4).
        H(a_τ) = −E[log π_θ(a_τ | z_τ)]
        """
        entropies = []
        for z in z_trajectory[:-1]:  # exclude z_T
            mu, std = self.policy(z)
            from torch.distributions import Normal
            dist    = Normal(mu, std)
            H       = dist.entropy().sum(dim=-1).mean().item()
            entropies.append(H)
        return entropies

    def set_eval_mode(self):
        """Set to evaluation mode; feedback bank must be built first."""
        self.eval()
        return self

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter counts per component."""
        def count(m): return sum(p.numel() for p in m.parameters())
        return {
            "wsi_encoder":    count(self.wsi_encoder),        # ~88M ConvNeXt-Base
            "rna_encoder":    count(self.rna_encoder),
            "bcsc_encoder":   count(self.bcsc_encoder),
            "clin_encoder":   count(self.clin_encoder),
            "projector":      count(self.projector),
            "ppo_policy":     count(self.policy),             # ~0.6M
            "ppo_value":      count(self.value),
            "twin_gru":       count(self.transition),          # ~0.7M
            "classifier":     count(self.classifier),
            "survival_head":  count(self.survival_head),
            "total":          count(self),
        }
