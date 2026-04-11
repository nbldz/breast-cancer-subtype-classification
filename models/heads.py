"""
models/heads.py
Output heads for CDLS.

1. PAM50SoftmaxClassifier   — primary output (5-class softmax)
2. CoxSurvivalHead          — auxiliary latent regularisation ONLY
   (NOT clinical prediction; C-index reported as auxiliary metric)

Section III-N, Eq. 14:
    L_Cox = - Σ_{i: e_i=1} [ η_i − log( Σ_{j: t_j ≥ t_i} exp(η_j) ) ]
    η_i = g(z_T^(i)) ∈ R (scalar risk score)

Section IV-B:
    Auxiliary survival head: z_T → [32] → scalar risk score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PAM50Classifier(nn.Module):
    """
    Calibrated 5-class PAM50 softmax classifier.
    Applied to z_T ∈ R^dtwin after trajectory refinement.

    Includes a temperature-scaling calibration layer.
    ECE = 0.031 ± 0.005 (Table VI).
    """

    def __init__(self, dtwin: int = 7, n_classes: int = 5):
        super().__init__()
        self.fc          = nn.Linear(dtwin, n_classes)
        # Learnable temperature for calibration (initialised to 1.0)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, z_T: torch.Tensor) -> torch.Tensor:
        """
        z_T: (B, dtwin) → logits: (B, 5)
        Calibrated by temperature scaling.
        """
        logits = self.fc(z_T)
        return logits / self.temperature.clamp(min=0.1)

    def predict_proba(self, z_T: torch.Tensor) -> torch.Tensor:
        """Returns calibrated softmax probabilities (B, 5)."""
        return F.softmax(self.forward(z_T), dim=-1)


class CoxSurvivalHead(nn.Module):
    """
    Auxiliary survival head: used for latent regularisation ONLY.
    NOT intended for clinical prediction.

    Section III-N, Section IV-B:
        z_T → Linear(dtwin, 32) → ReLU → Linear(32, 1) → scalar risk η

    Section VI-E: βrew = 0.5 (reward coefficient); removing Cox (βrew=0)
    changes accuracy by −0.008 < σ, confirming it provides latent
    regularisation only.
    """

    def __init__(self, dtwin: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dtwin, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_T: torch.Tensor) -> torch.Tensor:
        """z_T: (B, dtwin) → risk score η: (B,)"""
        return self.net(z_T).squeeze(-1)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,   # (B,) η_i = g(z_T^(i))
    surv_times:  torch.Tensor,   # (B,) t_i
    events:      torch.Tensor,   # (B,) e_i ∈ {0, 1}
) -> torch.Tensor:
    """
    Standard Cox partial log-likelihood (Eq. 14).

    L_Cox = − Σ_{i: e_i=1} [ η_i − log( Σ_{j: t_j ≥ t_i} exp(η_j) ) ]

    Numerically stable via log-sum-exp trick.
    Mini-batches are event-enforced (≥10 uncensored events per batch)
    to stabilise under 13.2% event rate (Table XVIII).
    """
    n = len(risk_scores)
    if events.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)

    # Sort by descending survival time for risk-set computation
    order      = torch.argsort(surv_times, descending=True)
    risks_ord  = risk_scores[order]
    events_ord = events[order]

    # log(Σ_{j: t_j ≥ t_i} exp(η_j)) via cumulative log-sum-exp
    # Numerically stable: subtract max before exp
    max_risk = risks_ord.max().detach()
    log_cumsum = torch.log(
        torch.cumsum(torch.exp(risks_ord - max_risk), dim=0)
    ) + max_risk

    # Only sum over uncensored events
    losses = (risks_ord - log_cumsum) * events_ord
    return -losses.sum() / (events_ord.sum() + 1e-8)


class EventEnforcedSampler:
    """
    Event-enforced mini-batch sampler for Cox loss stability.
    Ensures ≥ min_events uncensored events per mini-batch.
    Section III-N, Section VI-C.

    Natural sampling with batch=256 leads to instability in 3/4 seeds.
    Event-enforced sampling stabilises without biasing survival metrics.
    """

    def __init__(self, events: torch.Tensor, min_events: int = 10, batch_size: int = 256):
        self.events     = events
        self.min_events = min_events
        self.batch_size = batch_size

        self.event_idx   = (events == 1).nonzero(as_tuple=True)[0]
        self.censored_idx = (events == 0).nonzero(as_tuple=True)[0]

    def sample_batch(self) -> torch.Tensor:
        """
        Returns indices for one event-enforced mini-batch.
        """
        # Guarantee min_events uncensored patients
        n_events = min(self.min_events, len(self.event_idx))
        event_sample = self.event_idx[
            torch.randperm(len(self.event_idx))[:n_events]
        ]

        # Fill remainder with random patients
        n_fill  = self.batch_size - n_events
        all_idx = torch.randperm(len(self.events))[:n_fill]
        return torch.cat([event_sample, all_idx])
