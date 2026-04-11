"""
utils/losses.py
Loss functions used in CDLS training.

Training objective (Eq. 1):
    min_{θ,ϕ} E[ α·L_CE(f_{θ,ϕ}(x), y_sub) + β·L_Cox(f_{θ,ϕ}(x), t_i, e_i) ]
    α = 1.0, β = 0.5

PPO reward (Eq. 16):
    R = −( α_rew·L_CE(z_T) + β_rew·L_Cox(z_T) + γ_rew·||a_{T-1}||² )
    α_rew = 1.0, β_rew = 0.5, γ_rew = 1e-3
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy over PAM50 subtypes."""
    return F.cross_entropy(logits, targets)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    surv_times:  torch.Tensor,
    events:      torch.Tensor,
) -> torch.Tensor:
    """
    Cox partial log-likelihood (Eq. 14).
    Numerically stable implementation.
    """
    if events.sum() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    order      = torch.argsort(surv_times, descending=True)
    risks_ord  = risk_scores[order]
    events_ord = events[order]

    max_risk   = risks_ord.max().detach()
    log_cumsum = torch.log(
        torch.cumsum(torch.exp(risks_ord - max_risk), dim=0) + 1e-8
    ) + max_risk

    return -((risks_ord - log_cumsum) * events_ord).sum() / (events_ord.sum() + 1e-8)


def combined_pretrain_loss(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    risk_scores: torch.Tensor,
    surv_times:  torch.Tensor,
    events:      torch.Tensor,
    alpha: float = 1.0,
    beta:  float = 0.5,
) -> torch.Tensor:
    """Combined pretraining loss: α·L_CE + β·L_Cox (Eq. 1)."""
    L_CE  = cross_entropy_loss(logits, targets)
    L_Cox = cox_partial_likelihood_loss(risk_scores, surv_times, events)
    return alpha * L_CE + beta * L_Cox


def ppo_terminal_reward(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    risk_scores: torch.Tensor,
    surv_times:  torch.Tensor,
    events:      torch.Tensor,
    last_action: torch.Tensor,
    alpha_rew:   float = 1.0,
    beta_rew:    float = 0.5,
    gamma_rew:   float = 1e-3,
) -> torch.Tensor:
    """PPO terminal reward (Eq. 16). Scalar reward."""
    L_CE  = cross_entropy_loss(logits, targets)
    L_Cox = cox_partial_likelihood_loss(risk_scores, surv_times, events)
    a_pen = (last_action ** 2).sum(dim=-1).mean()
    return -(alpha_rew * L_CE + beta_rew * L_Cox + gamma_rew * a_pen)
