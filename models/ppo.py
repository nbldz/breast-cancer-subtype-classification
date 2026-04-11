"""
models/ppo.py
PPO Policy and Value Networks for CDLS latent trajectory optimisation.

Section III-O, Eqs. 11–16.

PPO is selected for its trajectory geometry properties (broader latent coverage,
higher path diversity) rather than marginal accuracy gains (Table V):
  - Path KL divergence: 1.73 (PPO) vs 1.14 (SAC)
  - Latent coverage:    0.71 (PPO) vs 0.58 (SAC)
  - Action entropy:     2.47 (PPO) vs 1.89 (SAC)

PPO hyperparameters:
  - Policy LR:       3e-4
  - Clip ε:          0.2 (clipping rate: 18.3%)
  - GAE λ:           0.95
  - K update epochs: 10 per rollout
  - Empirical KL:    0.015

Scenario conditioning (Eqs. 11–12):
  a_τ ~ π_θ(z_τ, t) = N(μ_θ(z_τ, t), Σ_θ(z_τ, t))
  z'_0 = z_0 + W_t · t,   W_t ∈ R^{7×K}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple


class PPOPolicy(nn.Module):
    """
    Stochastic Gaussian policy network.
    Optionally conditioned on scenario embedding t ∈ R^K.

    Architecture (Section IV-B):
        Two layers [dtwin (+K) → 128 → 64] with tanh activations
        Gaussian output head: μ_θ and log σ_θ ∈ R^dtwin
        Actions clipped to [-1, 1]^dtwin (PPO standard)
    """

    def __init__(
        self,
        dtwin:       int = 7,
        hidden_dims: tuple = (128, 64),
        scenario_dim: int = 0,
    ):
        super().__init__()
        input_dim = dtwin + scenario_dim

        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev    = h

        self.net     = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, dtwin)
        self.log_std = nn.Parameter(torch.zeros(dtwin))     # learnable log std

        self.dtwin        = dtwin
        self.scenario_dim = scenario_dim

    def forward(
        self,
        z:  torch.Tensor,                    # (B, dtwin)
        t:  Optional[torch.Tensor] = None,   # (B, K) scenario embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: mu (B, dtwin), std (B, dtwin)
        """
        if t is not None and self.scenario_dim > 0:
            inp = torch.cat([z, t], dim=-1)
        else:
            inp = z

        h   = self.net(inp)
        mu  = self.mu_head(h)
        std = torch.exp(self.log_std.clamp(-4, 2)).unsqueeze(0).expand_as(mu)
        return mu, std

    def sample(
        self,
        z:  torch.Tensor,
        t:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action a_τ ~ π_θ(z_τ, t).
        Returns: action (B, dtwin), log_prob (B,)
        """
        mu, std = self.forward(z, t)
        dist    = Normal(mu, std)
        action  = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self,
        z:      torch.Tensor,
        a:      torch.Tensor,
        t:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy of given action.
        Returns: log_prob (B,), entropy (B,)
        """
        mu, std  = self.forward(z, t)
        dist     = Normal(mu, std)
        log_prob = dist.log_prob(a).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PPOValueNetwork(nn.Module):
    """
    Value (critic) network.
    Shares architecture with policy: [dtwin (+K) → 128 → 64 → 1]
    """

    def __init__(
        self,
        dtwin:       int = 7,
        hidden_dims: tuple = (128, 64),
        scenario_dim: int = 0,
    ):
        super().__init__()
        input_dim = dtwin + scenario_dim
        layers    = []
        prev      = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev    = h
        layers.append(nn.Linear(prev, 1))
        self.net  = nn.Sequential(*layers)
        self.scenario_dim = scenario_dim

    def forward(
        self,
        z: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns value estimate V (B, 1)"""
        if t is not None and self.scenario_dim > 0:
            inp = torch.cat([z, t], dim=-1)
        else:
            inp = z
        return self.net(inp)


class ScenarioConditioner(nn.Module):
    """
    Scenario embedding module (Section III-M, Eqs. 10–11).

    Scenario embeddings are derived from cohort-level statistical shifts
    (mean-difference vectors between patient subgroups), NOT arbitrary vectors.

    t_s = (1/|G_s|) Σ_{i ∈ G_s} z_0^(i) − (1/|Ḡ_s|) Σ_{j ∈ Ḡ_s} z_0^(j)
    z'_0 = z_0 + W_t · t,   W_t ∈ R^{dtwin × K}

    IMPORTANT: Scenario outputs are model-consistent representation-space
    responses. They imply no causal identifiability and constitute
    NO treatment recommendations.
    """

    def __init__(self, dtwin: int = 7, n_scenarios: int = 3):
        super().__init__()
        self.W_t   = nn.Linear(n_scenarios, dtwin, bias=False)
        self.dtwin = dtwin
        self.n_scenarios = n_scenarios

        # Registered scenario embeddings (set after training)
        self.register_buffer("scenario_embeddings", torch.zeros(n_scenarios, dtwin))

    def set_scenario_embeddings(self, embeddings: torch.Tensor):
        """
        Set scenario embeddings derived from training data.
        embeddings: (n_scenarios, dtwin)
        """
        assert embeddings.shape == (self.n_scenarios, self.dtwin)
        self.scenario_embeddings.copy_(embeddings)

    def condition_z0(
        self,
        z0:           torch.Tensor,   # (B, dtwin)
        scenario_idx: Optional[int] = None,
        t_custom:     Optional[torch.Tensor] = None,   # (B, K) or (K,)
    ) -> torch.Tensor:
        """
        Apply scenario conditioning: z'_0 = z_0 + W_t · t  (Eq. 11).
        """
        if t_custom is not None:
            t = t_custom
            if t.dim() == 1:
                t = t.unsqueeze(0).expand(z0.shape[0], -1)
        elif scenario_idx is not None:
            t = torch.zeros(z0.shape[0], self.n_scenarios, device=z0.device)
            t[:, scenario_idx] = 1.0
        else:
            return z0

        return z0 + self.W_t(t)


class GAEComputer:
    """
    Generalised Advantage Estimation (GAE).
    Used in PPO update (Section III-O.1).
    λ_GAE = 0.95 (paper default)
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma      = gamma
        self.gae_lambda = gae_lambda

    def compute(
        self,
        rewards:  torch.Tensor,   # (B, T)
        values:   torch.Tensor,   # (B, T+1) — includes bootstrap value
        dones:    torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: advantages (B, T), returns (B, T)
        """
        T          = rewards.shape[1]
        advantages = torch.zeros_like(rewards)
        last_gae   = torch.zeros(rewards.shape[0], device=rewards.device)

        for t in reversed(range(T)):
            next_val     = values[:, t + 1]
            delta        = rewards[:, t] + self.gamma * next_val * (1 - dones[:, t]) - values[:, t]
            last_gae     = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values[:, :T]
        return advantages, returns


def ppo_clip_loss(
    log_probs_new: torch.Tensor,   # (B,)
    log_probs_old: torch.Tensor,   # (B,)
    advantages:    torch.Tensor,   # (B,)
    clip_eps:      float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PPO clipped surrogate objective (Eq. 15).
    L^CLIP(θ) = E_τ [ min( r_τ * Â_τ, clip(r_τ, 1−ε, 1+ε) * Â_τ ) ]
    r_τ = π_θ / π_θ_old

    Returns: loss (scalar), clip_fraction (scalar)
    """
    ratio        = torch.exp(log_probs_new - log_probs_old)
    surr1        = ratio * advantages
    surr2        = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss         = -torch.min(surr1, surr2).mean()

    # Track clipping rate (paper reports 18.3%)
    clipped      = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
    return loss, clipped
