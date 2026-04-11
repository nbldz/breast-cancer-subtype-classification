"""
models/twin_gru.py
Twin-GRU transition model for latent state evolution.

Section III-O.4, Eq. 20:
    ẑ_{τ+1} = LayerNorm( zτ + σ · tanh( W_trans · GRU([zτ; aτ], h_{τ-1}) ) )
    σ = 0.5, dropout = 0.1, weight_decay = 1e-5

After this transition, the closed-loop feedback correction (Eq. 2) is applied
(implemented in feedback.py) to yield z_{τ+1}.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TwinGRUTransition(nn.Module):
    """
    Twin-GRU latent transition model.

    Takes current latent state z_τ ∈ R^dtwin and PPO action a_τ ∈ R^dtwin,
    concatenates them, and passes through a GRU to produce a residual update.

    Eq. 20:
        ẑ_{τ+1} = LayerNorm( z_τ + σ · tanh( W_trans · GRU([z_τ; a_τ], h_{τ-1}) ) )
    """

    def __init__(
        self,
        dtwin:       int   = 7,
        hidden_size: int   = 128,
        sigma:       float = 0.5,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.dtwin       = dtwin
        self.hidden_size = hidden_size
        self.sigma       = sigma

        # GRU input: [z_τ; a_τ] ∈ R^{2*dtwin}
        self.gru = nn.GRUCell(
            input_size=dtwin * 2,
            hidden_size=hidden_size,
        )
        self.dropout   = nn.Dropout(dropout)
        self.W_trans   = nn.Linear(hidden_size, dtwin)
        self.layer_norm = nn.LayerNorm(dtwin)

    def forward(
        self,
        z_tau:   torch.Tensor,              # (B, dtwin)
        a_tau:   torch.Tensor,              # (B, dtwin)
        h_prev:  Optional[torch.Tensor] = None,  # (B, hidden_size) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_hat_next: (B, dtwin) — pre-feedback transition output ẑ_{τ+1}
            h_next:     (B, hidden_size) — updated GRU hidden state
        """
        inp    = torch.cat([z_tau, a_tau], dim=-1)              # (B, 2*dtwin)
        if h_prev is None:
            h_prev = torch.zeros(z_tau.shape[0], self.hidden_size, device=z_tau.device)

        h_next  = self.gru(inp, h_prev)                          # (B, hidden_size)
        h_drop  = self.dropout(h_next)
        residual = self.sigma * torch.tanh(self.W_trans(h_drop)) # (B, dtwin)
        z_hat   = self.layer_norm(z_tau + residual)              # Eq. 20
        return z_hat, h_next
