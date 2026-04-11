"""
models/encoders.py
Modality encoders for CDLS.

Section III-K:
  1. WSI Hierarchical Attention Encoder   → h_wsi  ∈ R^768
  2. RNA MLP Encoder                      → ψ_rna  ∈ R^512
  3. BCSC GRU Encoder                     → e_long ∈ R^64
  4. Clinical ANN                         → h_clin ∈ R^256

Absent modalities are replaced by learned absence tokens before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─── 1. WSI Hierarchical Attention Encoder ────────────────────────────────────

class WSIHierarchicalAttentionEncoder(nn.Module):
    """
    Two-stage Transformer with gated attention pooling.
    Section III-K.1, Eqs. 6–7.

    Input:  patch features (B, N_patches, 1024)  from ConvNeXt-Base
    Output: slide-level embedding h_wsi ∈ R^768

    Architecture:
        1. Linear projection: 1024 → d_model=768
        2. Transformer stage 1: local patch-level self-attention (2 layers, 8 heads)
        3. Transformer stage 2: global slide-level self-attention (2 layers, 8 heads)
        4. Gated attention pooling → 768-d slide embedding
    """

    def __init__(
        self,
        input_dim:   int = 1024,
        d_model:     int = 768,
        n_layers:    int = 2,
        n_heads:     int = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_stage1 = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.transformer_stage2 = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Gated attention (Eq. 6–7)
        self.attn_a = nn.Linear(d_model, d_model // 2)  # W_a
        self.attn_g = nn.Linear(d_model, d_model // 2)  # W_g
        self.attn_w = nn.Linear(d_model // 2, 1)         # W_w

        self.output_dim = d_model

    def forward(
        self,
        x:           torch.Tensor,          # (B, N, 1024)
        patch_mask:  Optional[torch.Tensor] = None,  # (B, N) bool: True = valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_wsi:    (B, 768)  slide-level embedding
            attn_map: (B, N)    attention weights for visualisation (P2 panel)
        """
        h = self.input_proj(x)                         # (B, N, 768)
        h = self.transformer_stage1(h, src_key_padding_mask=~patch_mask if patch_mask is not None else None)
        h = self.transformer_stage2(h, src_key_padding_mask=~patch_mask if patch_mask is not None else None)

        # Gated attention pooling (Eq. 6)
        # α_j = tanh(W_a h_j^(2)) ⊙ sigmoid(W_g h_j^(2))
        alpha = torch.tanh(self.attn_a(h)) * torch.sigmoid(self.attn_g(h))   # (B, N, d/2)
        raw_w = self.attn_w(alpha).squeeze(-1)                                 # (B, N)

        if patch_mask is not None:
            raw_w = raw_w.masked_fill(~patch_mask, float("-inf"))

        # Eq. 7: w_j = softmax(W_w α_j)
        attn_weights = F.softmax(raw_w, dim=-1)                                # (B, N)
        attn_weights_safe = torch.nan_to_num(attn_weights, nan=0.0)

        h_wsi = (attn_weights_safe.unsqueeze(-1) * h).sum(dim=1)              # (B, 768)
        return h_wsi, attn_weights_safe


# ─── 2. RNA MLP Encoder ───────────────────────────────────────────────────────

class RNAMLPEncoder(nn.Module):
    """
    Three-layer MLP for RNA-seq features.
    Section III-K.2.
    Projection: [20481 → 1024 → 512 → 512], BN + Dropout(0.3)
    Output: ψ_rna ∈ R^512

    Preferred over VAE: stochastic bottleneck is counterproductive
    for stable PPO initialisation (Table XVII).
    """

    def __init__(
        self,
        input_dim:  int = 20481,
        layers:     Tuple[int, ...] = (1024, 512, 512),
        dropout:    float = 0.3,
        output_dim: int = 512,
    ):
        super().__init__()
        dims   = [input_dim] + list(layers)
        blocks = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            blocks += [
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        # Final layer without dropout
        blocks += [nn.Linear(dims[-1], output_dim)]
        self.net = nn.Sequential(*blocks)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 20481) → (B, 512)"""
        return self.net(x)


# ─── 3. BCSC GRU Encoder ──────────────────────────────────────────────────────

class BCSCGRUEncoder(nn.Module):
    """
    Two-layer GRU for BCSC longitudinal mammography sequences.
    Section III-K.3.
    Input:  x_long ∈ R^{T_i × 7}  (BCSC per-visit features, Table III)
    Output: e_long ∈ R^64

    For TCGA-BRCA and METABRIC patients (no BCSC sequences):
    the model substitutes the learned absence token m_long ∈ R^64.
    """

    def __init__(
        self,
        input_dim:  int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Learned absence token (used when BCSC sequences are absent)
        self.absence_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.absence_token, std=0.02)
        self.output_dim = hidden_dim

    def forward(
        self,
        x:          torch.Tensor,    # (B, T, 7)
        lengths:    torch.Tensor,    # (B,) actual sequence lengths
        present:    torch.Tensor,    # (B,) bool: True = BCSC sequences available
    ) -> torch.Tensor:
        """Returns e_long ∈ R^{B × 64}"""
        B = x.shape[0]
        output = torch.zeros(B, self.output_dim, device=x.device)

        bcsc_idx = present.nonzero(as_tuple=True)[0]
        if len(bcsc_idx) > 0:
            x_sub  = x[bcsc_idx]       # (n_bcsc, T, 7)
            len_sub = lengths[bcsc_idx]

            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sub, len_sub.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)        # hidden: (num_layers, n_bcsc, 64)
            e_bcsc    = hidden[-1]               # last layer: (n_bcsc, 64)
            output[bcsc_idx] = e_bcsc

        # Replace absent patients with learned token
        absent_idx = (~present).nonzero(as_tuple=True)[0]
        if len(absent_idx) > 0:
            output[absent_idx] = self.absence_token.unsqueeze(0).expand(len(absent_idx), -1)

        return output


# ─── 4. Clinical ANN ──────────────────────────────────────────────────────────

class ClinicalANN(nn.Module):
    """
    Two-layer ReLU network for clinical covariates.
    Section III-K.3.
    Output: h_clin ∈ R^256
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C) → (B, 256)"""
        return self.net(x)
