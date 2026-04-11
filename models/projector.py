"""
models/projector.py
Fusion projector: maps fused 1600-d representation → latent state z0 ∈ R^7.

Section III-L, Eqs. 8–9:
    x_fusion = [h_wsi; ψ_rna; e_long; h_clin] ∈ R^1600   (768+512+64+256)
    z0 = π(x_fusion) = LayerNorm( W3 · ReLU( Dropout( W2 · ReLU( Dropout( W1 · x_fusion )))))

For absent modalities the learned WSI / RNA absence tokens are substituted
before concatenation (Section III-I).
"""

import torch
import torch.nn as nn
from typing import Optional


class ModalityAbsenceEncoder(nn.Module):
    """
    Learned absence tokens for WSI and RNA modalities.
    Replaces the absent sub-vector before fusion.
    Section III-I.
    """

    def __init__(self, wsi_dim: int = 768, rna_dim: int = 512):
        super().__init__()
        self.m_wsi = nn.Parameter(torch.zeros(wsi_dim))
        self.m_rna = nn.Parameter(torch.zeros(rna_dim))
        nn.init.normal_(self.m_wsi, std=0.02)
        nn.init.normal_(self.m_rna, std=0.02)

    def apply_tokens(
        self,
        h_wsi:       torch.Tensor,   # (B, 768)
        wsi_present: torch.Tensor,   # (B,) bool
        psi_rna:     torch.Tensor,   # (B, 512)
        rna_present: torch.Tensor,   # (B,) bool
    ):
        """
        Replace absent sub-vectors with learned tokens.
        Returns: (h_wsi_masked, psi_rna_masked)
        """
        # WSI
        h_wsi_out = h_wsi.clone()
        wsi_absent = ~wsi_present
        if wsi_absent.any():
            h_wsi_out[wsi_absent] = self.m_wsi.unsqueeze(0).expand(wsi_absent.sum(), -1)

        # RNA
        psi_rna_out = psi_rna.clone()
        rna_absent  = ~rna_present
        if rna_absent.any():
            psi_rna_out[rna_absent] = self.m_rna.unsqueeze(0).expand(rna_absent.sum(), -1)

        return h_wsi_out, psi_rna_out


class FusionProjector(nn.Module):
    """
    Four-layer projector from fused 1600-d representation to z0 ∈ R^dtwin.

    Architecture (Section III-L, Eq. 9):
        [1600 → 512 → 128 → dtwin]
        with ReLU activations and Dropout between layers.
    Final LayerNorm on z0.
    """

    def __init__(
        self,
        fusion_dim:  int = 1600,
        dtwin:       int = 7,
        hidden_dims: tuple = (512, 128),
        dropout:     float = 0.1,
    ):
        super().__init__()
        dims   = [fusion_dim] + list(hidden_dims) + [dtwin]
        layers = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:   # all except last
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net   = nn.Sequential(*layers)
        self.norm  = nn.LayerNorm(dtwin)
        self.dtwin = dtwin

    def forward(self, x_fusion: torch.Tensor) -> torch.Tensor:
        """
        x_fusion: (B, 1600) → z0: (B, dtwin=7)
        """
        return self.norm(self.net(x_fusion))
