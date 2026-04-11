"""
models/feedback.py
Closed-Loop Latent Feedback Mechanism (Section III-E).

Core equation (Eq. 2):
    z_{t+1} = T(z_t, a_t) + λ · (1/k) Σ_{i ∈ N(z_t)} (z_i − z_t)

where:
  - N(z_t) is the set of k nearest neighbours of z_t in the offline
    training-set embedding bank (Euclidean distance)
  - z_i are the retrieved real patient embeddings
  - λ ∈ [0, 1] is the feedback strength (default 0.1)

Properties (stated precisely in Section III-E):
  1. Entirely data-driven: N(z_t) retrieved from offline training bank
  2. Correction term is the displacement toward the local empirical mean
  3. NOT causal inference — regularisation in representation space only

Default hyperparameters (Table II, optimal row):
    λ = 0.1, k = 5
"""

import torch
import numpy as np
from typing import Optional


class ClosedLoopFeedback:
    """
    kNN-based latent feedback correction.

    The embedding bank is built offline from training-set z0 embeddings
    (Algorithm 1, line 3) and is NOT updated during PPO rollouts.
    """

    def __init__(
        self,
        lambda_feedback: float = 0.1,
        k_neighbours:    int   = 5,
    ):
        self.lambda_feedback = lambda_feedback
        self.k               = k_neighbours
        self._bank: Optional[torch.Tensor] = None   # (N_train, dtwin)

    def build_bank(self, embeddings: torch.Tensor):
        """
        Build the offline kNN index from training-set embeddings.
        embeddings: (N_train, dtwin) — z0 or z_T from training set.
        Called once before PPO training begins (Algorithm 1, line 3).
        """
        self._bank = embeddings.detach().cpu()
        print(f"[ClosedLoopFeedback] Bank built: {self._bank.shape}")

    @property
    def bank_size(self) -> int:
        return 0 if self._bank is None else len(self._bank)

    def apply(
        self,
        z_hat: torch.Tensor,    # (B, dtwin) — ẑ_{τ+1} from Twin-GRU
    ) -> torch.Tensor:
        """
        Apply the closed-loop feedback correction (Eq. 2).

        z_{t+1} = ẑ_{t+1} + λ · (1/k) Σ_{i ∈ N(ẑ_{t+1})} (z_i − ẑ_{t+1})
                = ẑ_{t+1} · (1 − λ) + λ · mean_neighbours

        Returns:
            z_next: (B, dtwin) — corrected latent state
        """
        if self._bank is None or self.lambda_feedback == 0.0:
            return z_hat

        z_cpu  = z_hat.detach().cpu()              # (B, dtwin)
        bank   = self._bank                         # (N, dtwin)

        # Euclidean distance: (B, N)
        # ||z - b||^2 = ||z||^2 + ||b||^2 - 2 z·b
        z_sq   = (z_cpu ** 2).sum(dim=1, keepdim=True)      # (B, 1)
        b_sq   = (bank  ** 2).sum(dim=1).unsqueeze(0)       # (1, N)
        dot    = z_cpu @ bank.T                              # (B, N)
        dist2  = z_sq + b_sq - 2 * dot                      # (B, N)
        dist2  = dist2.clamp(min=0.0)

        # k nearest neighbours (indices)
        k_actual = min(self.k, bank.shape[0])
        _, idx   = dist2.topk(k_actual, dim=1, largest=False)  # (B, k)

        # Retrieve neighbour embeddings and compute mean
        neighbours   = bank[idx]                               # (B, k, dtwin)
        mean_nbr     = neighbours.mean(dim=1)                  # (B, dtwin)

        # Correction: displacement toward local empirical mean
        correction   = mean_nbr.to(z_hat.device) - z_hat      # (B, dtwin)
        z_next       = z_hat + self.lambda_feedback * correction  # Eq. 2

        return z_next

    def retrieve_neighbours(
        self,
        z: torch.Tensor,      # (B, dtwin)
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Retrieve k nearest neighbour embeddings from the bank.
        Used for kNN patient retrieval in interface panel P6 and
        for the cross-cohort latent retrieval (Section III-N).

        Returns: (B, k, dtwin)
        """
        if self._bank is None:
            raise RuntimeError("Bank not built. Call build_bank() first.")

        k = k or self.k
        z_cpu = z.detach().cpu()
        bank  = self._bank

        z_sq  = (z_cpu ** 2).sum(dim=1, keepdim=True)
        b_sq  = (bank  ** 2).sum(dim=1).unsqueeze(0)
        dot   = z_cpu @ bank.T
        dist2 = (z_sq + b_sq - 2 * dot).clamp(min=0.0)

        k_actual  = min(k, bank.shape[0])
        vals, idx = dist2.topk(k_actual, dim=1, largest=False)

        return bank[idx], idx, vals.sqrt()   # embeddings, indices, distances


class CosineSimilarityRetrieval:
    """
    Cosine-similarity-based kNN retrieval for cross-cohort patient retrieval.
    Section III-N, Eq. 13.

    Recall@5 reported in paper: 81.3% (TCGA-BRCA), 74.2% (METABRIC cross-cohort).
    """

    def __init__(self, bank: Optional[torch.Tensor] = None):
        self._bank    = bank
        self._bank_n  = None   # normalised bank

    def set_bank(self, bank: torch.Tensor):
        self._bank   = bank.detach().cpu()
        self._bank_n = torch.nn.functional.normalize(self._bank, dim=1)

    def retrieve(
        self,
        z:  torch.Tensor,    # (B, dtwin)
        k:  int = 5,
    ):
        """
        Returns top-k matches by cosine similarity (Eq. 13).
        Returns: indices (B, k), similarities (B, k)
        """
        z_n   = torch.nn.functional.normalize(z.detach().cpu(), dim=1)
        sims  = z_n @ self._bank_n.T          # (B, N)
        vals, idx = sims.topk(k, dim=1, largest=True)
        return idx, vals

    def recall_at_k(
        self,
        z_query:     torch.Tensor,   # (N_q, dtwin)
        labels_q:    torch.Tensor,   # (N_q,)   subtype labels
        labels_bank: torch.Tensor,   # (N_bank,)
        k: int = 5,
    ) -> float:
        """
        Computes Recall@k: fraction of queries where ≥1 top-k neighbour
        shares the same PAM50 subtype label.
        """
        idx, _ = self.retrieve(z_query, k=k)
        correct = 0
        for i, q_label in enumerate(labels_q):
            retrieved_labels = labels_bank[idx[i]]
            if (retrieved_labels == q_label).any():
                correct += 1
        return correct / len(labels_q)
