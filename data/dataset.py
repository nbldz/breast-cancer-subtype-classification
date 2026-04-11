"""
data/dataset.py
Dataset classes for CDLS: TCGA-BRCA, BCSC, METABRIC.

Modality-disjoint regime: no patient has all modalities simultaneously.
  - TCGA-BRCA : WSI + RNA-seq + Clinical (no BCSC sequences)
  - BCSC       : Mammography sequences + Clinical (no WSI, no RNA)
  - METABRIC   : RNA-seq + Clinical (no WSI, no BCSC sequences)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, List, Optional, Tuple


# ─── PAM50 subtype mapping ─────────────────────────────────────────────────────
SUBTYPE2IDX = {"Basal": 0, "HER2": 1, "LumA": 2, "LumB": 3, "Normal": 4}
IDX2SUBTYPE = {v: k for k, v in SUBTYPE2IDX.items()}

# ─── Cohort identifiers ────────────────────────────────────────────────────────
COHORT_TCGA    = 0
COHORT_BCSC    = 1
COHORT_METABRIC = 2


class AbsenceToken:
    """
    Placeholder class; actual learned absence token embeddings live in the model.
    Datasets return NaN-filled tensors for absent modalities; the model replaces
    them with learned mwsi / mrna tokens before fusion.
    """
    pass


class TCGABRCADataset(Dataset):
    """
    TCGA-BRCA dataset.
    Modalities present: WSI patch features, RNA-seq, Clinical.
    RNA is absent token for patients missing RNA; WSI absent token if no slide.

    Directory layout (post preprocessing):
        processed/tcga_brca/
            wsi_features/   <patient_id>.npy   shape: (N_patches, 1024)
            rna_features/   <patient_id>.npy   shape: (20481,)
            clinical.csv                        columns: patient_id, ...covariates..., survival_time, event
            labels.csv                          columns: patient_id, subtype, split
    """

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "val" | "test"
        max_patches: int = 2000,
    ):
        super().__init__()
        self.root       = root
        self.split      = split
        self.max_patches = max_patches

        labels_df   = pd.read_csv(os.path.join(root, "labels.csv"))
        clinical_df = pd.read_csv(os.path.join(root, "clinical.csv"))

        self.df = labels_df[labels_df["split"] == split].reset_index(drop=True)
        self.df = self.df.merge(clinical_df, on="patient_id")

        # Clinical feature columns (all except identifiers / split / label columns)
        exclude = {"patient_id", "subtype", "split", "survival_time", "event"}
        self.clin_cols = [c for c in self.df.columns if c not in exclude]

        # Fit z-score stats on train split only
        train_df = labels_df[labels_df["split"] == "train"].merge(clinical_df, on="patient_id")
        self._clin_mean = train_df[self.clin_cols].mean().values.astype(np.float32)
        self._clin_std  = train_df[self.clin_cols].std().values.astype(np.float32) + 1e-6

    def __len__(self) -> int:
        return len(self.df)

    def _load_wsi(self, patient_id: str) -> Tuple[torch.Tensor, bool]:
        path = os.path.join(self.root, "wsi_features", f"{patient_id}.npy")
        if os.path.exists(path):
            feats = np.load(path).astype(np.float32)          # (N, 1024)
            if len(feats) > self.max_patches:
                idx = np.random.choice(len(feats), self.max_patches, replace=False)
                feats = feats[idx]
            return torch.from_numpy(feats), True
        # Absent: return empty tensor; model applies learned absence token
        return torch.zeros(1, 1024), False

    def _load_rna(self, patient_id: str) -> Tuple[torch.Tensor, bool]:
        path = os.path.join(self.root, "rna_features", f"{patient_id}.npy")
        if os.path.exists(path):
            return torch.from_numpy(np.load(path).astype(np.float32)), True
        return torch.zeros(20481), False

    def __getitem__(self, idx: int) -> Dict:
        row         = self.df.iloc[idx]
        patient_id  = row["patient_id"]
        subtype     = SUBTYPE2IDX[row["subtype"]]
        surv_time   = float(row["survival_time"])
        event       = int(row["event"])

        wsi_feats, wsi_present = self._load_wsi(patient_id)
        rna_feats, rna_present = self._load_rna(patient_id)

        clin_raw = row[self.clin_cols].values.astype(np.float32)
        clin_norm = (clin_raw - self._clin_mean) / self._clin_std
        clin_feats = torch.from_numpy(clin_norm)

        return {
            "patient_id":   patient_id,
            "cohort":       COHORT_TCGA,
            "wsi":          wsi_feats,          # (N_patches, 1024) or zeros
            "wsi_present":  wsi_present,
            "rna":          rna_feats,           # (20481,) or zeros
            "rna_present":  rna_present,
            "bcsc":         torch.zeros(1, 7),   # absent for TCGA
            "bcsc_present": False,
            "bcsc_len":     0,
            "clinical":     clin_feats,
            "subtype":      torch.tensor(subtype, dtype=torch.long),
            "surv_time":    torch.tensor(surv_time, dtype=torch.float32),
            "event":        torch.tensor(event,     dtype=torch.float32),
        }


class BCSCDataset(Dataset):
    """
    BCSC mammography screening dataset.
    Modalities present: longitudinal mammography sequences + Clinical.
    No WSI; no RNA-seq.

    Directory layout:
        processed/bcsc/
            sequences/  <patient_id>.npy   shape: (T_visits, 7)
            clinical.csv
            labels.csv  (columns: patient_id, subtype, split)
    """

    def __init__(self, root: str, split: str = "train", max_visits: int = 20):
        super().__init__()
        self.root      = root
        self.split     = split
        self.max_visits = max_visits

        labels_df   = pd.read_csv(os.path.join(root, "labels.csv"))
        clinical_df = pd.read_csv(os.path.join(root, "clinical.csv"))

        self.df = labels_df[labels_df["split"] == split].reset_index(drop=True)
        self.df = self.df.merge(clinical_df, on="patient_id")

        exclude = {"patient_id", "subtype", "split", "survival_time", "event"}
        self.clin_cols = [c for c in self.df.columns if c not in exclude]

        train_df = labels_df[labels_df["split"] == "train"].merge(clinical_df, on="patient_id")
        self._clin_mean = train_df[self.clin_cols].mean().values.astype(np.float32)
        self._clin_std  = train_df[self.clin_cols].std().values.astype(np.float32) + 1e-6

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row        = self.df.iloc[idx]
        patient_id = row["patient_id"]
        subtype    = SUBTYPE2IDX[row["subtype"]]
        surv_time  = float(row.get("survival_time", 0.0))
        event      = int(row.get("event", 0))

        seq_path = os.path.join(self.root, "sequences", f"{patient_id}.npy")
        if os.path.exists(seq_path):
            seq = np.load(seq_path).astype(np.float32)        # (T, 7)
            seq = seq[: self.max_visits]
            seq_len = len(seq)
            # Pad to max_visits
            pad = np.zeros((self.max_visits - seq_len, 7), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq     = np.zeros((self.max_visits, 7), dtype=np.float32)
            seq_len = 0

        clin_raw  = row[self.clin_cols].values.astype(np.float32)
        clin_norm = (clin_raw - self._clin_mean) / self._clin_std

        return {
            "patient_id":   patient_id,
            "cohort":       COHORT_BCSC,
            "wsi":          torch.zeros(1, 1024),
            "wsi_present":  False,
            "rna":          torch.zeros(20481),
            "rna_present":  False,
            "bcsc":         torch.from_numpy(seq),             # (max_visits, 7)
            "bcsc_present": True,
            "bcsc_len":     seq_len,
            "clinical":     torch.from_numpy(clin_norm),
            "subtype":      torch.tensor(subtype,   dtype=torch.long),
            "surv_time":    torch.tensor(surv_time, dtype=torch.float32),
            "event":        torch.tensor(event,     dtype=torch.float32),
        }


class METABRICDataset(Dataset):
    """
    METABRIC dataset.
    Modalities present: RNA-seq + Clinical. No WSI, no BCSC sequences.
    Used for transfer-learning evaluation only (Section V-E).

    Directory layout:
        processed/metabric/
            rna_features/  <patient_id>.npy   shape: (20481,)  (same exclusion as TCGA)
            clinical.csv
            labels.csv
    """

    def __init__(self, root: str, split: str = "test"):
        super().__init__()
        self.root  = root
        self.split = split

        labels_df   = pd.read_csv(os.path.join(root, "labels.csv"))
        clinical_df = pd.read_csv(os.path.join(root, "clinical.csv"))

        self.df = labels_df[labels_df["split"] == split].reset_index(drop=True)
        self.df = self.df.merge(clinical_df, on="patient_id")

        exclude = {"patient_id", "subtype", "split", "survival_time", "event"}
        self.clin_cols = [c for c in self.df.columns if c not in exclude]

        # Use METABRIC train split stats if available; else compute from all
        all_df = labels_df.merge(clinical_df, on="patient_id")
        self._clin_mean = all_df[self.clin_cols].mean().values.astype(np.float32)
        self._clin_std  = all_df[self.clin_cols].std().values.astype(np.float32) + 1e-6

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row        = self.df.iloc[idx]
        patient_id = row["patient_id"]
        subtype    = SUBTYPE2IDX[row["subtype"]]
        surv_time  = float(row.get("survival_time", 0.0))
        event      = int(row.get("event", 0))

        rna_path = os.path.join(self.root, "rna_features", f"{patient_id}.npy")
        if os.path.exists(rna_path):
            rna_feats, rna_present = torch.from_numpy(np.load(rna_path).astype(np.float32)), True
        else:
            rna_feats, rna_present = torch.zeros(20481), False

        clin_raw  = row[self.clin_cols].values.astype(np.float32)
        clin_norm = (clin_raw - self._clin_mean) / self._clin_std

        return {
            "patient_id":   patient_id,
            "cohort":       COHORT_METABRIC,
            "wsi":          torch.zeros(1, 1024),
            "wsi_present":  False,
            "rna":          rna_feats,
            "rna_present":  rna_present,
            "bcsc":         torch.zeros(1, 7),
            "bcsc_present": False,
            "bcsc_len":     0,
            "clinical":     torch.from_numpy(clin_norm),
            "subtype":      torch.tensor(subtype,   dtype=torch.long),
            "surv_time":    torch.tensor(surv_time, dtype=torch.float32),
            "event":        torch.tensor(event,     dtype=torch.float32),
        }


def collate_modality_disjoint(batch: List[Dict]) -> Dict:
    """
    Custom collate for variable-length WSI patch sequences.
    WSI tensors are zero-padded to max patch count within batch.
    """
    max_patches = max(b["wsi"].shape[0] for b in batch)

    wsi_padded = []
    for b in batch:
        p   = b["wsi"]
        pad = torch.zeros(max_patches - p.shape[0], 1024)
        wsi_padded.append(torch.cat([p, pad], dim=0))

    keys_simple = [
        "rna", "bcsc", "clinical", "subtype", "surv_time", "event",
        "cohort", "bcsc_len",
    ]
    out = {
        "patient_id":   [b["patient_id"]   for b in batch],
        "wsi":          torch.stack(wsi_padded),
        "wsi_present":  torch.tensor([b["wsi_present"]  for b in batch]),
        "rna_present":  torch.tensor([b["rna_present"]  for b in batch]),
        "bcsc_present": torch.tensor([b["bcsc_present"] for b in batch]),
    }
    for k in keys_simple:
        out[k] = torch.stack([b[k] if isinstance(b[k], torch.Tensor)
                               else torch.tensor(b[k]) for b in batch])
    return out


def build_datasets(cfg, split: str = "train"):
    """
    Build the modality-disjoint ConcatDataset for a given split.
    """
    tcga = TCGABRCADataset(cfg["data"]["tcga_brca_dir"], split=split)
    bcsc = BCSCDataset(cfg["data"]["bcsc_dir"],          split=split)
    return ConcatDataset([tcga, bcsc])
