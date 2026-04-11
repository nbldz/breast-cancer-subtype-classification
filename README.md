# CDLS: Cross-Cohort Modality-Disjoint Latent Simulation

Official implementation of:

> **A Digital Twin–Inspired Closed-Loop Latent Simulation Framework for Cross-Cohort Breast Cancer Subtype Classification under Modality-Disjoint Learning**
> Nabil Hezil, Ahmed Bouridane, Rifat Hamoudi, Somaya Al-Maadeed, Suchithra Kunhoth, Faseela Abdullakutty, Younes Akbari
> *IEEE Journal of Biomedical and Health Informatics*, 2025

---

## Overview

CDLS is a closed-loop latent trajectory classification system for breast cancer PAM50 subtype classification. It:

- Integrates WSI, RNA-seq, mammography sequences, and clinical covariates from three non-overlapping cohorts (TCGA-BRCA, BCSC, METABRIC) under a **modality-disjoint** regime
- Applies a **PPO-governed stochastic policy** to refine latent state `z ∈ R^7` across `T=5` optimisation steps via a Twin-GRU transition model
- Uses a **closed-loop latent feedback** step after each transition to align simulated states with real patient embeddings via kNN retrieval

**Results:** Balanced Accuracy 0.870±0.044, MCC 0.904±0.046 (n=4 seeds); 5-fold CV Accuracy 0.871±0.029.

---

## Repository Structure

```
cdls/
├── configs/
│   └── default.yaml          # All hyperparameters
├── data/
│   ├── dataset.py            # TCGA-BRCA, BCSC, METABRIC dataset classes
│   ├── preprocessing.py      # WSI preprocessing pipeline
│   └── augmentation.py       # Adaptive class-balanced augmentation
├── models/
│   ├── encoders.py           # WSI, RNA, BCSC, Clinical encoders
│   ├── projector.py          # Fusion projector to latent z0
│   ├── twin_gru.py           # Twin-GRU transition model
│   ├── ppo.py                # PPO policy and value networks
│   ├── feedback.py           # Closed-loop kNN latent feedback
│   ├── heads.py              # PAM50 classifier + Cox survival head
│   └── cdls.py               # Full CDLS model
├── trainers/
│   ├── pretrain.py           # Encoder + projector pretraining
│   └── ppo_trainer.py        # PPO trajectory optimisation loop
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   ├── losses.py             # CrossEntropy + Cox partial likelihood
│   ├── intrinsic_dim.py      # Two-NN and MLE intrinsic dimensionality
│   ├── leakage_check.py      # Split integrity verification
│   └── calibration.py        # ECE, Brier score, reliability diagrams
├── interface/
│   └── visualisation.py      # 6-panel research visualisation tool
├── scripts/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   └── transfer_metabric.py  # METABRIC transfer-learning evaluation
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/nbldz/breast-cancer-subtype-classification
cd breast-cancer-subtype-classification
pip install -r requirements.txt
```

---

## Data Preparation

### TCGA-BRCA
1. Download WSI slides and RNA-seq from [GDC Portal](https://portal.gdc.cancer.gov/)
2. Download clinical data; place under `data/raw/tcga_brca/`

### BCSC
- Request access at [bcsc-research.org](https://www.bcsc-research.org/)
- Place mammography + clinical CSV under `data/raw/bcsc/`

### METABRIC
- Available via [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)
- Place RNA-seq + clinical under `data/raw/metabric/`

---

## Preprocessing

```bash
# WSI: patch extraction + stain normalisation + ConvNeXt-Base feature extraction
python scripts/preprocess_wsi.py --cohort tcga_brca --magnification 20 --patch_size 256 --extractor convnext_base

# RNA-seq: log2(TPM+1) + z-score normalisation + PAM50 gene exclusion
python scripts/preprocess_rna.py --cohort tcga_brca
python scripts/preprocess_rna.py --cohort metabric

# BCSC: per-visit feature extraction
python scripts/preprocess_bcsc.py
```

---

## Training

```bash
# Stage 1: Pre-train encoders + projector
python scripts/train.py --stage pretrain --config configs/default.yaml --seed 42

# Stage 2: PPO trajectory optimisation
python scripts/train.py --stage ppo --config configs/default.yaml --seed 42

# Run all 4 seeds
for seed in 42 123 789 2024; do
    python scripts/train.py --config configs/default.yaml --seed $seed
done
```

---

## Evaluation

```bash
# Held-out test set
python scripts/evaluate.py --config configs/default.yaml --seeds 42 123 789 2024

# 5-fold cross-validation
python scripts/evaluate.py --config configs/default.yaml --seeds 42 123 789 2024 --cv

# METABRIC transfer learning
python scripts/transfer_metabric.py --config configs/default.yaml --seed 789
```

---

## Reproducibility

Seeds `{42, 123, 789, 2024}` control all stochastic components (PyTorch, NumPy, Python random, CUDA). To reproduce the paper results exactly:

```bash
python scripts/train.py --config configs/default.yaml --seed 42 --deterministic
```

---

## Research Visualisation Interface

```bash
python interface/visualisation.py --checkpoint checkpoints/seed_789/best.pt --patient_id TCGA-3C-AAAU
```

Launches the 6-panel interface:
- **P1**: Scenario input and trajectory launch
- **P2**: WSI attention heatmap with step-wise playback
- **P3**: UMAP trajectory vs. real patient manifold
- **P4**: Per-step entropy, probability, and response-score curves
- **P5**: Baseline vs. scenario PAM50 probability shift
- **P6**: kNN patient retrieval with survival statistics

---

## Citation

```bibtex
@article{hezil2025cdls,
  title={A Digital Twin--Inspired Closed-Loop Latent Simulation Framework for Cross-Cohort Breast Cancer Subtype Classification under Modality-Disjoint Learning},
  author={Hezil Nabil,  Ahmed Bouridane, Hamoudi Rifat,  Somaya Al-Maadeed, Kunhoth Suchithra, Abdullakutty Faseela,  Akbari Younes},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License.
