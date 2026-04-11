"""
data/augmentation.py
Adaptive class-balanced augmentation (Section III-F).

For class y with n_y samples out of N total and K=5 classes:
    p_aug(y) = min(1, N / (K * n_y))

Applied in the image domain only before feature extraction,
exclusively during training; validation and test splits receive no augmentation.
"""

import math
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Dict, Optional


# PAM50 class counts in TCGA-BRCA training set
TCGA_TRAIN_COUNTS = {
    0: 191,   # Basal
    1: 82,    # HER2
    2: 562,   # LumA
    3: 208,   # LumB
    4: 54,    # Normal
}


def compute_aug_probs(class_counts: Dict[int, int], K: int = 5) -> Dict[int, float]:
    """
    Eq. 3: p_aug(y) = min(1, N / (K * n_y))
    Returns dictionary mapping class index → augmentation probability.
    """
    N = sum(class_counts.values())
    return {
        y: min(1.0, N / (K * ny))
        for y, ny in class_counts.items()
    }


class AdaptiveAugmentation:
    """
    Adaptive class-balanced augmentation wrapper.
    Applied to PIL images / numpy arrays before feature extraction.

    Augmentation (Eq. 4):
        x̃_wsi = A(x_wsi) with probability p_aug(y)

    A comprises:
        - Random horizontal flip (p=0.5)
        - Colour jitter (brightness ±0.2, contrast ±0.2)
        - Random rotation (±15°)
    """

    def __init__(
        self,
        class_counts: Optional[Dict[int, int]] = None,
        K: int = 5,
        flip_p: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        rotation: float = 15.0,
    ):
        if class_counts is None:
            class_counts = TCGA_TRAIN_COUNTS

        self.aug_probs = compute_aug_probs(class_counts, K)
        self.K         = K

        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=flip_p),
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
            ),
            T.RandomRotation(degrees=rotation),
        ])

    def __call__(self, patch: np.ndarray, label: int) -> np.ndarray:
        """
        Args:
            patch:  uint8 RGB numpy array (H, W, 3) — a single WSI patch
            label:  PAM50 class index (0–4)
        Returns:
            Augmented or unchanged patch as numpy array (H, W, 3).
        """
        p = self.aug_probs.get(label, 0.0)
        if np.random.random() < p:
            from PIL import Image
            img = Image.fromarray(patch)
            img = self.transform(img)
            return np.array(img)
        return patch

    def aug_probability(self, label: int) -> float:
        return self.aug_probs.get(label, 0.0)


class TrainingAugmentation:
    """
    Standard augmentation for patch tensors during training
    (after feature extraction, using tensor-level augmentation
    of the patch features is not applicable; this class provides
    augmentation at the raw-image / patch level for the pretrain stage).
    """

    def __init__(self, flip_p: float = 0.5, brightness: float = 0.2,
                 contrast: float = 0.2, rotation: float = 15.0):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=flip_p),
            T.ColorJitter(brightness=brightness, contrast=contrast),
            T.RandomRotation(degrees=rotation),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, patch_pil):
        return self.transform(patch_pil)


class EvalTransform:
    """No augmentation for validation / test splits."""

    def __init__(self):
        self.transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, patch_pil):
        return self.transform(patch_pil)
