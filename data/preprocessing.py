"""
data/preprocessing.py
WSI preprocessing pipeline for CDLS.

Protocol (Table IV):
  - Magnification: 20× (0.5 µm/px)
  - Patch size:    256×256 px, non-overlapping
  - Tissue filter: ≥50% foreground (Otsu HSV)
  - Stain norm:    Macenko [35]
  - Extractor:     ConvNeXt-Base (ImageNet pre-trained), 1024-d output

CRITICAL: Split-then-extract protocol.
  Patient-level splits are determined BEFORE feature extraction
  to prevent patch-level data leakage.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("WARNING: openslide not found. WSI loading disabled.")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("WARNING: timm not found. Feature extraction disabled.")


# ─── Macenko Stain Normalisation ──────────────────────────────────────────────

class MacenkoNormaliser:
    """
    Macenko H&E stain normalisation.
    Reference: Macenko et al., ISBI 2009.
    """

    def __init__(self, Io: int = 240, alpha: float = 1.0, beta: float = 0.15):
        self.Io    = Io
        self.alpha = alpha
        self.beta  = beta
        # Target stain matrix and max concentrations (fitted to a reference slide)
        self.HERef = np.array([[0.5626, 0.2159],
                                [0.7201, 0.8012],
                                [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def normalise(self, img: np.ndarray) -> np.ndarray:
        """
        img: uint8 RGB array (H, W, 3).
        Returns normalised uint8 RGB array.
        """
        img = img.reshape(-1, 3).astype(np.float64)
        img = np.clip(img, 1, 255)
        OD  = -np.log(img / self.Io)

        # Remove transparent pixels
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if len(ODhat) == 0:
            return img.reshape(-1, 3).astype(np.uint8)

        _, V = np.linalg.eigh(np.cov(ODhat.T))
        V = V[:, [2, 1]]
        if V[0, 0] < 0: V[:, 0] *= -1
        if V[0, 1] < 0: V[:, 1] *= -1

        That  = ODhat @ V
        phi   = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
        vMax = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T

        HE   = HE / np.linalg.norm(HE, axis=0)
        C    = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        maxC = np.percentile(C, 99, axis=1)

        C2    = C * (self.maxCRef / (maxC + 1e-6))[:, None]
        Inorm = self.Io * np.exp(-self.HERef @ C2)
        Inorm = np.clip(Inorm.T, 0, 255).astype(np.uint8)
        return Inorm


# ─── Tissue Segmentation ──────────────────────────────────────────────────────

def tissue_mask_otsu(patch: np.ndarray, threshold: float = 0.50) -> bool:
    """
    Returns True if the patch contains ≥threshold fraction of tissue.
    Uses Otsu thresholding in HSV space (S channel).
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    s   = hsv[:, :, 1]
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    foreground_fraction = np.sum(mask > 0) / mask.size
    return foreground_fraction >= threshold


# ─── Patch Extractor ──────────────────────────────────────────────────────────

class WSIPatchExtractor:
    """
    Extracts non-overlapping 256×256 patches at 20× magnification
    after Otsu tissue filtering and Macenko stain normalisation.
    """

    def __init__(
        self,
        patch_size: int = 256,
        tissue_threshold: float = 0.50,
        stain_norm: bool = True,
    ):
        self.patch_size        = patch_size
        self.tissue_threshold  = tissue_threshold
        self.normaliser        = MacenkoNormaliser() if stain_norm else None

    def extract(self, slide_path: str, target_mpp: float = 0.5) -> List[np.ndarray]:
        """
        Returns list of RGB patches (H, W, 3) uint8.
        """
        if not HAS_OPENSLIDE:
            raise RuntimeError("openslide is required for WSI processing.")

        slide    = openslide.OpenSlide(slide_path)
        # Find the level closest to target_mpp (20×)
        mpp_x    = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
        downsample = mpp_x / target_mpp
        level    = slide.get_best_level_for_downsample(downsample)
        dims     = slide.level_dimensions[level]
        W, H     = dims

        patches = []
        for y in range(0, H, self.patch_size):
            for x in range(0, W, self.patch_size):
                region = slide.read_region(
                    (int(x * downsample), int(y * downsample)),
                    level,
                    (self.patch_size, self.patch_size),
                )
                patch = np.array(region.convert("RGB"))

                if not tissue_mask_otsu(patch, self.tissue_threshold):
                    continue

                if self.normaliser is not None:
                    try:
                        flat = self.normaliser.normalise(patch)
                        patch = flat.reshape(self.patch_size, self.patch_size, 3)
                    except Exception:
                        pass  # keep unnormalised if stain norm fails

                patches.append(patch)

        slide.close()
        return patches


# ─── ConvNeXt-Base Feature Extractor ──────────────────────────────────────────

class ConvNeXtBaseExtractor(nn.Module):
    """
    ConvNeXt-Base patch-level feature extractor.
    ImageNet pre-trained; fine-tuned on training-split patches.
    Output: 1024-d feature vector per patch.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise RuntimeError("timm is required for ConvNeXt-Base.")
        import timm
        self.model = timm.create_model(
            "convnext_base",
            pretrained=pretrained,
            num_classes=0,          # remove classifier head → feature extractor
        )
        self.out_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, 1024)"""
        return self.model(x)


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def preprocess_cohort(
    slide_dir:      str,
    out_dir:        str,
    patient_ids:    List[str],
    extractor:      nn.Module,
    device:         torch.device,
    batch_size:     int = 32,
    max_patches:    int = 2000,
):
    """
    End-to-end WSI preprocessing for a list of patient IDs.
    Saves per-patient feature arrays as .npy files.

    IMPORTANT: call this only on the training-split patient IDs first
    to prevent leakage, then call separately for val/test patients.
    """
    import torchvision.transforms as T

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    extractor = extractor.to(device).eval()
    patch_ext = WSIPatchExtractor(patch_size=256)
    os.makedirs(out_dir, exist_ok=True)

    for pid in tqdm(patient_ids, desc="Preprocessing WSI"):
        out_path = os.path.join(out_dir, f"{pid}.npy")
        if os.path.exists(out_path):
            continue

        slide_path = _find_slide(slide_dir, pid)
        if slide_path is None:
            print(f"  No slide found for {pid}; skipping.")
            continue

        patches = patch_ext.extract(slide_path)
        if len(patches) == 0:
            print(f"  No tissue patches extracted for {pid}.")
            continue

        if len(patches) > max_patches:
            idx     = np.random.choice(len(patches), max_patches, replace=False)
            patches = [patches[i] for i in idx]

        # Extract features in batches
        all_feats = []
        for i in range(0, len(patches), batch_size):
            batch = torch.stack([transform(p) for p in patches[i:i+batch_size]])
            with torch.no_grad():
                feats = extractor(batch.to(device)).cpu().numpy()
            all_feats.append(feats)

        features = np.concatenate(all_feats, axis=0)   # (N, 1024)
        np.save(out_path, features)


def _find_slide(slide_dir: str, patient_id: str) -> Optional[str]:
    """Search for a slide file matching patient_id."""
    for ext in [".svs", ".ndpi", ".tiff", ".tif", ".mrxs"]:
        path = os.path.join(slide_dir, f"{patient_id}{ext}")
        if os.path.exists(path):
            return path
    # Search recursively one level deep
    for sub in os.listdir(slide_dir):
        sub_path = os.path.join(slide_dir, sub)
        if os.path.isdir(sub_path):
            for ext in [".svs", ".ndpi", ".tiff", ".tif", ".mrxs"]:
                path = os.path.join(sub_path, f"{patient_id}{ext}")
                if os.path.exists(path):
                    return path
    return None


# ─── RNA-seq Preprocessing ────────────────────────────────────────────────────

def preprocess_rna(
    expression_csv:   str,
    out_dir:          str,
    patient_ids_train: List[str],
    pam50_gene_list:  str,
    n_genes:          int = 20481,
):
    """
    RNA-seq preprocessing:
      1. log2(TPM + 1) transformation
      2. Exclude PAM50 classifier genes
      3. z-score normalisation (statistics computed on train split only)
      4. Save per-patient .npy files

    Args:
        expression_csv:    Path to expression matrix CSV (genes × patients or patients × genes).
        out_dir:           Output directory for .npy files.
        patient_ids_train: Patient IDs in the training split (for z-score stats).
        pam50_gene_list:   Path to text file listing PAM50 genes to exclude.
        n_genes:           Expected number of genes after exclusion.
    """
    os.makedirs(out_dir, exist_ok=True)

    expr = pd.read_csv(expression_csv, index_col=0)

    # Exclude PAM50 genes
    with open(pam50_gene_list) as f:
        pam50_genes = {g.strip() for g in f.readlines()}
    expr = expr.loc[~expr.index.isin(pam50_genes)]

    # log2(TPM + 1)
    expr = np.log2(expr + 1)

    # z-score: fit on training split only
    train_mask = expr.columns.isin(patient_ids_train)
    mean_ = expr.loc[:, train_mask].mean(axis=1)
    std_  = expr.loc[:, train_mask].std(axis=1) + 1e-6
    expr  = (expr.subtract(mean_, axis=0)).divide(std_, axis=0)

    assert expr.shape[0] == n_genes, (
        f"Expected {n_genes} genes after PAM50 exclusion, got {expr.shape[0]}"
    )

    for pid in expr.columns:
        out_path = os.path.join(out_dir, f"{pid}.npy")
        np.save(out_path, expr[pid].values.astype(np.float32))

    print(f"Saved RNA features for {len(expr.columns)} patients → {out_dir}")
