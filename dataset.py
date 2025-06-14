"""
TCGA-BRCA Multimodal Dataset Module

This module provides the dataset class for loading and preprocessing
TCGA-BRCA data including WSI features and RNA-seq data.

Author: Nabil Hezil
Date: 2025
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BRCA_Multimodal_Dataset(Dataset):
    """
    TCGA-BRCA Multimodal Dataset for combining WSI features and RNA-seq data.
    
    This dataset loads histopathology patch features and RNA-seq gene expression
    data for breast cancer subtype classification.
    
    Args:
        pt_dir (str): Directory containing .pt files with WSI features
        tpm_file (str): Path to TPM normalized RNA-seq data CSV file
        subtype_file (str): Path to subtype labels CSV file
        max_patches (int): Maximum number of patches per slide (default: 512)
        log_transform (bool): Whether to log-transform RNA data (default: True)
        
    Attributes:
        feature_dim (int): Dimension of WSI features
        rna_dim (int): Dimension of RNA-seq features
        num_classes (int): Number of cancer subtypes
        subtype_map (dict): Mapping from patient ID to subtype label
        samples (list): List of (filename, patient_id, path) tuples
    """
    
    def __init__(
        self, 
        pt_dir: str, 
        tpm_file: str, 
        subtype_file: str, 
        max_patches: int = 512,
        log_transform: bool = True
    ):
        self.pt_dir = pt_dir
        self.max_patches = max_patches
        self.log_transform = log_transform
        
        logger.info("Initializing BRCA Multimodal Dataset...")
        
        # Load and process subtype labels
        self._load_subtypes(subtype_file)
        
        # Load and process RNA-seq data
        self._load_rna_data(tpm_file)
        
        # Create sample list and get feature dimensions
        self._create_sample_list()
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
        logger.info(f"Feature dimensions - WSI: {self.feature_dim}, RNA: {self.rna_dim}")
        logger.info(f"Number of classes: {self.num_classes}")
        
    def _load_subtypes(self, subtype_file: str) -> None:
        """Load and process subtype labels."""
        logger.info("Loading subtype labels...")
        
        # Load subtype data
        self.subtypes = pd.read_csv(subtype_file, sep=None, engine='python')
        
        # Clean column names
        self.subtypes.columns = [col.strip() for col in self.subtypes.columns]
        
        # Rename column if needed
        if "BRCA_Subtype_PAM50" not in self.subtypes.columns:
            # Find the subtype column (assuming it contains 'subtype' or 'PAM50')
            subtype_cols = [col for col in self.subtypes.columns 
                          if 'subtype' in col.lower() or 'pam50' in col.lower()]
            if subtype_cols:
                self.subtypes = self.subtypes.rename(columns={subtype_cols[0]: "BRCA_Subtype_PAM50"})
        
        # Filter valid subtypes
        self.subtypes = self.subtypes[self.subtypes['BRCA_Subtype_PAM50'].notna()]
        self.subtypes = self.subtypes[self.subtypes['BRCA_Subtype_PAM50'] != 'NA']
        self.subtypes = self.subtypes[self.subtypes['BRCA_Subtype_PAM50'] != 'Normal']
        
        # Encode labels
        label_encoder = LabelEncoder()
        self.subtypes['label'] = label_encoder.fit_transform(self.subtypes['BRCA_Subtype_PAM50'])
        
        # Create mapping from patient ID to label
        patient_col = 'patient' if 'patient' in self.subtypes.columns else self.subtypes.columns[0]
        self.subtype_map = dict(zip(self.subtypes[patient_col], self.subtypes['label']))
        
        # Store unique subtypes for reference
        self.unique_subtypes = sorted(self.subtypes['BRCA_Subtype_PAM50'].unique())
        self.num_classes = len(self.unique_subtypes)
        
        logger.info(f"Loaded {len(self.subtypes)} samples with subtypes: {self.unique_subtypes}")
    
    def _load_rna_data(self, tpm_file: str) -> None:
        """Load and process RNA-seq data."""
        logger.info("Loading RNA-seq data...")
        
        # Load RNA-seq data
        rna = pd.read_csv(tpm_file, sep='\t', index_col=0)
        
        # Log transform if specified
        if self.log_transform:
            rna = np.log2(rna + 1)
            logger.info("Applied log2(TPM + 1) transformation")
        
        # Standardize features
        self.rna_scaler = StandardScaler()
        rna_scaled = self.rna_scaler.fit_transform(rna)
        self.rna = pd.DataFrame(rna_scaled, index=rna.index, columns=rna.columns)
        
        # Filter to common patients
        common_ids = set(self.subtype_map.keys()).intersection(set(self.rna.index))
        self.rna = self.rna.loc[list(common_ids)]
        self.subtypes = self.subtypes[self.subtypes.iloc[:, 0].isin(common_ids)]
        
        logger.info(f"RNA-seq data shape: {self.rna.shape}")
        logger.info(f"Common patients after filtering: {len(common_ids)}")
        
    def _create_sample_list(self) -> None:
        """Create list of samples and determine feature dimensions."""
        logger.info("Creating sample list...")
        
        # Get all .pt files
        all_pt_files = [f for f in os.listdir(self.pt_dir) if f.endswith(".pt")]
        
        # Extract patient ID from filename (first 12 characters)
        slide_patient_map = {f: f[:12] for f in all_pt_files}
        
        # Filter to patients with both WSI and RNA data
        common_ids = set(self.subtype_map.keys()).intersection(set(self.rna.index))
        
        self.samples = [
            (fname, slide_patient_map[fname], os.path.join(self.pt_dir, fname))
            for fname in all_pt_files
            if slide_patient_map[fname] in common_ids
        ]
        
        # Get feature dimensions from first sample
        if self.samples:
            first_sample_features = torch.load(self.samples[0][2], map_location='cpu')
            self.feature_dim = first_sample_features.shape[1]
            self.rna_dim = self.rna.shape[1]
        else:
            raise ValueError("No valid samples found!")
            
        logger.info(f"Created {len(self.samples)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing:
                - histology: WSI patch features [max_patches, feature_dim]
                - mask: Padding mask [max_patches]
                - rna: RNA-seq features [rna_dim]
                - label: Subtype label [1]
        """
        fname, patient_id, pt_path = self.samples[idx]
        
        # Load and normalize WSI features
        feats = torch.load(pt_path, map_location='cpu')
        feats = self._normalize_features(feats)
        
        # Handle patch sampling and padding
        feats, mask = self._process_patches(feats)
        
        # Load RNA-seq data
        rna = torch.tensor(self.rna.loc[patient_id].values, dtype=torch.float32)
        
        # Get label
        label = self.subtype_map[patient_id]
        
        return {
            'histology': feats,
            'mask': mask,
            'rna': rna,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'slide_id': fname
        }
    
    def _normalize_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Normalize features using z-score normalization."""
        return (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    
    def _process_patches(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process patches by sampling and padding.
        
        Args:
            feats: Patch features [num_patches, feature_dim]
            
        Returns:
            tuple: (processed_features, mask)
        """
        num_patches = feats.shape[0]
        
        if num_patches >= self.max_patches:
            # Random sampling if we have too many patches
            idxs = torch.randperm(num_patches)[:self.max_patches]
            feats = feats[idxs]
            mask = torch.zeros(self.max_patches)
        else:
            # Pad if we have too few patches
            pad_size = self.max_patches - num_patches
            pad = torch.zeros(pad_size, feats.shape[1])
            feats = torch.cat([feats, pad], dim=0)
            mask = torch.cat([
                torch.zeros(num_patches), 
                torch.ones(pad_size)
            ])
        
        return feats, mask
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        distribution = {}
        for _, patient_id, _ in self.samples:
            label = self.subtype_map[patient_id]
            subtype = self.unique_subtypes[label]
            distribution[subtype] = distribution.get(subtype, 0) + 1
        return distribution
    
    def get_sample_info(self, idx: int) -> Dict[str, str]:
        """Get information about a specific sample."""
        fname, patient_id, pt_path = self.samples[idx]
        label = self.subtype_map[patient_id]
        subtype = self.unique_subtypes[label]
        
        return {
            'slide_id': fname,
            'patient_id': patient_id,
            'subtype': subtype,
            'label': label,
            'pt_path': pt_path
        }


def create_data_loaders(
    dataset: BRCA_Multimodal_Dataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 4,
    random_state: int = 42
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset: BRCA_Multimodal_Dataset instance
        train_split: Fraction of data for training (default: 0.7)
        val_split: Fraction of data for validation (default: 0.15)
        test_split: Fraction of data for testing (default: 0.15)
        batch_size: Batch size for data loaders (default: 16)
        num_workers: Number of worker processes (default: 4)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Get labels for stratified split
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_split / (val_split + test_split),
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders - Train: {len(train_indices)}, "
                f"Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    dataset = BRCA_Multimodal_Dataset(
        pt_dir="/path/to/pt_files",
        tpm_file="/path/to/tpm_unstranded.csv",
        subtype_file="/path/to/TCGA_BRCA_subtypes.csv"
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(dataset)
    
    # Test loading a batch
    sample_batch = next(iter(train_loader))
    print(f"Batch keys: {sample_batch.keys()}")
    print(f"Histology shape: {sample_batch['histology'].shape}")
    print(f"RNA shape: {sample_batch['rna'].shape}")
