# breast-cancer-subtype-classification
This module implements the main multimodal classification models that combine WSI features and RNA-seq data for breast cancer subtype classificatio
# 
A PyTorch implementation for breast cancer subtype classification using both Whole Slide Images (WSI) and RNA-seq data from The Cancer Genome Atlas (TCGA) Breast Invasive Carcinoma (BRCA) dataset.

## 🎯 Overview

This project implements a deep learning framework that combines histopathological images and genomic data to classify breast cancer into molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like) using the PAM50 classification system.

### Key Features

- **Multi-modal Learning**: Combines WSI features and RNA-seq data for improved classification
- **Attention Mechanism**: Uses attention pooling to identify important tissue regions
- **Comparative Analysis**: Includes WSI-only and RNA-only models for performance comparison
- **Advanced Training**: Implements label smoothing, gradient clipping, and cosine annealing
- **Visualization Tools**: Provides attention maps and t-SNE embeddings visualization

## 🏗️ Architecture

### Multi-modal Model Components

1. **WSI Encoder**: Processes histopathological image features with attention pooling
2. **RNA Encoder**: Processes normalized RNA-seq expression data
3. **Feature Fusion**: Concatenates encoded features from both modalities
4. **Classifier**: Final classification layer with LayerNorm and dropout

### Model Variants

- **Multimodal**: WSI + RNA-seq features
- **WSI-only**: Histopathological features only
- **RNA-only**: Genomic features only

## 📊 Dataset

### TCGA-BRCA Dataset Requirements

- **WSI Features**: Pre-extracted patch features (.pt files) from whole slide images
- **RNA-seq Data**: TPM (Transcripts Per Million) expression values
- **Subtype Labels**: PAM50 molecular subtype annotations

### Data Structure
```
├── wsi/
│   └── features_UNI/
│       └── pt_files/
│           ├── TCGA-XX-XXXX-01Z-00-DX1.pt
│           └── ...
├── genomic/
│   └── tpm_unstranded.csv
└── subtypes/
    └── TCGA_BRCA_subtypes.csv
```

## 🚀 Installation

### Prerequisites

```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install tqdm
```

### Clone Repository

```bash
git clone https://github.com/yourusername/tcga-brca-multimodal.git
cd tcga-brca-multimodal
```

## 💻 Usage

### Basic Usage

```python
from multimodal_classifier import run

# Run the complete pipeline
run()
```

### Custom Configuration

```python
from multimodal_classifier import BRCA_Multimodal_Dataset, EnhancedMultimodalClassifier

# Initialize dataset
dataset = BRCA_Multimodal_Dataset(
    pt_dir="path/to/pt_files",
    tpm_file="path/to/tpm_data.csv",
    subtype_file="path/to/subtypes.csv",
    max_patches=512
)

# Create model
model = EnhancedMultimodalClassifier(
    hist_dim=dataset.feature_dim,
    rna_dim=dataset.rna_dim,
    hidden=1024,
    num_classes=dataset.num_classes
)
```

## 📈 Model Performance

### Hyperparameters

- **Learning Rate**: 2e-4 with AdamW optimizer
- **Batch Size**: 16
- **Max Patches**: 512 per WSI
- **Hidden Dimension**: 1024
- **Dropout**: 0.2-0.4 (layer-dependent)
- **Label Smoothing**: 0.1
- **Weight Decay**: 1e-4

### Training Features

- Cosine annealing learning rate scheduler
- Gradient clipping (max norm: 1.0)
- Early stopping based on validation accuracy
- Cross-entropy loss with label smoothing

## 📊 Results & Visualization

The framework provides several visualization tools:

### 1. Confusion Matrix
- Shows classification performance across all subtypes
- Displays percentages for better interpretation

### 2. Attention Visualization
- Highlights important tissue regions in WSI
- Shows attention weight distribution across patches

### 3. t-SNE Embeddings
- Visualizes learned feature representations
- Compares embeddings across different model variants

## 🔧 Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_patches` | Maximum patches per WSI | 512 |
| `hidden_dim` | Hidden layer dimension | 1024 |
| `num_epochs` | Training epochs | 50 |
| `learning_rate` | Initial learning rate | 2e-4 |
| `batch_size` | Training batch size | 16 |

### File Paths (Modify in `run()` function)

```python
pt_dir = "path/to/wsi/features"
rna_file = "path/to/genomic/tpm_data.csv"
subtype_file = "path/to/subtypes.csv"
```

## 📁 Output Files

The code generates several output files:

- `multimodal_confusion_matrix_percent.png`
- `wsi_only_confusion_matrix_percent.png`
- `rna_only_confusion_matrix_percent.png`
- `multimodal_attention_visualization.png`
- `embeddings_visualization.png`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{tcga_brca_multimodal,
  title={Enhanced Multi-modal Subtype Classification for TCGA-BRCA},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tcga-brca-multimodal}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Cancer Genome Atlas (TCGA) for providing the BRCA dataset
- UNI model for feature extraction from histopathological images
- PyTorch community for the deep learning framework

## 📞 Contact

- **Author**: Dr. Nabil Hezil
- **Email**: nhezil@sharjah.ac.ae
- 

## 🐛 Issues

If you encounter any issues or have questions, please open an issue on the [GitHub Issues](https://github.com/yourusername/tcga-brca-multimodal/issues) page.

---

*This project is part of ongoing research in computational pathology and precision medicine for breast cancer diagnosis and treatment.*
