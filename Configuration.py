"""
Configuration management for TCGA-BRCA multimodal classification
Contains all hyperparameters, paths, and model configurations
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data-related configuration"""
    pt_dir: str = "/kaggle/working/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files"
    rna_file: str = "/kaggle/working/LAB_WSI_Genomics/TCGA_BRCA/Data/genomic/tpm_unstranded.csv"
    subtype_file: str = "/kaggle/input/subtype-tcga/TCGA_BRCA_subtypes.csv"
    max_patches: int = 512
    test_size: float = 0.2
    val_size: float = 0.5  # From test split
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_dim: int = 1024
    num_classes: int = 4
    dropout_wsi: float = 0.3
    dropout_rna: float = 0.4
    dropout_classifier: float = 0.4
    use_layer_norm: bool = True
    use_attention: bool = True


@dataclass  
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_name: str = "tcga_brca_multimodal"
    output_dir: str = "./results"
    save_models: bool = True
    save_plots: bool = True
    
    # Evaluation
    run_cross_validation: bool = False
    cv_folds: int = 5
    
    # Visualization
    visualize_attention: bool = True
    visualize_embeddings: bool = True
    num_attention_samples: int = 3


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    
    def __post_init__(self):
        # Create output directory
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.model_dir = os.path.join(self.experiment.output_dir, "models")
        self.plot_dir = os.path.join(self.experiment.output_dir, "plots")
        self.log_dir = os.path.join(self.experiment.output_dir, "logs")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def save_config(self, path: Optional[str] = None):
        """Save configuration to file"""
        if path is None:
            path = os.path.join(self.experiment.output_dir, "config.yaml")
        
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        # Reconstruct config object
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# Predefined configurations for different experiments
def get_default_config():
    """Get default configuration"""
    return Config()


def get_ablation_config():
    """Configuration for ablation studies"""
    config = Config()
    config.training.num_epochs = 30
    config.experiment.experiment_name = "ablation_study"
    return config


def get_cross_validation_config():
    """Configuration for cross-validation experiments"""
    config = Config()
    config.experiment.run_cross_validation = True
    config.experiment.experiment_name = "cross_validation"
    config.training.num_epochs = 30
    return config


def get_hyperparameter_search_config():
    """Configuration for hyperparameter search"""
    config = Config()
    config.experiment.experiment_name = "hyperparameter_search"
    return config


# Hyperparameter search spaces
HYPERPARAMETER_SEARCH_SPACE = {
    'learning_rate': [1e-4, 2e-4, 5e-4, 1e-3],
    'hidden_dim': [512, 1024, 2048],
    'dropout_wsi': [0.2, 0.3, 0.4],
    'dropout_rna': [0.3, 0.4, 0.5],
    'batch_size': [8, 16, 32],
    'weight_decay': [1e-5, 1e-4, 1e-3]
}
