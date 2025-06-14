"""
Main execution script for TCGA-BRCA multimodal classification
Orchestrates the entire training and evaluation pipeline
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Import your existing modules
from dataset import BRCA_Multimodal_Dataset
from multimodal import EnhancedMultimodalClassifier, WSIOnlyClassifier, RNAOnlyClassifier
from training_module import train_model, evaluate_model, compare_models
from utility import visualize_attention_weights, visualize_embeddings, plot_training_curves
from config import Config, get_default_config


def setup_data_loaders(config):
    """Setup train, validation, and test data loaders"""
    print("Setting up data loaders...")
    
    # Create dataset
    dataset = BRCA_Multimodal_Dataset(
        pt_dir=config.data.pt_dir,
        tpm_file=config.data.rna_file,
        subtype_file=config.data.subtype_file,
        max_patches=config.data.max_patches
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Feature dimensions - WSI: {dataset.feature_dim}, RNA: {dataset.rna_dim}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Create stratified splits
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    
    train_ids, test_ids = train_test_split(
        range(len(dataset)), 
        test_size=config.data.test_size,
        stratify=labels,
        random_state=config.data.random_state
    )
    
    val_ids, test_ids = train_test_split(
        test_ids,
        test_size=config.data.val_size,
        stratify=[labels[i] for i in test_ids],
        random_state=config.data.random_state
    )
    
    print(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create data loaders
    train_loader = DataLoader(
        Subset(dataset, train_ids),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_ids),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    test_loader = DataLoader(
        Subset(dataset, test_ids),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    return dataset, train_loader, val_loader, test_loader


def train_and_evaluate_models(dataset, train_loader, val_loader, test_loader, config):
    """Train and evaluate all model variants"""
    results = {}
    models = {}
    label_names = dataset.get_label_names()
    
    # 1. Train Multimodal Model
    print("\n" + "="*60)
    print("TRAINING MULTIMODAL MODEL")
    print("="*60)
    
    multimodal_model = EnhancedMultimodalClassifier(
        hist_dim=dataset.feature_dim,
        rna_dim=dataset.rna_dim,
        hidden=config.model.hidden_dim,
        num_classes=dataset.num_classes
    )
    
    multimodal_model = train_model(
        model=multimodal_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        mode="multimodal",
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=config.training.device
    )
    
    models['multimodal'] = multimodal_model
    
    # Save model
    if config.experiment.save_models:
        torch.save(multimodal_model.state_dict(), 
                  os.path.join(config.model_dir, "multimodal_model.pth"))
    
    # 2. Train WSI-Only Model
    print("\n" + "="*60)
    print("TRAINING WSI-ONLY MODEL")
    print("="*60)
    
    wsi_model = WSIOnlyClassifier(
        hist_dim=dataset.feature_dim,
        hidden=config.model.hidden_dim,
        num_classes=dataset.num_classes
    )
    
    wsi_model = train_model(
        model=wsi_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        mode="wsi_only",
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=config.training.device
    )
    
    models['wsi_only'] = wsi_model
    
    if config.experiment.save_models:
        torch.save(wsi_model.state_dict(), 
                  os.path.join(config.model_dir, "wsi_only_model.pth"))
    
    # 3. Train RNA-Only Model
    print("\n" + "="*60)
    print("TRAINING RNA-ONLY MODEL")
    print("="*60)
    
    rna_model = RNAOnlyClassifier(
        rna_dim=dataset.rna_dim,
        hidden=config.model.hidden_dim,
        num_classes=dataset.num_classes
    )
    
    rna_model = train_model(
        model=rna_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        mode="rna_only",
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=config.training.device
    )
    
    models['rna_only'] = rna_model
    
    if config.experiment.save_models:
        torch.save(rna_model.state_dict(), 
                  os.path.join(config.model_dir, "rna_only_model.pth"))
    
    # 4. Compare Models on Test Set
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON ON TEST SET")
    print("="*60)
    
    results = compare_models(
        models=models,
        test_loader=test_loader,
        label_names=label_names,
        device=config.training.device
    )
    
    return models, results


def run_visualizations(models, test_loader, config):
    """Run visualization analyses"""
    if not config.experiment.save_plots:
        return
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Attention visualization
    if config.experiment.visualize_attention:
        print("Generating attention visualizations...")
        for model_name, model in models.items():
            if model_name in ['multimodal', 'wsi_only']:  # Only for models with attention
                save_path = os.path.join(config.plot_dir, f"{model_name}_attention.png")
                visualize_attention_weights(
                    model=model,
                    data_loader=test_loader,
                    num_samples=config.experiment.num_attention_samples,
                    mode=model_name,
                    save_path=save_path,
                    device=config.training.device
                )
    
    # Embedding visualization
    if config.experiment.visualize_embeddings:
        print("Generating embedding visualizations...")
        save_path = os.path.join(config.plot_dir, "embeddings_tsne.png")
        visualize_embeddings(
            models=models,
            data_loader=test_loader,
            save_path=save_path,
            device=config.training.device
        )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="TCGA-BRCA Multimodal Classification")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--experiment", type=str, default="default", 
                       choices=["default", "ablation", "cv", "hyperparameter"],
                       help="Type of experiment to run")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load_config(args.config)
    else:
        if args.experiment == "default":
            config = get_default_config()
        elif args.experiment == "ablation":
            from config import get_ablation_config
            config = get_ablation_config()
        elif args.experiment == "cv":
            from config import get_cross_validation_config
            config = get_cross_validation_config()
        else:
            config = get_default_config()
    
    # Save configuration
    config.save_config()
    
    print("TCGA-BRCA Multimodal Classification Pipeline")
    print("="*60)
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Device: {config.training.device}")
    print(f"Output directory: {config.experiment.output_dir}")
    
    # Setup data
    dataset, train_loader, val_loader, test_loader = setup_data_loaders(config)
    
    # Train and evaluate models
    models, results = train_and_evaluate_models(
        dataset, train_loader, val_loader, test_loader, config
    )
    
    # Run visualizations
    run_visualizations(models, test_loader, config)
    
    # Save results
    import json
    results_path = os.path.join(config.experiment.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment completed! Results saved to: {config.experiment.output_dir}")
    
    # Print final summary
    print("\nFINAL RESULTS SUMMARY:")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"{model_name:15s}: {metrics['accuracy']*100:6.2f}% accuracy")


if __name__ == "__main__":
    main()
