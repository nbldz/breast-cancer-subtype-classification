"""
Utility functions for TCGA-BRCA multimodal classification
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from copy import deepcopy


def evaluate_model(model, loader, label_names=None, mode="multimodal", verbose=True, device='cuda'):
    """
    Evaluate model performance on given data loader
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing evaluation data
        label_names: Dictionary mapping label indices to names
        mode: Evaluation mode ("multimodal", "wsi_only", or "rna_only")
        verbose: Whether to print detailed results
        device: Device to run evaluation on
    
    Returns:
        tuple: (accuracy, average_loss)
    """
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0
    total_samples = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            if mode == "multimodal":
                x = batch['histology'].to(device)
                m = batch['mask'].to(device)
                r = batch['rna'].to(device)
                y = batch['label'].to(device)
                logits = model(x, m, r)
            elif mode == "rna_only":
                r = batch['rna'].to(device)
                y = batch['label'].to(device)
                logits = model(r)
            elif mode == "wsi_only":
                x = batch['histology'].to(device)
                m = batch['mask'].to(device)
                y = batch['label'].to(device)
                logits = model(x, m)
            else:
                raise ValueError(f"Unsupported evaluation mode: {mode}")
            
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    if not all_true or not all_preds:
        print("⚠️ No predictions or ground truth available for evaluation.")
        return 0, 0

    avg_loss = total_loss / total_samples
    acc = np.mean(np.array(all_preds) == np.array(all_true))
    
    if verbose and label_names is not None:
        print_evaluation_results(all_true, all_preds, label_names, mode)
    
    return acc, avg_loss


def print_evaluation_results(all_true, all_preds, label_names, mode):
    """Print detailed evaluation results including classification report and confusion matrix"""
    # Classification report
    unique_labels = sorted(set(all_true) | set(all_preds))
    report = classification_report(
        all_true, all_preds, 
        labels=unique_labels, 
        target_names=[label_names[i] for i in unique_labels]
    )
    print(f"\nClassification Report ({mode}):\n{report}")

    # Confusion matrix with percentages
    cm = confusion_matrix(all_true, all_preds, labels=unique_labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        xticklabels=[label_names[i] for i in unique_labels],
        yticklabels=[label_names[i] for i in unique_labels]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({mode}) - Percentage")
    plt.savefig(f"{mode}_confusion_matrix_percent.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    acc = np.mean(np.array(all_preds) == np.array(all_true))
    print(f"\nOverall Accuracy: {acc*100:.2f}%")


def create_label_mapping(subtypes_df):
    """
    Create mapping from label indices to subtype names
    
    Args:
        subtypes_df: DataFrame containing subtype information
        
    Returns:
        dict: Mapping from label index to subtype name
    """
    label_names = {}
    for item in subtypes_df.itertuples():
        label_names[item.label] = item.BRCA_Subtype_PAM50
    return label_names


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def setup_device():
    """Setup and return the appropriate device for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
