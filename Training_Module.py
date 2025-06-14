"""
Training Module
Contains training functions and utilities for multimodal models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Utility class to track training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
    def get_latest(self, key):
        return self.metrics[key][-1] if self.metrics[key] else None
        
    def get_average(self, key, last_n=None):
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        return np.mean(values) if values else 0
        
    def plot_metrics(self, save_path=None):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        if 'train_loss' in self.metrics:
            axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        if 'val_loss' in self.metrics:
            axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Accuracy plots
        if 'train_acc' in self.metrics:
            axes[0, 1].plot(self.metrics['train_acc'], label='Train Acc')
        if 'val_acc' in self.metrics:
            axes[0, 1].plot(self.metrics['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # F1 Score plots
        if 'train_f1' in self.metrics:
            axes[1, 0].plot(self.metrics['train_f1'], label='Train F1')
        if 'val_f1' in self.metrics:
            axes[1, 0].plot(self.metrics['val_f1'], label='Val F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        # Learning rate
        if 'learning_rate' in self.metrics:
            axes[1, 1].plot(self.metrics['learning_rate'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for evaluation."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=True):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move data to device
        if isinstance(batch, dict):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(**inputs)
        else:
            # Assume batch is (inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics['loss'] = avg_loss
    
    return metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            # Move data to device
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                outputs = model(**inputs)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics['loss'] = avg_loss
    
    return metrics


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001,
                weight_decay=0.01, patience=10, device='cuda', save_path=None):
    """Complete training pipeline."""
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=patience)
    metrics_tracker = MetricsTracker()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        metrics_tracker.update(
            train_loss=train_metrics['loss'],
            train_acc=train_metrics['accuracy'],
            train_f1=train_metrics['f1'],
            val_loss=val_metrics['loss'],
            val_acc=val_metrics['accuracy'],
            val_f1=val_metrics['f1'],
            learning_rate=current_lr
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'metrics': metrics_tracker.metrics
                }, save_path)
                logger.info(f"Saved best model with Val F1: {best_val_f1:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['loss'], model):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training completed!")
    return model, metrics_tracker


def evaluate_model(model, test_loader, device='cuda', class_names=None):
    """Comprehensive model evaluation."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Print detailed metrics
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm
    }
