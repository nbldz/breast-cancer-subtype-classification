"""
Multimodal Classification Models Module
Contains various neural network architectures for multimodal classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, MultiHeadAttention, CrossAttention


class EarlyFusionCNN(nn.Module):
    """Early fusion CNN model that concatenates features early in the network."""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EarlyFusionCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class LateFusionCNN(nn.Module):
    """Late fusion CNN model that processes modalities separately before fusion."""
    
    def __init__(self, audio_dim, visual_dim, text_dim, hidden_dim, num_classes):
        super(LateFusionCNN, self).__init__()
        
        # Audio branch
        self.audio_conv1 = nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1)
        self.audio_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        
        # Visual branch
        self.visual_conv1 = nn.Conv1d(visual_dim, hidden_dim, kernel_size=3, padding=1)
        self.visual_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.visual_pool = nn.AdaptiveAvgPool1d(1)
        
        # Text branch
        self.text_conv1 = nn.Conv1d(text_dim, hidden_dim, kernel_size=3, padding=1)
        self.text_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion layer
        self.fusion_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, audio, visual, text):
        # Process audio
        audio = F.relu(self.audio_conv1(audio))
        audio = F.relu(self.audio_conv2(audio))
        audio = self.audio_pool(audio).squeeze(-1)
        
        # Process visual
        visual = F.relu(self.visual_conv1(visual))
        visual = F.relu(self.visual_conv2(visual))
        visual = self.visual_pool(visual).squeeze(-1)
        
        # Process text
        text = F.relu(self.text_conv1(text))
        text = F.relu(self.text_conv2(text))
        text = self.text_pool(text).squeeze(-1)
        
        # Fusion
        fused = torch.cat([audio, visual, text], dim=1)
        fused = F.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)
        
        return self.classifier(fused)


class AttentionFusionModel(nn.Module):
    """Attention-based fusion model using cross-modal attention mechanisms."""
    
    def __init__(self, audio_dim, visual_dim, text_dim, hidden_dim, num_classes, num_heads=8):
        super(AttentionFusionModel, self).__init__()
        
        # Feature projections
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Self-attention for each modality
        self.audio_self_attn = SelfAttention(hidden_dim, num_heads)
        self.visual_self_attn = SelfAttention(hidden_dim, num_heads)
        self.text_self_attn = SelfAttention(hidden_dim, num_heads)
        
        # Cross-modal attention
        self.cross_attn = CrossAttention(hidden_dim, num_heads)
        
        # Fusion and classification
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio, visual, text):
        # Project to common dimension
        audio = self.audio_proj(audio)
        visual = self.visual_proj(visual)
        text = self.text_proj(text)
        
        # Self-attention for each modality
        audio_attn = self.audio_self_attn(audio)
        visual_attn = self.visual_self_attn(visual)
        text_attn = self.text_self_attn(text)
        
        # Cross-modal attention
        audio_cross = self.cross_attn(audio_attn, torch.cat([visual_attn, text_attn], dim=1))
        visual_cross = self.cross_attn(visual_attn, torch.cat([audio_attn, text_attn], dim=1))
        text_cross = self.cross_attn(text_attn, torch.cat([audio_attn, visual_attn], dim=1))
        
        # Global pooling
        audio_pooled = torch.mean(audio_cross, dim=1)
        visual_pooled = torch.mean(visual_cross, dim=1)
        text_pooled = torch.mean(text_cross, dim=1)
        
        # Fusion
        fused = torch.cat([audio_pooled, visual_pooled, text_pooled], dim=1)
        fused = self.layer_norm(F.relu(self.fusion_layer(fused)))
        fused = self.dropout(fused)
        
        return self.classifier(fused)


class HierarchicalFusionModel(nn.Module):
    """Hierarchical fusion model that combines modalities at multiple levels."""
    
    def __init__(self, audio_dim, visual_dim, text_dim, hidden_dim, num_classes):
        super(HierarchicalFusionModel, self).__init__()
        
        # Low-level feature extraction
        self.audio_low = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.visual_low = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.text_low = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Mid-level fusion
        self.audio_visual_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # High-level fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, audio, visual, text):
        # Low-level processing
        audio_low = self.audio_low(audio)
        visual_low = self.visual_low(visual)
        text_low = self.text_low(text)
        
        # Mid-level fusion (audio + visual)
        audio_visual = torch.cat([audio_low, visual_low], dim=-1)
        audio_visual_fused = self.audio_visual_fusion(audio_visual)
        
        # High-level fusion (audio_visual + text)
        final_input = torch.cat([audio_visual_fused, text_low], dim=-1)
        final_fused = self.final_fusion(final_input)
        
        return self.classifier(final_fused)


class TransformerMultimodalModel(nn.Module):
    """Transformer-based multimodal model using multi-head attention."""
    
    def __init__(self, audio_dim, visual_dim, text_dim, hidden_dim, num_classes, 
                 num_heads=8, num_layers=6):
        super(TransformerMultimodalModel, self).__init__()
        
        # Modality embeddings
        self.audio_embedding = nn.Linear(audio_dim, hidden_dim)
        self.visual_embedding = nn.Linear(visual_dim, hidden_dim)
        self.text_embedding = nn.Linear(text_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, audio, visual, text):
        batch_size = audio.size(0)
        
        # Embed modalities
        audio_emb = self.audio_embedding(audio).unsqueeze(1)  # [B, 1, H]
        visual_emb = self.visual_embedding(visual).unsqueeze(1)  # [B, 1, H]
        text_emb = self.text_embedding(text).unsqueeze(1)  # [B, 1, H]
        
        # Concatenate modalities
        x = torch.cat([audio_emb, visual_emb, text_emb], dim=1)  # [B, 3, H]
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Apply transformer layers
        x = x.transpose(0, 1)  # [3, B, H] for transformer
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.transpose(0, 1)  # [B, 3, H]
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [B, H]
        x = self.dropout(x)
        
        return self.classifier(x)
