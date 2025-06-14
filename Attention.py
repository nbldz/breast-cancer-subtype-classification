"""
Attention Mechanisms Module

This module implements various attention mechanisms for multimodal learning,
including attention pooling for aggregating patch-level features.

Author: Nabil Hezil
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class AttentionPooling(nn.Module):
    """
    Attention pooling mechanism for aggregating variable-length sequences.
    
    This module computes attention weights for each element in a sequence
    and returns a weighted average representation.
    
    Args:
        dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for attention computation
        dropout (float): Dropout probability (default: 0.1)
        
    Returns:
        tuple: (pooled_output, attention_weights)
            - pooled_output: Weighted average of input features [batch_size, dim]
            - attention_weights: Attention weights [batch_size, seq_len]
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim // 2
        
        self.attention_net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention pooling.
        
        Args:
            x: Input features [batch_size, seq_len, dim]
            mask: Padding mask [batch_size, seq_len] (1 for padded positions)
            
        Returns:
            tuple: (pooled_output, attention_weights)
        """
        # Compute attention scores
        attn_scores = self.attention_net(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.bool(), -1e10)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted average
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, dim]
        
        return pooled_output, attn_weights


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for richer feature representations.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable query token for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention pooling.
        
        Args:
            x: Input features [batch_size, seq_len, dim]
            mask: Padding mask [batch_size, seq_len]
            
        Returns:
            tuple: (pooled_output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand pool query for batch
        pool_query = self.pool_query.expand(batch_size, -1, -1)  # [batch_size, 1, dim]
        
        # Compute Q, K, V
        Q = self.query(pool_query)  # [batch_size, 1, dim]
        K = self.key(x)            # [batch_size, seq_len, dim]
        V = self.value(x)          # [batch_size, seq_len, dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn_scores: [batch_size, num_heads, 1, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask.bool(), -1e10)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch_size, num_heads, 1, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        
        # Output projection
        pooled_output = self.out_proj(attended).squeeze(1)  # [batch_size, dim]
        
        # Average attention weights across heads for visualization
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(1)  # [batch_size, seq_len]
        
        return pooled_output, avg_attn_weights


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for feature enhancement.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input features [batch_size, seq_len, dim]
            mask: Padding mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Enhanced features [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask.bool(), -1e10)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing features from different modalities.
    
    Args:
        dim1 (int): Dimension of first modality
        dim2 (int): Dimension of second modality
        hidden_dim (int): Hidden dimension for attention computation
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, dim1: int, dim2: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hidden_dim
        
        # Projection layers to common dimension
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        
        # Cross-attention layers
        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projections
        self.out_proj1 = nn.Linear(hidden_dim, dim1)
        self.out_proj2 = nn.Linear(hidden_dim, dim2)
        
        self.layer_norm1 = nn.LayerNorm(dim1)
        self.layer_norm2 = nn.LayerNorm(dim2)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-modal attention.
        
        Args:
            x1: First modality features [batch_size, seq_len1, dim1]
            x2: Second modality features [batch_size, seq_len2, dim2]
            
        Returns:
            tuple: (enhanced_x1, enhanced_x2)
        """
        # Project to common dimension
        proj_x1 = self.proj1(x1)  # [batch_size, seq_len1, hidden_dim]
        proj_x2 = self.proj2(x2)  # [batch_size, seq_len2, hidden_dim]
        
        # Cross-attention: x1 attends to x2
        attended_x1, _ = self.cross_attn_1to2(proj_x1, proj_x2, proj_x2)
        
        # Cross-attention: x2 attends to x1
        attended_x2, _ = self.cross_attn_2to1(proj_x2, proj_x1, proj_x1)
        
        # Project back to original dimensions
        out_x1 = self.out_proj1(attended_x1)
        out_x2 = self.out_proj2(attended_x2)
        
        # Residual connections and layer normalization
        enhanced_x1 = self.layer_norm1(x1 + out_x1)
        enhanced_x2 = self.layer_norm2(x2 + out_x2)
        
        return enhanced_x1, enhanced_x2


class AdaptiveAttentionPooling(nn.Module):
    """
    Adaptive attention pooling that can handle varying sequence lengths efficiently.
    
    Args:
        dim (int): Input feature dimension
        temperature (float): Temperature for attention softmax (default: 1.0)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, dim: int, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        
        self.attention_weights = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of adaptive attention pooling.
        
        Args:
            x: Input features [batch_size, seq_len, dim]
            mask: Padding mask [batch_size, seq_len]
            
        Returns:
            tuple: (pooled_output, attention_weights)
        """
        # Compute attention scores using learnable weights
        attn_scores = torch.matmul(x, self.attention_weights) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.bool(), -1e10)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted average
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        
        return pooled_output, attn_weights


def create_attention_module(
    attention_type: str,
    dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of attention modules.
    
    Args:
        attention_type (str): Type of attention ('simple', 'multihead', 'adaptive')
        dim (int): Input feature dimension
        **kwargs: Additional arguments for specific attention types
        
    Returns:
        nn.Module: Attention module
    """
    if attention_type == 'simple':
        return AttentionPooling(dim, **kwargs)
    elif attention_type == 'multihead':
        return MultiHeadAttentionPooling(dim, **kwargs)
    elif attention_type == 'adaptive':
        return AdaptiveAttentionPooling(dim, **kwargs)
    elif attention_type == 'self':
        return SelfAttention(dim, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Example usage and testing
    batch_size, seq_len, dim = 4, 100, 512
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, dim)
    mask = torch.randint(0, 2, (batch_size, seq_len)).float()
    
    # Test different attention mechanisms
    print("Testing Attention Mechanisms...")
    
    # Simple attention pooling
    attn_pool = AttentionPooling(dim)
    pooled, weights = attn_pool(x, mask)
    print(f"Simple Attention - Pooled shape: {pooled.shape}, Weights shape: {weights.shape}")
    
    # Multi-head attention pooling
    multihead_attn = MultiHeadAttentionPooling(dim, num_heads=8)
    pooled_mh, weights_mh = multihead_attn(x, mask)
    print(f"Multi-head Attention - Pooled shape: {pooled_mh.shape}, Weights shape: {weights_mh.shape}")
    
    # Self-attention
    self_attn = SelfAttention(dim, num_heads=8)
    enhanced = self_attn(x, mask)
    print(f"Self Attention - Enhanced shape: {enhanced.shape}")
    
    # Cross-modal attention
    x2 = torch.randn(batch_size, seq_len//2, dim//2)
    cross_attn = CrossModalAttention(dim, dim//2, hidden_dim=256)
    enhanced_x1, enhanced_x2 = cross_attn(x, x2)
    print(f"Cross-modal Attention - Enhanced shapes: {enhanced_x1.shape}, {enhanced_x2.shape}")
    
    print("All attention mechanisms tested successfully!")
