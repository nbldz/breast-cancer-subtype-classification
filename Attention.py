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
