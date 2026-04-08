"""
Simplified Multi-Head Attention implementation for combination testing.

This module provides a reference implementation of multi-head attention
that can be tested with FlagGems operators.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention implementation.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """

    def __init__(self, d_model, nhead, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, is_causal=False):
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, nhead, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Compute attention scores: (batch, nhead, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Apply custom mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention if available.

    Note: This is a simplified wrapper for testing purposes.
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout

    def forward(self, x, is_causal=False):
        """
        Forward pass using standard attention (fallback if no Flash Attention).

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            is_causal: Whether to use causal masking

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Standard attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output