"""
Complete Transformer Block implementation for combination testing.

This module provides a full Transformer encoder/decoder layer
that combines attention, FFN, and normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .ffn import StandardFFN, SwiGLUFFN, GeGLUFFN


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (LLaMA-style).

    Unlike LayerNorm, RMSNorm does not re-center the activations
    and only re-scales them using the root mean square.
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block.

    Structure:
        x + Dropout(Attention(LayerNorm(x)))
        x + Dropout(FFN(LayerNorm(x)))

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        activation: FFN activation ('relu', 'gelu', 'silu')
        norm_eps: Epsilon for layer normalization
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        norm_eps=1e-5,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward network
        self.ffn = StandardFFN(d_model, dim_feedforward, activation, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, is_causal=False):
        """
        Forward pass of Transformer block.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=mask, is_causal=is_causal)
        x = residual + self.dropout1(x)

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)

        return x


class LLaMABlock(nn.Module):
    """
    LLaMA-style Transformer block with RMSNorm and SwiGLU.

    Structure:
        x + Attention(RMSNorm(x))
        x + SwiGLU-FFN(RMSNorm(x))

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension (typically 2.67x d_model)
        dropout: Dropout probability
        norm_eps: Epsilon for RMS normalization
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, norm_eps=1e-5):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, dim_feedforward, dropout)

        # RMS Normalization
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, mask=None, is_causal=False):
        """
        Forward pass of LLaMA block.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=mask, is_causal=is_causal)
        x = residual + x

        # SwiGLU FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block with cross-attention.

    Structure:
        x + Dropout(SelfAttention(LayerNorm(x)))
        x + Dropout(CrossAttention(LayerNorm(x), memory))
        x + Dropout(FFN(LayerNorm(x)))

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
        activation: FFN activation
        norm_eps: Epsilon for normalization
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        norm_eps=1e-5,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward network
        self.ffn = StandardFFN(d_model, dim_feedforward, activation, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, is_causal=False):
        """
        Forward pass of decoder block.

        Args:
            x: Target sequence of shape (batch, tgt_len, d_model)
            memory: Encoder output of shape (batch, src_len, d_model)
            tgt_mask: Target mask
            memory_mask: Memory mask
            is_causal: Whether to use causal masking for target

        Returns:
            Output of shape (batch, tgt_len, d_model)
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask=tgt_mask, is_causal=is_causal)
        x = residual + self.dropout1(x)

        # Cross-attention with pre-norm (Q from target, K/V from memory)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, kv_input=memory, mask=memory_mask, is_causal=False)
        x = residual + self.dropout2(x)

        # FFN with pre-norm
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + self.dropout3(x)

        return x