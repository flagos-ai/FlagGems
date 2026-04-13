"""
Feed-Forward Network implementations for combination testing.

This module provides FFN variants used in Transformer models.
"""

import torch.nn as nn
import torch.nn.functional as F


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network with GELU activation.

    FFN(x) = Linear(ReLU/GeLU(Linear(x)))

    Args:
        d_model: Input/output dimension
        dim_feedforward: Hidden dimension
        activation: Activation function ('relu', 'gelu', 'silu')
        dropout: Dropout probability
    """

    def __init__(self, d_model, dim_feedforward, activation="gelu", dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU-style FFN used in LLaMA and similar models.

    This combines Swish (SiLU) activation with a gating mechanism.

    SwiGLU(x) = (Swish(xW1) ⊙ xV) W2

    Args:
        d_model: Input/output dimension
        dim_feedforward: Hidden dimension (should be 2/3 of typical FFN size)
        dropout: Dropout probability
    """

    def __init__(self, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        # SwiGLU uses gate + value architecture
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)  # Gate projection
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)  # Value projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass with SwiGLU mechanism.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Gate path
        gate = F.silu(self.w1(x))

        # Value path
        value = self.w3(x)

        # Gating
        hidden = gate * value

        # Output projection
        output = self.w2(hidden)
        output = self.dropout(output)

        return output


class GeGLUFFN(nn.Module):
    """
    GeGLU-style FFN combining GELU with gating.

    GeGLU(x) = (GELU(xW1) ⊙ xV) W2

    Args:
        d_model: Input/output dimension
        dim_feedforward: Hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)  # Gate projection
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)  # Value projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass with GeGLU mechanism.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Gate path
        gate = F.gelu(self.w1(x))

        # Value path
        value = self.w3(x)

        # Gating
        hidden = gate * value

        # Output projection
        output = self.w2(hidden)
        output = self.dropout(output)

        return output
