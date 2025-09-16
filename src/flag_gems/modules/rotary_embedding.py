# Copyright (c) 2025 FlagGems. All rights reserved.
#
# This module is designed to provide a unified interface for various Rotary Position Embedding (RoPE) implementations.
# Currently, it includes only the Yarn-style RoPE used by DeepSeek,
# but support for other variants will be added progressively.
#
# The following components are adapted from DeepSeek-R1:
# - yarn_find_correction_dim
# - yarn_find_correction_range
# - yarn_get_mscale
# - yarn_linear_ramp_mask
# - _set_cos_sin_cache method in `GemsDeepseekYarnRoPE`
#
# Source: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py
# License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import flag_gems
from flag_gems.config import use_c_extension

logger = logging.getLogger(__name__)

__all__ = [
    "gems_rope_forward",
    "GemsDeepseekYarnRoPE",
    "GemsRope",
]


def gems_rope_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
) -> Union[torch.Tensor, torch.Tensor]:
    if use_c_extension:
        logger.debug("GEMS CUSTOM ROPE FORWARD(C EXTENSION)")
        if inplace:
            torch.ops.flag_gems.rotary_embedding_inplace(
                query, key, cos, sin, position_ids, rotary_interleaved
            )
            return query, key
        else:
            return torch.ops.flag_gems.rotary_embedding(
                query, key, cos, sin, position_ids, rotary_interleaved
            )
    else:
        logger.debug("GEMS CUSTOM ROPE FORWARD")
        # Fallback to pure python implementation
        return flag_gems.apply_rotary_pos_emb(
            query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )


class GemsRope(nn.Module):
    """
    Base class for Rotary Position Embedding (RoPE) modules.
    This class is intended to be subclassed for specific RoPE implementations.

    Args:
        rotary_dim (int): The rotary embedding dimension (typically equal to head_dim).
        max_position_embeddings (int): Initial maximum position length to precompute cache.
        base (float): Frequency base used to compute inverse frequencies (default 10000).
        device (torch.device or None): Device to place initial buffers on.
        dtype (torch.dtype): Data type for the cos/sin cache buffers.
        rotary_interleaved (bool): Whether to use interleaved rotary layout (GPT-NeoX-style).

    Inputs:
        query (torch.Tensor): Shape (..., q_heads, head_dim)
        key (torch.Tensor): Shape (..., k_heads, head_dim)
        position_ids (torch.IntTensor, optional): Shape (..., seq_len), positions for cos/sin lookup.
        inplace (bool): If True, modifies query and key in place (default False).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed (query, key) tensors with RoPE applied.
    """

    def __init__(
        self,
        rotary_dim,
        max_position_embeddings,
        base,
        rotary_interleaved,
        dtype,
        device,
    ):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rotary_interleaved = rotary_interleaved
        self.dtype = dtype
        self.device = device
        self._set_cos_sin_cache()

    def _compute_inv_freq(self) -> torch.Tensor:
        """
        Compute the inverse frequency tensor: shape [dim/2]
        """
        return 1.0 / (
            self.base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=self.device
                )
                / self.rotary_dim
            )
        )

    def _set_cos_sin_cache(self):
        """
        Default implementation of rotary embeddings (vanilla RoPE).
        Can be overridden in subclasses for NTK, YaRN, etc.
        """
        inv_freq = self._compute_inv_freq()
        t = torch.arange(
            self.max_position_embeddings, device=self.device, dtype=torch.float32
        )
        freqs = torch.outer(t, inv_freq)  # [max_position_embeddings, rotary_dim // 2]

        self.register_buffer(
            "cos_cached", freqs.cos().to(self.dtype), persistent=False
        )  # [max_position_embeddings, rotary_dim // 2]
        self.register_buffer(
            "sin_cached", freqs.sin().to(self.dtype), persistent=False
        )  # [max_position_embeddings, rotary_dim // 2]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: Optional[torch.IntTensor] = None,
        inplace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "cos_cached") or not hasattr(self, "sin_cached"):
            self._set_cos_sin_cache()

        return gems_rope_forward(
            query,
            key,
            self.cos_cached,
            self.sin_cached,
            position_ids,
            self.rotary_interleaved,
            inplace,
        )


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class GemsDeepseekYarnRoPE(GemsRope):
    """
    Yarn-based Rotary Position Embedding (RoPE) for DeepSeek models.
    Args:
        scaling_factor (float): Scaling factor for Yarn extrapolation.
        original_max_position_embeddings (int): Original pretraining context size.
        beta_fast (float): Controls rapid frequency decay.
        beta_slow (float): Controls slow frequency decay.
        mscale (float): Multiplicative scale factor for selected frequencies.
        mscale_all_dim (float): Global multiplicative baseline.
    """

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        rotary_interleaved: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            rotary_interleaved=rotary_interleaved,
            dtype=dtype,
            device=device,
        )

    def _compute_inv_freq(self) -> torch.Tensor:
        freq_extra = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=self.device
                )
                / self.rotary_dim
            )
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=self.device
                )
                / self.rotary_dim
            )
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.original_max_position_embeddings,
        )

        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2).to(
            device=self.device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        return inv_freq

    def _set_cos_sin_cache(self):
        inv_freq = self._compute_inv_freq()
        # self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_position_embeddings, device=self.device, dtype=torch.float32
        )
        freqs = torch.outer(t, inv_freq)  # [max_position_embeddings, rotary_dim // 2]

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        self.register_buffer(
            "cos_cached", (freqs.cos() * _mscale).to(self.dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (freqs.sin() * _mscale).to(self.dtype), persistent=False
        )
