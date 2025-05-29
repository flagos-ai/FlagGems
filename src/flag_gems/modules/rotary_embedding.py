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
# - _set_cos_sin_cache method in `GemsDeepseekV3YarnRoPE`
#
# Source: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py
# License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn

import flag_gems

logger = logging.getLogger(__name__)

has_c_extension = False  # Disable C extension for now, as we have not implemented c++ wrapper for rotary_embedding yet.

__all__ = [
    "gems_rope_forward",
    "GemsDeepseekV3YarnRoPE",
]


def gems_rope_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
) -> Union[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS CUSTOM ROPE FORWARD")
    # TODO: Implement C++ wrapper for rotary_embedding
    return flag_gems.apply_rotary_pos_emb(
        query, key, cos, sin, position_ids, rotary_interleaved
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


class GemsDeepseekV3YarnRoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) module for Deepseek-V3 with Yarn enhancements.

    This module implements a RoPE variant that supports:
    - Dynamic position cache construction and updating (used in HuggingFace-like models)
    - Position interpolation and extrapolation via YARN (Yield-Aware RoPE Normalization)
    - Optional forward-only mode to freeze cache size for inference
    - Interleaved rotary layout (e.g., GPT-NeoX style) support

    Compared to static RoPE (as used in some vLLM versions), this module builds
    position-dependent cosine/sine caches dynamically at runtime, enabling generality
    for long-context fine-tuning and inference.

    Args:
        dim (int): The rotary embedding dimension (typically equal to head_dim).
        max_position_embeddings (int): Initial maximum position length to precompute cache.
        base (float): Frequency base used to compute inverse frequencies (default 10000).
        device (torch.device or None): Device to place initial buffers on.
        scaling_factor (float): Scaling factor for position extrapolation in Yarn.
        original_max_position_embeddings (int): Original pretraining context size.
        beta_fast (float): Yarn parameter controlling short-term frequency decay.
        beta_slow (float): Yarn parameter controlling long-term frequency floor.
        mscale (float): Yarn multiplier for frequency scaling (usually 1.0).
        mscale_all_dim (float): Yarn scaling baseline across all dims (usually 0 or 1).
        forward_only (bool): If True, disables runtime cache resizing (for inference-only use).

    Inputs:
        query (torch.Tensor): Shape (..., seq_len, head_dim)
        key (torch.Tensor): Shape (..., seq_len, head_dim)
        position_ids (torch.IntTensor, optional): Shape (..., seq_len), positions for cos/sin lookup.
        rotary_interleaved (bool): Whether inputs use interleaved layout (GPT-NeoX-style).
        seq_len (int, optional): Current sequence length (defaults to query.shape[-2])

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed (query, key) tensors with RoPE applied.

    Usage:
        rope = GemsDeepseekV3YarnRoPE(dim=128, forward_only=False)
        q, k = rope(q, k, seq_len=q.shape[-2])
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
        forward_only=False,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.forward_only = forward_only

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = max_position_embeddings if forward_only else None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: Optional[torch.IntTensor] = None,
        rotary_interleaved: bool = False,
        seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = query.shape[-2]

        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            if self.forward_only:
                raise ValueError(
                    f"Requested seq_len={seq_len} exceeds cached length in forward-only mode."
                )
            self._set_cos_sin_cache(
                seq_len=seq_len, device=query.device, dtype=query.dtype
            )

        cos = self.cos_cached[:seq_len].to(dtype=query.dtype)
        sin = self.sin_cached[:seq_len].to(dtype=query.dtype)

        return flag_gems.apply_rotary_pos_emb(
            query, key, cos, sin, position_ids, rotary_interleaved
        )
