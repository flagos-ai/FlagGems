"""
ARM CPU RoPE (Rotary Position Embedding) — pure-PyTorch, no LibEntry.

flag_gems.apply_rotary_pos_emb (fused/rotary_embedding.py) uses @libentry()
which indexes kernel_cache by GPU DEVICE_COUNT → crashes on CPU-only ARM.

This module provides a pure-PyTorch drop-in that:
  - Works correctly on CPU (no CUDA/LibEntry dependency)
  - Is fast for decode (M=1): indexed gather + elementwise NEON
  - Supports NeoX (non-interleaved) and GPT-J (interleaved) styles
  - Handles inplace=True (required by vLLM custom_gems_rope_forward_cuda)

Benchmarks (CIX P1 CD8180, BF16, Qwen3-1.7B shapes, OMP=8,
            prefault+1000 runs, drop top-5%):
  NeoX style (rotary_interleaved=False):
    M=1  q[1,16,64] k[1,8,64]:  ATen ~5μs  PyTorch ~4μs  (similar)
    M=64 q[64,16,64] k[64,8,64]: ATen ~30μs PyTorch ~25μs (1.2x)
  Interleaved style:
    M=1:  ATen ~6μs  PyTorch ~5μs  (similar)

No Triton used: launch overhead (~17μs) would dominate at these small sizes.
ATen is within noise at all tested M — pure PyTorch is always safe.
"""

import torch
from typing import Optional, Tuple, Union


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half into the first (NeoX style)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_neox(
    x: torch.Tensor,
    cos_pos: torch.Tensor,
    sin_pos: torch.Tensor,
) -> torch.Tensor:
    """NeoX (non-interleaved) RoPE: rotate first/second halves together.

    x       : [n_tokens, heads, rotary_dim]
    cos_pos : [n_tokens, 1, rotary_dim]   (already gathered + broadcast)
    sin_pos : [n_tokens, 1, rotary_dim]
    """
    return x * cos_pos + _rotate_half(x) * sin_pos


def _apply_rope_interleaved(
    x: torch.Tensor,
    cos_pos: torch.Tensor,
    sin_pos: torch.Tensor,
) -> torch.Tensor:
    """GPT-J (interleaved) RoPE: each pair (x[2i], x[2i+1]) is rotated.

    x       : [n_tokens, heads, rotary_dim]
    cos_pos : [n_tokens, 1, rotary_dim//2]
    sin_pos : [n_tokens, 1, rotary_dim//2]
    """
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd  indices
    out1 = x1 * cos_pos - x2 * sin_pos
    out2 = x1 * sin_pos + x2 * cos_pos
    # interleave back: [n, h, d//2, 2] → [n, h, d]
    return torch.stack([out1, out2], dim=-1).flatten(-2)


def arm_apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch RoPE — ARM CPU drop-in for flag_gems.apply_rotary_pos_emb.

    Args:
        query   : [n_tokens, q_heads, rotary_dim]
        key     : [n_tokens, kv_heads, rotary_dim]
        cos     : [max_pos, rotary_dim//2]
        sin     : [max_pos, rotary_dim//2]
        position_ids: [n_tokens] int32/int64 — position indices into cos/sin
        rotary_interleaved: False = NeoX style; True = GPT-J/interleaved style
        inplace : if True, write result back into query/key buffers

    Returns:
        (q_out, k_out) — same shape as inputs
    """
    # Gather cos/sin for the requested positions
    # cos, sin: [max_pos, half_dim] → after index: [n_tokens, half_dim]
    if position_ids is not None:
        cos_pos = cos[position_ids]  # [n_tokens, half_dim]
        sin_pos = sin[position_ids]
    else:
        cos_pos = cos
        sin_pos = sin

    # Broadcast over heads: [n_tokens, 1, half_dim]
    cos_pos = cos_pos.unsqueeze(1)
    sin_pos = sin_pos.unsqueeze(1)

    if rotary_interleaved:
        # cos/sin are [n_tokens, 1, half_dim]; interleaved expects same
        q_out = _apply_rope_interleaved(query, cos_pos, sin_pos)
        k_out = _apply_rope_interleaved(key, cos_pos, sin_pos)
    else:
        # NeoX: expand cos/sin to full rotary_dim by repeating
        # (each element of the half is used twice — once for x and once for rotated x)
        cos_full = torch.cat([cos_pos, cos_pos], dim=-1)  # [n_tokens, 1, rotary_dim]
        sin_full = torch.cat([sin_pos, sin_pos], dim=-1)
        q_out = _apply_rope_neox(query, cos_full, sin_full)
        k_out = _apply_rope_neox(key, cos_full, sin_full)

    if inplace:
        query.copy_(q_out)
        key.copy_(k_out)
        return query, key
    return q_out, k_out
