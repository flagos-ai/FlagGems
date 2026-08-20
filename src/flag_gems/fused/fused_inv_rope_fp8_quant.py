# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_device_capability

if torch_device_fn.is_available() and get_device_capability() >= (9, 0):
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


def _get_tma_aligned_size(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scales = block_absmax * (1.0 / fp8_max)
    if TMA_ALIGNED_SCALES:
        scales = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    tma_aligned_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton draft of DeepSeek-V4 fused inverse-RoPE + FP8 group quant.

    Args:
        o: [num_tokens, num_heads, head_dim]
        positions: [num_tokens]
        cos_sin_cache: [max_position, rope_dim] laid out as cos || sin

    Returns:
        o_fp8: [num_tokens, n_groups, heads_per_group * head_dim]
        o_scale: [num_tokens, n_groups, num_scale_blocks] or packed UE8M0 view
    """
    logger.debug("GEMS FUSED INV ROPE FP8 QUANT")

    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert fp8_dtype == torch.float8_e4m3fn, "only torch.float8_e4m3fn is supported"
    assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
    assert positions.ndim == 1, "`positions` must be 1D"
    assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
    assert o.stride(-1) == 1, "head_dim must be contiguous"
    assert positions.shape[0] == o.shape[0], "positions and o token count mismatch"

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    chunks_per_head = head_dim // quant_group_size
    if tma_aligned_scales:
        assert (
            chunks_per_head <= 4
        ), "packed UE8M0 path currently expects at most 4 scale blocks per head"

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    tma_aligned_t = _get_tma_aligned_size(num_tokens, 4)

    if tma_aligned_scales:
        scale_inner = (num_scale_blocks + 3) // 4
        scale_dtype = torch.int32
    else:
        scale_inner = num_scale_blocks
        scale_dtype = torch.float32

    finfo = torch.finfo(fp8_dtype)
    fp8_q = torch.empty((n_groups, num_tokens, d), dtype=fp8_dtype, device=o.device)
    scale = torch.empty(
        n_groups * scale_inner * tma_aligned_t,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )

    grid = (tma_aligned_t, n_groups * heads_per_group)
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_q,
        scale,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_q.stride(0),
        fp8_stride_token=fp8_q.stride(1),
        scale_stride_group=scale.stride(0),
        scale_stride_k=scale.stride(2),
        fp8_max=finfo.max,
        eps=eps,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_warps=1,
        num_stages=1,
    )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)
