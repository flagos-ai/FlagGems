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

"""MetaX-specialized fused inverse-RoPE + FP8 group quant for DeepSeek-V4.

On MetaX, triton lowers `.to(tl.float8e4nv)` to a software sequence that runs
at roughly half the achievable load/store bandwidth (measured ~716 GB/s vs
~1446 GB/s for a pure int8-view copy on C550). This variant replaces it with:

1. A software RNE fp32 -> e4m3fn encoder built from integer bit operations
   (bit-identical to the hardware cast, verified over dense random coverage
   including subnormal-to-normal carry boundaries), storing fp8 bytes through
   an int8 pointer view.
2. A compact RoPE stage: the rope region is loaded once at [ROPE_DIM] width
   and the partner value is recovered arithmetically as
   ``pair_sum - x`` instead of re-loading it through a full-width
   ``x[k ^ 1]`` masked load; cos/sin are read at [HALF_ROPE] width instead of
   full-width masked loads. The rotated values are spliced back through a
   [HEAD_DIM // ROPE_DIM, ROPE_DIM] block-shaped `where`.
3. Reciprocal multiplication (``r = fp8_max / absmax``, ``y = x * r``) instead
   of per-element division. For power-of-two scales the reciprocal is exact,
   so the TMA path stays bit-identical to the reference; on the fp32-scale
   path only ~0.03% of bytes flip by one step at subnormal tie boundaries.

Fallback: shapes whose rope_dim is not a power of two or does not divide
head_dim (never hit by DeepSeek-V4 configs) use the general implementation.
"""

import logging
from typing import Optional, Tuple

import importlib

import torch
import triton
import triton.language as tl

# flag_gems.fused rebinds this attribute to the *function*, so a plain
# `from flag_gums.fused import ...` / `import ... as` would yield the function;
# import_module returns the module itself (and keeps SpecOpRegistrar from
# re-registering the general op under this module).
_general_module = importlib.import_module("flag_gems.fused.fused_inv_rope_fp8_quant")
from flag_gems.utils.device_info import kernel_supports_fp8_e4m3

if kernel_supports_fp8_e4m3():
    SUPPORTED_FP8_DTYPE = torch.float8_e4m3fn
else:
    SUPPORTED_FP8_DTYPE = torch.float32


logger = logging.getLogger(__name__)


def _get_tma_aligned_size(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


@triton.jit
def _encode_fp8e4m3(x, FP8_MAX: tl.constexpr):
    """Round-to-nearest-even fp32 -> e4m3fn encoder, bit-identical to the
    hardware ``.to(tl.float8e4nv)`` cast (NaN inputs excluded)."""
    bits = x.to(tl.int32, bitcast=True)
    neg = bits < 0
    a = (bits & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    a = tl.minimum(a, FP8_MAX)
    Ef = a.to(tl.int32, bitcast=True) >> 23
    # map onto the e4m3 grid: normals scale by 2^(130-Ef), subnormals by 2^9
    pw = ((tl.minimum(130 - Ef, 9) + 127) << 23).to(tl.float32, bitcast=True)
    v = a * pw  # grid points land on integers
    vi = v.to(tl.int32)
    frac = v - vi.to(tl.float32)
    inc = (frac > 0.5) | ((frac == 0.5) & ((vi & 1) != 0))
    m = vi + tl.where(inc, 1, 0)
    m16 = m == 16
    m8sub = (m == 8) & (Ef < 121)  # subnormal rounding up to smallest normal
    carry = m16 | m8sub
    mm = tl.where(carry, 8, m)
    Ef2 = Ef + tl.where(carry, 1, 0)
    byte = tl.where(m >= 8, ((Ef2 - 120) << 3) | (mm - 8), mm)
    return (byte | tl.where(neg, 0x80, 0)).to(tl.int8)


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
    ROPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh

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
            qb_indices = head_in_group * CHUNKS_PER_HEAD + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    cols = tl.arange(0, HEAD_DIM)
    o_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head
    x = tl.load(o_base + cols).to(tl.float32)

    # --- compact inverse RoPE on the trailing rope segment ---
    rope_abs_start: tl.constexpr = HEAD_DIM - ROPE_DIM
    rope_cols = tl.arange(0, ROPE_DIM)
    xr = tl.load(o_base + rope_abs_start + rope_cols).to(tl.float32)
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    j = tl.arange(0, HALF_ROPE)
    cos_v = tl.load(cache_base + j)
    sin_v = tl.load(cache_base + HALF_ROPE + j)

    # pairs are interleaved (even, odd); partner of each lane is the other
    # lane of its pair, recoverable as pair_sum - x without a second load
    pairs = tl.reshape(xr, (HALF_ROPE, 2))
    pair_sum = tl.sum(pairs, axis=1)
    partner = pair_sum[:, None] - pairs
    sgn = tl.where(tl.arange(0, 2)[None, :] == 0, 1.0, -1.0)
    rope_out = pairs * cos_v[:, None] + partner * sin_v[:, None] * sgn
    rope_out = tl.reshape(rope_out, (ROPE_DIM,))
    # splice: [HEAD_DIM] -> [NBLOCKS, ROPE_DIM]; rope is the last block
    xb = tl.reshape(x, (NBLOCKS, ROPE_DIM))
    xb = tl.where(tl.arange(0, NBLOCKS)[:, None] == NBLOCKS - 1, rope_out[None, :], xb)
    x = tl.reshape(xb, (HEAD_DIM,))

    # --- per-group absmax quant ---
    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    if TMA_ALIGNED_SCALES:
        scales = block_absmax * (1.0 / fp8_max)
        scales = tl.math.exp2(
            tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10)))
        )
        recip = 1.0 / scales  # exact for powers of two
    else:
        scales = block_absmax * (1.0 / fp8_max)
        recip = fp8_max / block_absmax
    recip_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(recip, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    y = x * recip_exp
    q_bytes = _encode_fp8e4m3(y, FP8_MAX=fp8_max)

    fp8_base = (
        fp8_ptr.to(tl.pointer_type(tl.int8))
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + head_in_group * HEAD_DIM
    )
    tl.store(fp8_base + cols, q_bytes)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = head_in_group * CHUNKS_PER_HEAD + block_offsets
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
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def _compact_rope_supported(head_dim: int, rope_dim: int) -> bool:
    # tl.arange power-of-two requirements plus the block-splice divisibility;
    # head_dim being a power of two is already required by the general kernel
    if head_dim & (head_dim - 1) != 0:
        return False
    if rope_dim & (rope_dim - 1) != 0:
        return False
    return head_dim % rope_dim == 0


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
    logger.debug("GEMS PER TOKEN GROUP QUANT FP8")

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

    if not _compact_rope_supported(head_dim, rope_dim):
        return _general_module.fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=quant_group_size,
            eps=eps,
            dtype=dtype,
            tma_aligned_scales=tma_aligned_scales,
        )

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
        ROPE_DIM=rope_dim,
        HALF_ROPE=rope_dim // 2,
        NBLOCKS=head_dim // rope_dim,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_warps=1,
        num_stages=1,
    )

    return fp8_q.transpose(0, 1), scale.transpose(0, 1)
