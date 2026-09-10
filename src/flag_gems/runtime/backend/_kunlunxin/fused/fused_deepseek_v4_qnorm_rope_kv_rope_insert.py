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

"""Kunlunxin (XPU / TritonXPU) override for
``fused_deepseek_v4_qnorm_rope_kv_rope_insert``.

The generic implementation in ``flag_gems/fused`` cannot be used on this
backend for two independent reasons, both measured on XPU / xpu_arch=3
(probe logs archived under
``harness/results/performance/fused_deepseek_v4_qnorm_rope_kv_rope_insert_xpu6_20260831/``):

1. It does not compile.  Its grid-stride loop is a ``while`` loop and its body
   contains a cross-lane reduction (``tl.sum`` for the RMSNorm).  A reduction
   inside a ``while`` loop makes ``ConvertTritonXPUToLLVM`` abort with
   ``RuntimeError: PassManager::run failed``.

2. Even if it compiled it would be numerically wrong.  It writes the rotated
   dims with 32-lane STRIDE-2 scatter stores.  On this backend such a store
   also writes a 64-element contiguous block into the *next* row.  Every store
   here is therefore a single **contiguous, unmasked** store.

Further backend constraints that shaped this kernel:

* Narrow (64-lane) tiles positioned at column offset 448 return wrong values
  even when every access is affine; the full 512-lane row form is bit exact.
* Masked 2D tiles cost ~54x (a [32,512] masked RMSNorm tile measured 246 ms vs
  4.5 ms unmasked for the same 537 MB), so the *hot* ``x`` load and the store
  stay unmasked; only the two boundary ``+/-1`` partner probes carry a
  compile-time-constant mask whose masked-out lanes are discarded anyway.
* ``tl.reshape`` and ``tl.join`` are unsupported (``out of resource: uni_sram
  ... Required: 0, Hardware limit: 0``).

Performance (2026-09-05): the two non-affine gathers of the previous kernel --
the interleaved-pair partner (``offs ^ 1``) and the ``cos/sin`` table lookup
(``(offs-448)//2``) -- were the whole cost (0.2-3.8 GB/s).  They are replaced
with two purely affine mechanisms:

* **cos/sin** are pre-expanded in Python/torch into ``[N, 512]`` per-lane tables
  (``cos_item`` / ``sin_item``, indexed directly by token), so the kernel reads
  them with a contiguous 512-lane affine load (scalar base + ``arange``).
* **partner** is read with two 512-lane affine loads at ``base +/- 1`` plus a
  data-level ``tl.where`` by lane parity; the two boundary lanes (``col 0`` of
  the very first row and ``col 511`` of the very last) are masked out with a
  compile-time-constant mask and their value is discarded by the parity select
  anyway.

The Q kernel is a ``[TILE_H, 512]`` 2D tile (one program per token x head-block,
``TILE_H``/``NUM_HEADS`` are ``tl.constexpr``).  Keeping ``NUM_HEADS`` a
``constexpr`` is required: a runtime ``num_heads`` multiplier in the row-base
expression silently degrades the whole tile to a non-affine gather
(measured 416 ms vs 4 ms for the identical kernel with ``NUM_HEADS`` constexpr).
The 2D reduction over ``axis=1`` changes the fp32 accumulation order vs the 1D
form; the result differs by at most 1 bf16 ulp (``q_maxabs`` 1.56e-2), well
inside the test tolerance ``ATOL_Q = 5.12e-2``.
"""

import logging

import torch  # noqa: F401
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Default TILE_H for the Q kernel (heads per program).  Picked as the largest
# power of two <= 16 that divides the number of heads; measured optimal range
# on the benchmark shapes is TILE_H in {8, 16, 32} (N=4096: 22.8 ms).
_MAX_TILE_H = 16


def _pick_tile_h(num_heads: int) -> int:
    """Largest power of two <= _MAX_TILE_H that divides num_heads; else 1."""
    tile_h = _MAX_TILE_H
    while tile_h > 1:
        if num_heads % tile_h == 0:
            return tile_h
        tile_h //= 2
    return 1


@triton.jit
def _qnorm_rope_kernel_2d(
    q_ptr,
    cos_item_ptr,
    sin_item_ptr,
    eps,
    num_tokens,
    TILE_H: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
):
    """Per-head RMSNorm (no weight) + GPT-J RoPE on the trailing dims of q.

    Grid is (num_tokens, num_heads // TILE_H); each program handles TILE_H
    contiguous heads of one token.  The rotated output is merged into the
    normed row with ``tl.where`` so the row leaves the kernel through a single
    contiguous unmasked store.
    """
    r = tl.arange(0, TILE_H)
    c = tl.arange(0, HEAD_DIM)
    nope = c < NOPE_DIM
    # 448 is even, so lane parity == parity inside the interleaved pair.
    is_odd = (c & 1) == 1
    sgn = tl.where(is_odd, 1.0, -1.0)

    tok = tl.program_id(0)
    hb = tl.program_id(1)
    rows = tok * NUM_HEADS + hb * TILE_H + r
    base = rows[:, None] * HEAD_DIM + c[None, :]

    x = tl.load(q_ptr + base).to(tl.float32)
    rsqrt_val = tl.math.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)
    # Round to bf16 before the rotation: the reference normalises to bf16
    # first and then rotates, so this keeps the result within 1 bf16 ulp.
    xs = (x * rsqrt_val[:, None]).to(tl.bfloat16).to(tl.float32)

    # cos/sin are pre-expanded [N, HEAD_DIM] tables indexed by token -> affine.
    cos = tl.load(cos_item_ptr + tok * HEAD_DIM + c).to(tl.float32)
    sin = tl.load(sin_item_ptr + tok * HEAD_DIM + c).to(tl.float32)

    # partner via two affine +/-1 loads; boundary lanes (col 0 / col 511) are
    # masked out and their (garbage) value is discarded by the parity select.
    xp1 = tl.load(
        q_ptr + base + 1, mask=(c < HEAD_DIM - 1)[None, :], other=0.0
    ).to(tl.float32)
    xm1 = tl.load(
        q_ptr + base - 1, mask=(c > 0)[None, :], other=0.0
    ).to(tl.float32)
    xp_raw = tl.where(is_odd[None, :], xm1, xp1)
    xp = (xp_raw * rsqrt_val[:, None]).to(tl.bfloat16).to(tl.float32)

    # even lane j : x[j]*cos - x[j+1]*sin      (sgn = -1)
    # odd  lane j : x[j]*cos + x[j-1]*sin      (sgn = +1)
    out = tl.where(
        nope[None, :],
        xs,
        xs * cos[None, :] + xp * sin[None, :] * sgn[None, :],
    )
    tl.store(q_ptr + base, out.to(tl.bfloat16))


@triton.jit
def _kv_rope_insert_kernel(
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    cos_item_ptr,
    sin_item_ptr,
    stride_kv_tok,
    stride_cache_block,
    stride_cache_token,
    n_insert,
    num_progs,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
):
    """GPT-J RoPE on the trailing dims of kv + paged bf16 cache insert.

    No reduction here, so the data dependent ``if slot_id >= 0`` guard is legal
    inside the ``for``-range loop.  kv has one row per token, so the token index
    is the row index and cos_item/sin_item are indexed directly (affine).
    """
    offs = tl.arange(0, HEAD_DIM)
    nope = offs < NOPE_DIM
    sgn = tl.where((offs & 1) == 1, 1.0, -1.0)

    for kv_idx in range(tl.program_id(0), n_insert, num_progs):
        slot_id = tl.load(slot_mapping_ptr + kv_idx)
        if slot_id >= 0:
            kv_base = kv_idx * stride_kv_tok
            d = tl.load(kv_ptr + kv_base + offs).to(tl.float32)
            dp1 = tl.load(
                kv_ptr + kv_base + offs + 1,
                mask=offs < (HEAD_DIM - 1),
                other=0.0,
            ).to(tl.float32)
            dm1 = tl.load(
                kv_ptr + kv_base + offs - 1,
                mask=offs > 0,
                other=0.0,
            ).to(tl.float32)
            dp_raw = tl.where((offs & 1) == 1, dm1, dp1)

            cos = tl.load(cos_item_ptr + kv_idx * HEAD_DIM + offs).to(tl.float32)
            sin = tl.load(sin_item_ptr + kv_idx * HEAD_DIM + offs).to(tl.float32)

            out = tl.where(nope, d, d * cos + dp_raw * sin * sgn)
            cache_off = (slot_id // CACHE_BLOCK_SIZE) * stride_cache_block + (
                slot_id % CACHE_BLOCK_SIZE
            ) * stride_cache_token
            tl.store(k_cache_ptr + cache_off + offs, out.to(tl.bfloat16))


def _build_items(cos_sin_cache, position_ids, head_dim, rope_dim):
    """Pre-expand cos/sin into per-token, per-lane ``[N, HEAD_DIM]`` tables.

    Row ``tok`` of ``cos_item`` holds, at column ``j``, the cosine that the
    rope applies to lane ``j`` of any head of token ``tok`` (0 for the NoPE
    lanes, which are discarded by ``tl.where``).  This turns the in-kernel
    cos/sin lookup into a contiguous affine load.
    """
    half_rope = rope_dim // 2
    nope_dim = head_dim - rope_dim
    device = cos_sin_cache.device
    pair = torch.clamp(
        (torch.arange(head_dim, device=device) - nope_dim) // 2,
        0,
        half_rope - 1,
    )
    cs_tok = cos_sin_cache[position_ids]  # [N, rope_dim] fp32
    cos_item = cs_tok[:, :half_rope][:, pair].contiguous()  # [N, HEAD_DIM]
    sin_item = cs_tok[:, half_rope:][:, pair].contiguous()  # [N, HEAD_DIM]
    return cos_item, sin_item


def fused_deepseek_v4_qnorm_rope_kv_rope_insert(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    """Fused QNorm+RoPE (Q) and RoPE+Insert (KV), BF16 variant.

    Args:
        q: [N, H, 512] bfloat16, modified in-place (RMSNorm + RoPE).
        kv: [N, 512] bfloat16, input KV data.
        k_cache: [num_blocks, block_size, 512] bfloat16, paged KV cache.
        slot_mapping: [N_insert] int64, slot indices for cache insertion.
        position_ids: [N] int64, position indices for RoPE.
        cos_sin_cache: [max_pos, 64] float32, precomputed cos||sin cache.
        eps: RMSNorm epsilon (default 1e-6).
        cache_block_size: KV cache page size (default 16).
    """
    logger.debug("GEMS_KUNLUNXIN FUSED_DEEPSEEK_V4_QNORM_ROPE_KV_ROPE_INSERT")

    head_dim = q.shape[-1]
    rope_dim = cos_sin_cache.shape[-1]
    nope_dim = head_dim - rope_dim

    total_q = q.shape[0] * q.shape[1]
    n_insert = slot_mapping.shape[0]
    if total_q + n_insert == 0:
        return

    cos_item, sin_item = _build_items(
        cos_sin_cache, position_ids, head_dim, rope_dim
    )

    if total_q > 0:
        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        tile_h = _pick_tile_h(num_heads)
        grid_q = (num_tokens, num_heads // tile_h)
        _qnorm_rope_kernel_2d[grid_q](
            q,
            cos_item,
            sin_item,
            eps,
            num_tokens,
            tile_h,
            num_heads,
            head_dim,
            nope_dim,
            num_warps=4,
            num_stages=1,
        )

    if n_insert > 0:
        grid_kv = min(n_insert, 4096)
        _kv_rope_insert_kernel[(grid_kv,)](
            kv,
            k_cache,
            slot_mapping,
            cos_item,
            sin_item,
            kv.stride(0),
            k_cache.stride(0),
            k_cache.stride(1),
            n_insert,
            grid_kv,
            cache_block_size,
            head_dim,
            nope_dim,
            num_warps=1,
            num_stages=1,
        )
