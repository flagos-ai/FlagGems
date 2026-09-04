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
backend for two independent reasons, both measured on XPU 6 / xpu_arch=3
(probe logs are archived under
``harness/results/performance/fused_deepseek_v4_qnorm_rope_kv_rope_insert_xpu6_20260831/``):

1. It does not compile.  Its grid-stride loop is a ``while`` loop and its body
   contains a cross-lane reduction (``tl.sum`` for the RMSNorm).  A reduction
   inside a ``while`` loop makes ``ConvertTritonXPUToLLVM`` abort with
   ``RuntimeError: PassManager::run failed``.  One-factor-at-a-time bisection:

       while + tl.sum (nothing else)          -> FAIL
       for   + tl.sum (nothing else)          -> OK
       for   + if/else + tl.sum               -> FAIL
       no loop + if/else + tl.sum             -> OK

   So the reduction must sit at the top level of the kernel or directly inside a
   ``for``-range loop, never inside a loop *and* a conditional.

2. Even if it compiled it would be numerically wrong.  It writes the rotated
   dims with 32-lane STRIDE-2 scatter stores
   ``tl.store(ptr + base + 448 + 2*ho, v)``.  On this backend such a store also
   writes a 64-element contiguous block into the *next* row: a sentinel run with
   rows pre-filled to -1 showed row 0's rope store silently clobbering row 1
   columns 0..63 (64 mismatches at grid=1, 448 at grid=8).  Every store here is
   therefore a single **contiguous, unmasked** store.

Further backend constraints that shaped this kernel:

* Narrow (64-lane) tiles positioned at column offset 448 return wrong values
  even when every access is affine; the full 512-lane row form is bit exact.
* Masked 2D tiles cost ~54x (a [32,512] masked RMSNorm tile measured 246 ms vs
  4.5 ms unmasked for the same 537 MB), so masks are avoided on the hot path.
* ``tl.reshape`` and ``tl.join`` are unsupported (``out of resource: uni_sram
  ... Required: 0, Hardware limit: 0``).

Known cost: the rope needs the partner element of each interleaved pair, which
is a non-affine gather (``offs ^ 1``), and the cos/sin lookup is a non-affine
gather too.  Non-affine gathers run at ~0.2-3.8 GB/s here, which puts this
kernel below the torch reference on the large benchmark shapes.  It is kept
because it is the correctness fix: without it every call raises.
"""

import logging

import torch  # noqa: F401
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Cap on the number of programs; the kernels use a `for`-range grid-stride loop
# so any cap is functionally equivalent.  Measured: grid 12 .. M give the same
# latency within noise, so a modest cap keeps launch resources bounded.
MAX_GRID = 4096


@triton.jit
def _qnorm_rope_kernel(
    q_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    stride_q_tok,
    stride_q_head,
    stride_cos_sin_pos,
    eps,
    num_q_items,
    num_progs,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Per-head RMSNorm (no weight) + GPT-J RoPE on the trailing dims of q.

    One full 512-lane row per loop iteration.  The rotated output is merged into
    the normed row with ``tl.where`` so the row leaves the kernel through a
    single contiguous unmasked store.
    """
    offs = tl.arange(0, HEAD_DIM)
    nope = offs < NOPE_DIM
    # Pair index of each rope lane; clamped to 0 on the NoPE lanes so the
    # address is always inside cos_sin_cache (the value is discarded by
    # `tl.where` below).
    pair_idx = tl.where(nope, 0, (offs - NOPE_DIM) // 2)
    # 448 is even, so lane parity == parity inside the interleaved pair.
    is_odd = (offs & 1) == 1
    sgn = tl.where(is_odd, 1.0, -1.0)
    partner = offs ^ 1  # always inside [0, HEAD_DIM), no mask needed

    for item_id in range(tl.program_id(0), num_q_items, num_progs):
        tok_idx = item_id // NUM_HEADS
        base = tok_idx * stride_q_tok + (item_id % NUM_HEADS) * stride_q_head

        x = tl.load(q_ptr + base + offs).to(tl.float32)
        rsqrt_val = tl.math.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        # Round to bf16 before the rotation: the reference normalises to bf16
        # first and then rotates, so this keeps the result bit identical.
        xs = (x * rsqrt_val).to(tl.bfloat16).to(tl.float32)
        xp = (
            (tl.load(q_ptr + base + partner).to(tl.float32) * rsqrt_val)
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        cs = (
            cos_sin_cache_ptr + tl.load(position_ids_ptr + tok_idx) * stride_cos_sin_pos
        )
        cos = tl.load(cs + pair_idx)
        sin = tl.load(cs + HALF_ROPE + pair_idx)

        # even lane j : x[j]*cos - x[j+1]*sin      (sgn = -1)
        # odd  lane j : x[j]*cos + x[j-1]*sin      (sgn = +1)
        out = tl.where(nope, xs, xs * cos + xp * sin * sgn)
        tl.store(q_ptr + base + offs, out.to(tl.bfloat16))


@triton.jit
def _kv_rope_insert_kernel(
    kv_ptr,
    k_cache_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    stride_kv_tok,
    stride_cache_block,
    stride_cache_token,
    stride_cos_sin_pos,
    n_insert,
    num_progs,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """GPT-J RoPE on the trailing dims of kv + paged bf16 cache insert.

    No reduction here, so the data dependent ``if slot_id >= 0`` guard is legal
    inside the ``for``-range loop.
    """
    offs = tl.arange(0, HEAD_DIM)
    nope = offs < NOPE_DIM
    pair_idx = tl.where(nope, 0, (offs - NOPE_DIM) // 2)
    sgn = tl.where((offs & 1) == 1, 1.0, -1.0)
    partner = offs ^ 1

    for kv_idx in range(tl.program_id(0), n_insert, num_progs):
        slot_id = tl.load(slot_mapping_ptr + kv_idx)
        if slot_id >= 0:
            kv_base = kv_idx * stride_kv_tok
            d = tl.load(kv_ptr + kv_base + offs).to(tl.float32)
            dp = tl.load(kv_ptr + kv_base + partner).to(tl.float32)

            cs = (
                cos_sin_cache_ptr
                + tl.load(position_ids_ptr + kv_idx) * stride_cos_sin_pos
            )
            cos = tl.load(cs + pair_idx)
            sin = tl.load(cs + HALF_ROPE + pair_idx)

            out = tl.where(nope, d, d * cos + dp * sin * sgn)
            cache_off = (slot_id // CACHE_BLOCK_SIZE) * stride_cache_block + (
                slot_id % CACHE_BLOCK_SIZE
            ) * stride_cache_token
            tl.store(k_cache_ptr + cache_off + offs, out.to(tl.bfloat16))


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
    half_rope = rope_dim // 2
    nope_dim = head_dim - rope_dim

    total_q = q.shape[0] * q.shape[1]
    n_insert = slot_mapping.shape[0]
    if total_q + n_insert == 0:
        return

    if total_q > 0:
        grid_q = min(total_q, MAX_GRID)
        _qnorm_rope_kernel[(grid_q,)](
            q,
            position_ids,
            cos_sin_cache,
            q.stride(0),
            q.stride(1),
            cos_sin_cache.stride(0),
            eps,
            total_q,
            grid_q,
            q.shape[1],
            head_dim,
            nope_dim,
            half_rope,
            num_warps=1,
            num_stages=1,
        )

    if n_insert > 0:
        grid_kv = min(n_insert, MAX_GRID)
        _kv_rope_insert_kernel[(grid_kv,)](
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            kv.stride(0),
            k_cache.stride(0),
            k_cache.stride(1),
            cos_sin_cache.stride(0),
            n_insert,
            grid_kv,
            cache_block_size,
            head_dim,
            nope_dim,
            half_rope,
            num_warps=1,
            num_stages=1,
        )
