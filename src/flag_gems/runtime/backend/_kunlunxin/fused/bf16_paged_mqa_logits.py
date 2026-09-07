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

"""BF16 Paged MQA Logits — Kunlunxin (XPU) implementation.

Problem (measured on XPU, triton 3.6.0 Baidu branch, 2026-09-06):
  The generic implementation in ``flag_gems.fused.bf16_paged_mqa_logits``
  fuses the whole computation into one kernel:

      scores = tl.dot(k_block, tl.trans(q_block))   # -> triton_sdnn.mma
      scores = tl.maximum(scores, 0.0)              # ReLU
      logits = tl.sum(scores * w[None, :], axis=1)  # -> ew reduce_sum

  On this backend the fused kernel **does not compile**:
  ``ConvertTritonSDNNToLLVM`` hard-fails (``llvm::SmallVector::operator[]``
  assertion ``idx < size()`` / ``PassManager::run failed`` at the
  ``triton_sdnn.ew maximum`` op; or ``failed to legalize operation
  'linalg.generic'`` when the frontend emits ``tl.where``).  Every
  formulation of an elementwise min/max/select on the MMA result crashes
  the pass, isolated one by one: ``tl.maximum``, ``tl.where``,
  ``s * (s > 0).to(f32)``, ``(s + tl.abs(s)) * 0.5``.
  Only ``ew mul`` / ``ew reduce_sum`` / ``ew fill`` on the MMA result and a
  plain ``tl.dot`` lower cleanly.

Fix:
  Split the computation into two kernels with an explicit global-memory
  round trip of the [P, H] scores tile (both stages verified to compile and
  match the PyTorch reference on the full test matrix):

    1. ``_mqa_logits_scores_kernel``: the GEMM only
         scores[row, kv_pos:kv_pos+64, :] = K[64, D] @ Q[H, D]^T   (fp32 acc)
    2. ``_mqa_logits_combine_kernel``: pure elementwise + row reduce
         logits[row, pos] = sum_h(relu(scores[row, pos, h]) * w[row, h])

  Stage 2 is a plain (non-SDNN) Triton kernel, so ``tl.maximum`` is legal
  there — but only with a 32-lane H sub-tile when combined with the
  ``if kv_pos >= ctx_len: return`` guard (a guarded [64, 64] tile hits an
  ``uni_sram`` OOM / vvmaxnumf type error / pass failure; the unguarded
  [64, 32] and split [64, 32]x2 forms are the only ones that compile).

  Notes:
  - The early ``if kv_pos >= ctx_len: return`` guard must stay *before* the
    ``block_table`` load: the launch grid uses ``ceil(max_ctx/64)`` blocks
    while ``block_table`` rows hold ``ceil(ctx/64)`` entries, so the guard is
    what keeps the block-table address in bounds on the tail blocks.
  - ``num_warps=4`` for both kernels (``num_warps=8`` on the guarded combine
    fails on this backend; measured).
  - The ``eviction_policy`` hints of the generic version are dropped
    (cosmetic only; keeps the kernels free of extra attributes).
  - ``clean_logits`` keeps the generic host-side loop semantics.
"""

import logging
import sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# block_size = 64 hardcoded for both specializations
_BLOCK = 64


@triton.jit
def _mqa_logits_scores_kernel(
    q_ptr,
    kv_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    scores_ptr,
    next_n,
    max_ctx,
    stride_bt,
    H: tl.constexpr,
):
    """scores[row, kv_pos:kv_pos+64, :] = K[64, 128] @ Q[H, 128]^T (fp32)."""
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    kv_pos = pid_blk * 64
    if kv_pos >= ctx_len:
        return

    h_range = tl.arange(0, H)
    d_range = tl.arange(0, 128)
    pos_range = tl.arange(0, 64)

    # Load Q [H, 128] bf16 -> transpose to [128, H]
    q_base = pid_row * (H * 128)
    q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
    q_block = tl.load(q_ptr + q_offs)
    q_t = tl.trans(q_block)

    # Load K [64, 128] bf16 from paged cache
    b_idx = pid_row // next_n
    phys_blk = tl.load(block_table_ptr + b_idx * stride_bt + pid_blk)
    k_offs = phys_blk * (64 * 128) + pos_range[:, None] * 128 + d_range[None, :]
    k_block = tl.load(kv_cache_ptr + k_offs)

    # GEMM: K[64,128] @ Q^T[128,H] -> scores[64,H] fp32
    scores = tl.dot(k_block, q_t)

    tl.store(
        scores_ptr
        + pid_row * (max_ctx * H)
        + kv_pos * H
        + pos_range[:, None] * H
        + h_range[None, :],
        scores,
    )


@triton.jit
def _mqa_logits_combine_kernel(
    scores_ptr,
    weights_ptr,
    context_lens_ptr,
    logits_ptr,
    max_ctx,
    H: tl.constexpr,
):
    """logits[row, pos] = sum_h(relu(scores[row, pos, h]) * w[row, h]).

    H is processed in 32-lane sub-tiles (see module docstring).
    """
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    kv_pos = pid_blk * 64
    if kv_pos >= ctx_len:
        return

    pos_range = tl.arange(0, 64)
    acc = tl.zeros((64,), dtype=tl.float32)
    for hh in tl.static_range(0, H, 32):
        h_range = tl.arange(0, 32)
        s_offs = (
            pid_row * (max_ctx * H)
            + kv_pos * H
            + pos_range[:, None] * H
            + (hh + h_range)[None, :]
        )
        scores = tl.load(scores_ptr + s_offs)
        scores = tl.maximum(scores, 0.0)
        w = tl.load(weights_ptr + pid_row * H + hh + h_range)
        acc = acc + tl.sum(scores * w[None, :], axis=1)

    out_base = pid_row * max_ctx + kv_pos
    # Store (mask only the last partial block)
    if kv_pos + 64 <= ctx_len:
        tl.store(logits_ptr + out_base + pos_range, acc)
    else:
        mask = pos_range < (ctx_len - kv_pos)
        tl.store(logits_ptr + out_base + pos_range, acc, mask=mask)


def bf16_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_metadata,
    max_context_len,
    clean_logits=False,
    logits_dtype=torch.float32,
):
    """BF16 Paged MQA Logits — Kunlunxin (XPU) two-kernel implementation.

    Computes weighted ReLU attention logits on paged KV cache:
      logits[row, pos] = sum_h( relu( q[b,n,h,:] . K[pos,:] ) * w[row, h] )
    """
    logger.debug("GEMS_KUNLUNXIN BF16_PAGED_MQA_LOGITS")

    B, next_n, H, D = q.shape
    total_tokens = B * next_n

    logits = torch.empty(
        total_tokens,
        max_context_len,
        dtype=logits_dtype,
        device=q.device,
    )

    if total_tokens == 0 or max_context_len == 0:
        return logits

    # block_size = 64 hardcoded for both specializations
    num_kv_blocks = (max_context_len + 63) >> 6
    grid = (total_tokens, num_kv_blocks)
    stride_bt = block_table.stride(0)

    # Explicit [total, max_ctx, H] fp32 scores round trip (see module docstring)
    scores = torch.empty(
        total_tokens,
        max_context_len,
        H,
        dtype=torch.float32,
        device=q.device,
    )

    _mqa_logits_scores_kernel[grid](
        q,
        kv_cache,
        block_table,
        context_lens,
        scores,
        next_n,
        max_context_len,
        stride_bt,
        H=H,
        num_warps=4,
        num_stages=1,
    )
    _mqa_logits_combine_kernel[grid](
        scores,
        weights,
        context_lens,
        logits,
        max_context_len,
        H=H,
        num_warps=4,
        num_stages=1,
    )

    if clean_logits:
        for b in range(B):
            for n in range(next_n):
                row_idx = b * next_n + n
                ctx_len = int(context_lens[b, n].item())
                if ctx_len < max_context_len:
                    logits[row_idx, ctx_len:] = float("-inf")

    return logits


def _install():
    """Wire the XPU implementation into the direct-import entrypoint.

    ``bf16_paged_mqa_logits`` is called via direct module import
    (``from flag_gems.fused import bf16_paged_mqa_logits``) in both
    tests/test_bf16_paged_mqa_logits.py and
    benchmark/test_bf16_paged_mqa_logits.py, so the SpecOpRegistrar
    namespace swap (which only patches the top-level ``flag_gems`` globals
    through ``_state.fused_module``) can not reach those bindings.  Replace
    the attributes on the already-imported modules (loaded during
    ``import flag_gems``).
    """
    from flag_gems.fused.bf16_paged_mqa_logits import (  # noqa: F401
        bf16_paged_mqa_logits as _generic_bf16_paged_mqa_logits,
    )

    fused_pkg = sys.modules.get("flag_gems.fused")
    if fused_pkg is not None:
        cur = getattr(fused_pkg, "bf16_paged_mqa_logits", None)
        if cur is _generic_bf16_paged_mqa_logits:
            fused_pkg.bf16_paged_mqa_logits = bf16_paged_mqa_logits

    sub = sys.modules.get("flag_gems.fused.bf16_paged_mqa_logits")
    if sub is not None:
        cur = getattr(sub, "bf16_paged_mqa_logits", None)
        if cur is _generic_bf16_paged_mqa_logits:
            sub.bf16_paged_mqa_logits = bf16_paged_mqa_logits

    top = sys.modules.get("flag_gems")
    if top is not None:
        cur = getattr(top, "bf16_paged_mqa_logits", None)
        if cur is _generic_bf16_paged_mqa_logits:
            top.bf16_paged_mqa_logits = bf16_paged_mqa_logits


_install()