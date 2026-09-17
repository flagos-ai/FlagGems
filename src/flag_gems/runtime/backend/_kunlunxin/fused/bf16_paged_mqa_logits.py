import logging
import sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

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

    q_base = pid_row * (H * 128)
    q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
    q_block = tl.load(q_ptr + q_offs)
    q_t = tl.trans(q_block)

    b_idx = pid_row // next_n
    phys_blk = tl.load(block_table_ptr + b_idx * stride_bt + pid_blk)
    k_offs = phys_blk * (64 * 128) + pos_range[:, None] * 128 + d_range[None, :]
    k_block = tl.load(kv_cache_ptr + k_offs)

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

    num_kv_blocks = (max_context_len + 63) >> 6
    grid = (total_tokens, num_kv_blocks)
    stride_bt = block_table.stride(0)

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
    from flag_gems.fused.bf16_paged_mqa_logits import (
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
