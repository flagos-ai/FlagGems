"""Persistent top-k kernel using workspace-cached ternary search.

Selects the top-k indices from each row of a logits tensor, designed for
DeepSeek-V4 sparse attention inference. Uses a two-phase approach:
  1. Cache row data to workspace for L2 locality, find min/max.
  2. Ternary search (4-way narrowing, 14 passes) to find threshold.
  3. Scatter indices above threshold.
  4. Fill remaining slots with elements equal to threshold.

Falls back to direct HBM reads when workspace is too small.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _topk_kernel_cached(
    input_ptr,
    output_ptr,
    lengths_ptr,
    cache_ptr,
    num_rows,
    stride,
    top_k: tl.constexpr,
    BLK: tl.constexpr,
    NITER: tl.constexpr,
):
    """Workspace-cached ternary search.

    Stores row data to workspace during min/max phase, then all subsequent
    passes read from the workspace buffer for L2 locality.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    seq_len = tl.load(lengths_ptr + row_idx).to(tl.int32)
    seq_len = tl.where(seq_len > stride, stride, seq_len)
    input_row = input_ptr + row_idx * stride
    cache_row = cache_ptr + row_idx * stride
    output_row = output_ptr + row_idx * top_k

    # Trivial case: seq_len <= k, return all valid indices
    if seq_len <= top_k:
        for start in range(0, top_k, BLK):
            offs = start + tl.arange(0, BLK)
            store_mask = offs < top_k
            vals = tl.where(
                offs < seq_len,
                offs.to(tl.int32),
                tl.full([BLK], -1, dtype=tl.int32),
            )
            tl.store(output_row + offs, vals, mask=store_mask)
        return

    pos_inf = tl.full([BLK], float("inf"), dtype=tl.float32)
    neg_inf = tl.full([BLK], float("-inf"), dtype=tl.float32)

    # Phase 1: find min/max, cache row in workspace for L2 locality
    row_min = tl.full([], float("inf"), dtype=tl.float32)
    row_max = tl.full([], float("-inf"), dtype=tl.float32)

    for start in range(0, seq_len, BLK):
        offs = start + tl.arange(0, BLK)
        mask = offs < seq_len
        vals = tl.load(input_row + offs, mask=mask, other=0.0)
        tl.store(cache_row + offs, vals, mask=mask)
        row_min = tl.minimum(row_min, tl.min(tl.where(mask, vals, pos_inf), axis=0))
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, vals, neg_inf), axis=0))

    K = top_k

    # Phase 2: ternary search from workspace cache
    # 4-way narrowing per iteration: 14 passes = 28 effective bisections
    lo = row_min
    hi = row_max

    for _ in range(NITER):
        span = hi - lo
        mid1 = lo + span * 0.25
        mid2 = lo + span * 0.5
        mid3 = lo + span * 0.75

        count1 = tl.zeros([], dtype=tl.int32)
        count2 = tl.zeros([], dtype=tl.int32)
        count3 = tl.zeros([], dtype=tl.int32)

        for start in range(0, seq_len, BLK):
            offs = start + tl.arange(0, BLK)
            mask = offs < seq_len
            vals = tl.load(cache_row + offs, mask=mask, other=float("-inf"))

            above1 = mask & (vals >= mid1)
            above2 = mask & (vals >= mid2)
            above3 = mask & (vals >= mid3)

            count1 += tl.sum(above1.to(tl.int32))
            count2 += tl.sum(above2.to(tl.int32))
            count3 += tl.sum(above3.to(tl.int32))

        lo = tl.where(
            count3 >= K,
            mid3,
            tl.where(count2 >= K, mid2, tl.where(count1 >= K, mid1, lo)),
        )
        hi = tl.where(
            count3 >= K,
            hi,
            tl.where(count2 >= K, mid3, tl.where(count1 >= K, mid2, mid1)),
        )

    threshold = lo

    # Phase 3: scatter indices where value > threshold (from cache)
    write_pos = tl.zeros([], dtype=tl.int32)
    for start in range(0, seq_len, BLK):
        offs = start + tl.arange(0, BLK)
        mask = offs < seq_len
        vals = tl.load(cache_row + offs, mask=mask, other=float("-inf"))
        above_mask = mask & (vals > threshold)
        above_int = above_mask.to(tl.int32)
        local_cumsum = tl.cumsum(above_int, axis=0)
        write_offs = local_cumsum - 1 + write_pos
        store_mask = above_mask & (write_offs < top_k)
        tl.store(output_row + write_offs, offs.to(tl.int32), mask=store_mask)
        write_pos += tl.sum(above_int)

    # Phase 4: fill remaining slots with elements == threshold
    equal_needed = top_k - write_pos
    equal_written = tl.zeros([], dtype=tl.int32)
    for start in range(0, seq_len, BLK):
        offs = start + tl.arange(0, BLK)
        mask = offs < seq_len
        vals = tl.load(cache_row + offs, mask=mask, other=float("-inf"))
        still_need = equal_needed - equal_written
        eq_base = mask & (vals == threshold)
        local_cumsum_eq = tl.cumsum(eq_base.to(tl.int32), axis=0)
        eq_mask = eq_base & (local_cumsum_eq <= still_need)
        eq_int = eq_mask.to(tl.int32)
        local_cumsum = tl.cumsum(eq_int, axis=0)
        write_offs = local_cumsum - 1 + write_pos + equal_written
        store_mask = eq_mask & (write_offs < top_k) & (write_offs >= 0)
        tl.store(output_row + write_offs, offs.to(tl.int32), mask=store_mask)
        equal_written += tl.sum(eq_int)


@triton.jit
def _topk_kernel_direct(
    input_ptr,
    output_ptr,
    lengths_ptr,
    num_rows,
    stride,
    top_k: tl.constexpr,
    BLK: tl.constexpr,
    NITER: tl.constexpr,
):
    """Direct ternary search fallback when workspace is too small.

    Same algorithm as cached variant but reads from HBM each iteration.
    """
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    seq_len = tl.load(lengths_ptr + row_idx).to(tl.int32)
    seq_len = tl.where(seq_len > stride, stride, seq_len)
    input_row = input_ptr + row_idx * stride
    output_row = output_ptr + row_idx * top_k

    # Trivial case: seq_len <= k, return all valid indices
    if seq_len <= top_k:
        for start in range(0, top_k, BLK):
            offs = start + tl.arange(0, BLK)
            store_mask = offs < top_k
            vals = tl.where(
                offs < seq_len,
                offs.to(tl.int32),
                tl.full([BLK], -1, dtype=tl.int32),
            )
            tl.store(output_row + offs, vals, mask=store_mask)
        return

    pos_inf = tl.full([BLK], float("inf"), dtype=tl.float32)
    neg_inf = tl.full([BLK], float("-inf"), dtype=tl.float32)

    # Phase 1: find min/max of the row
    row_min = tl.full([], float("inf"), dtype=tl.float32)
    row_max = tl.full([], float("-inf"), dtype=tl.float32)

    for start in range(0, seq_len, BLK):
        offs = start + tl.arange(0, BLK)
        mask = offs < seq_len
        vals = tl.load(input_row + offs, mask=mask, other=0.0)
        row_min = tl.minimum(row_min, tl.min(tl.where(mask, vals, pos_inf), axis=0))
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, vals, neg_inf), axis=0))

    K = top_k

    # Phase 2: ternary search from HBM
    lo = row_min
    hi = row_max

    for _ in range(NITER):
        span = hi - lo
        mid1 = lo + span * 0.25
        mid2 = lo + span * 0.5
        mid3 = lo + span * 0.75

        count1 = tl.zeros([], dtype=tl.int32)
        count2 = tl.zeros([], dtype=tl.int32)
        count3 = tl.zeros([], dtype=tl.int32)

        for start in range(0, seq_len, BLK):
            offs = start + tl.arange(0, BLK)
            mask = offs < seq_len
            vals = tl.load(input_row + offs, mask=mask, other=float("-inf"))

            above1 = mask & (vals >= mid1)
            above2 = mask & (vals >= mid2)
            above3 = mask & (vals >= mid3)

            count1 += tl.sum(above1.to(tl.int32))
            count2 += tl.sum(above2.to(tl.int32))
            count3 += tl.sum(above3.to(tl.int32))

        lo = tl.where(
            count3 >= K,
            mid3,
            tl.where(count2 >= K, mid2, tl.where(count1 >= K, mid1, lo)),
        )
        hi = tl.where(
            count3 >= K,
            hi,
            tl.where(count2 >= K, mid3, tl.where(count1 >= K, mid2, mid1)),
        )

    threshold = lo

    # Phase 3: scatter indices where value > threshold
    write_pos = tl.zeros([], dtype=tl.int32)
    for start in range(0, seq_len, BLK):
        offs = start + tl.arange(0, BLK)
        mask = offs < seq_len
        vals = tl.load(input_row + offs, mask=mask, other=float("-inf"))
        above_mask = mask & (vals > threshold)
        above_int = above_mask.to(tl.int32)
        local_cumsum = tl.cumsum(above_int, axis=0)
        write_offs = local_cumsum - 1 + write_pos
        store_mask = above_mask & (write_offs < top_k)
        tl.store(output_row + write_offs, offs.to(tl.int32), mask=store_mask)
        write_pos += tl.sum(above_int)

    # Phase 4: fill remaining slots with elements == threshold
    equal_needed = top_k - write_pos
    equal_written = tl.zeros([], dtype=tl.int32)
    for start in range(0, seq_len, BLK):
        offs = start + tl.arange(0, BLK)
        mask = offs < seq_len
        vals = tl.load(input_row + offs, mask=mask, other=float("-inf"))
        still_need = equal_needed - equal_written
        eq_base = mask & (vals == threshold)
        local_cumsum_eq = tl.cumsum(eq_base.to(tl.int32), axis=0)
        eq_mask = eq_base & (local_cumsum_eq <= still_need)
        eq_int = eq_mask.to(tl.int32)
        local_cumsum = tl.cumsum(eq_int, axis=0)
        write_offs = local_cumsum - 1 + write_pos + equal_written
        store_mask = eq_mask & (write_offs < top_k) & (write_offs >= 0)
        tl.store(output_row + write_offs, offs.to(tl.int32), mask=store_mask)
        equal_written += tl.sum(eq_int)


# Default kernel parameters (passed as tl.constexpr)
_DEFAULT_BLOCK_SIZE = 4096
_DEFAULT_NUM_SEARCH_ITERS = 14  # 14 ternary passes = 28 effective bisections

# Workspace size in bytes (must be >= 1 MiB for the CUDA baseline contract)
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    """Select top-k indices per row using persistent ternary search.

    Args:
        logits: [num_rows, stride] float32 input logits.
        seq_lens: [num_rows] or [batch, next_n] int32 sequence lengths.
        topk_indices: [num_rows, k] int32 output buffer (mutated in-place).
        topk_workspace: [>=1048576] uint8 workspace buffer.
        k: Number of top elements to select per row.
        max_seq_len: Maximum sequence length (used as stride).
    """
    logger.debug("GEMS PERSISTENT_TOPK")

    num_rows = topk_indices.shape[0]
    stride = max_seq_len

    # Handle multi-dimensional seq_lens (batch, next_n) format
    if seq_lens.dim() == 2:
        batch_size = seq_lens.shape[0]
        next_n = seq_lens.shape[1]
        if next_n == 1:
            tokens_per_batch = num_rows // batch_size
            seq_lens_flat = (
                seq_lens.squeeze(1)
                .unsqueeze(1)
                .expand(batch_size, tokens_per_batch)
                .reshape(-1)
                .contiguous()
            )
        else:
            seq_lens_flat = seq_lens.reshape(-1).contiguous()
    else:
        seq_lens_flat = seq_lens

    # Expand seq_lens if fewer entries than rows (broadcast per batch)
    if seq_lens_flat.shape[0] < num_rows:
        batch_size = seq_lens_flat.shape[0]
        tokens_per_batch = num_rows // batch_size
        seq_lens_flat = (
            seq_lens_flat.unsqueeze(1)
            .expand(batch_size, tokens_per_batch)
            .reshape(-1)
            .contiguous()
        )

    ws_float = topk_workspace.view(torch.float32)
    ws_needed = num_rows * stride
    use_cache = ws_float.numel() >= ws_needed

    grid = (num_rows,)

    if use_cache:
        _topk_kernel_cached[grid](
            logits,
            topk_indices,
            seq_lens_flat,
            ws_float,
            num_rows,
            stride,
            k,
            BLK=_DEFAULT_BLOCK_SIZE,
            NITER=_DEFAULT_NUM_SEARCH_ITERS,
        )
    else:
        _topk_kernel_direct[grid](
            logits,
            topk_indices,
            seq_lens_flat,
            num_rows,
            stride,
            k,
            BLK=_DEFAULT_BLOCK_SIZE,
            NITER=_DEFAULT_NUM_SEARCH_ITERS,
        )
