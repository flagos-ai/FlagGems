"""Triton top_k_per_row_decode for DeepSeek V4 decode-phase token selection.

Replaces vLLM's top_k_per_row_decode CUDA kernel with a pure Triton
implementation using radix-select (4-iteration 8-bit histogram radix).

Background:
    In DeepSeek V4 decode, each step selects the top-K token indices from a
    single row of logits [1, vocab_size]. The vLLM CUDA kernel uses a
    radix-based approach; this Triton kernel matches that strategy with
    three dispatch tiers optimized for different vocab sizes.

Strategy:
    1. Single-block path (vocab_size <= 8192): All data fits in one thread
       block's registers. Four radix iterations with tl.histogram, no
       inter-block synchronization, no global memory scratch.
    2. Medium multi-block path (8192 < vocab_size <= 32768): All blocks
       participate in all 4 radix iterations. Double-buffered per-block
       histograms with 4 barriers (1 per iteration). Eliminates serial
       block-0 bottleneck seen in buffer-based approaches.
    3. Large multi-block path (vocab_size > 32768): First radix iteration
       runs across all blocks with per-block histograms + barrier. Remaining
       3 iterations run on block-0 only using a compacted buffer, avoiding
       barrier overhead for high block counts.

Performance (DeepSeek V4 config, H20 GPU):
    - vocab=129280, k=1024: 1.82x faster than vLLM CUDA
    - vocab=32768,  k=512:  0.78x vs vLLM CUDA
    - vocab=8192,   k=128:  0.50x vs vLLM CUDA
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SIGN_BIT = tl.constexpr(-(1 << 31))


@triton.jit
def _float_to_sortable(val):
    """Convert IEEE 754 float to order-preserving unsigned integer.

    XOR with sign-dependent mask so that sorted int order == sorted float order.
    """
    bits = val.to(tl.int32, bitcast=True)
    sign_ext = bits >> 31
    mask = sign_ext | tl.full(bits.shape, _SIGN_BIT, dtype=tl.int32)
    return bits ^ mask


@triton.jit
def _topk_single_block(
    logits_ptr,
    seq_len_ptr,
    indices_ptr,
    stride1,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Single-block radix select: all 4 iterations in-register, no barriers."""
    offs = tl.arange(0, BLOCK)
    seq_len = tl.load(seq_len_ptr)
    valid = (offs < N) & (offs < seq_len)

    vals = tl.load(logits_ptr + offs * stride1, mask=valid, other=float("-inf"))
    sortable = _float_to_sortable(vals)

    bins = tl.arange(0, 256)

    # Radix iteration 0: byte 3 (MSB)
    bucket_0 = (sortable >> 24) & 0xFF
    counts_0 = tl.histogram(bucket_0, 256, mask=valid)
    total_0 = tl.sum(counts_0)
    ps_0 = tl.cumsum(counts_0, axis=0)
    ss_0 = total_0 - ps_0 + counts_0
    pivot_0 = tl.max(tl.where(ss_0 >= TOP_K, bins, -1))
    ca_0 = tl.sum(tl.where(bins > pivot_0, counts_0, 0))
    remaining_k = TOP_K - ca_0
    match_0 = (bucket_0 == pivot_0) & valid

    # Radix iteration 1: byte 2
    bucket_1 = (sortable >> 16) & 0xFF
    counts_1 = tl.histogram(bucket_1, 256, mask=match_0)
    total_1 = tl.sum(counts_1)
    ps_1 = tl.cumsum(counts_1, axis=0)
    ss_1 = total_1 - ps_1 + counts_1
    pivot_1 = tl.max(tl.where(ss_1 >= remaining_k, bins, -1))
    ca_1 = tl.sum(tl.where(bins > pivot_1, counts_1, 0))
    remaining_k = remaining_k - ca_1
    match_1 = match_0 & (bucket_1 == pivot_1)

    # Radix iteration 2: byte 1
    bucket_2 = (sortable >> 8) & 0xFF
    counts_2 = tl.histogram(bucket_2, 256, mask=match_1)
    total_2 = tl.sum(counts_2)
    ps_2 = tl.cumsum(counts_2, axis=0)
    ss_2 = total_2 - ps_2 + counts_2
    pivot_2 = tl.max(tl.where(ss_2 >= remaining_k, bins, -1))
    ca_2 = tl.sum(tl.where(bins > pivot_2, counts_2, 0))
    remaining_k = remaining_k - ca_2
    match_2 = match_1 & (bucket_2 == pivot_2)

    # Radix iteration 3: byte 0 (LSB)
    bucket_3 = sortable & 0xFF
    counts_3 = tl.histogram(bucket_3, 256, mask=match_2)
    total_3 = tl.sum(counts_3)
    ps_3 = tl.cumsum(counts_3, axis=0)
    ss_3 = total_3 - ps_3 + counts_3
    pivot_3 = tl.max(tl.where(ss_3 >= remaining_k, bins, -1))
    ca_3 = tl.sum(tl.where(bins > pivot_3, counts_3, 0))
    remaining_k = remaining_k - ca_3

    # Selection: write indices for elements above threshold, then equal
    threshold = (pivot_0 << 24) | (pivot_1 << 16) | (pivot_2 << 8) | pivot_3
    above_total = TOP_K - remaining_k

    s_shifted = sortable ^ tl.full(sortable.shape, _SIGN_BIT, dtype=tl.int32)
    t_shifted = threshold ^ _SIGN_BIT

    above = (s_shifted > t_shifted) & valid
    equal = (sortable == threshold) & valid

    pa = tl.cumsum(above.to(tl.int32), axis=0)
    tl.store(
        indices_ptr + pa - 1,
        offs.to(tl.int32),
        mask=above & (pa - 1 >= 0) & (pa - 1 < TOP_K),
    )

    pe = tl.cumsum(equal.to(tl.int32), axis=0)
    wpe = above_total + pe - 1
    tl.store(
        indices_ptr + wpe,
        offs.to(tl.int32),
        mask=equal & ((pe - 1) < remaining_k) & (wpe >= 0) & (wpe < TOP_K),
    )


@triton.jit
def _topk_medium_block(
    logits_ptr,
    seq_len_ptr,
    pb_hist_a_ptr,
    pb_hist_b_ptr,
    sync_ptr,
    counter_ptr,
    indices_ptr,
    stride1,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Multi-block radix select for medium vocab (8K-32K).

    All blocks participate in all 4 radix iterations using double-buffered
    per-block histograms. 4 barriers total (1 per iteration).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    seq_len = tl.load(seq_len_ptr)
    valid = (offs < N) & (offs < seq_len)

    vals = tl.load(logits_ptr + offs * stride1, mask=valid, other=float("-inf"))
    sortable = _float_to_sortable(vals)

    bins = tl.arange(0, 256)
    ha_base = pb_hist_a_ptr + pid * 256
    hb_base = pb_hist_b_ptr + pid * 256

    # Iteration 0: byte 3 (MSB), write to buf_A
    bucket_0 = (sortable >> 24) & 0xFF
    local_hist_0 = tl.histogram(bucket_0, 256, mask=valid)
    tl.store(ha_base + bins, local_hist_0)

    tl.debug_barrier()
    tl.atomic_add(sync_ptr, 1)
    while tl.atomic_add(sync_ptr, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_hist_a_ptr + i * 256 + bins)

    total_0 = tl.sum(counts)
    ps_0 = tl.cumsum(counts, axis=0)
    ss_0 = total_0 - ps_0 + counts
    pivot_0 = tl.max(tl.where(ss_0 >= TOP_K, bins, -1))
    ca_0 = tl.sum(tl.where(bins > pivot_0, counts, 0))
    remaining_k = TOP_K - ca_0
    match = (bucket_0 == pivot_0) & valid

    # Iteration 1: byte 2, write to buf_B
    bucket_1 = (sortable >> 16) & 0xFF
    local_hist_1 = tl.histogram(bucket_1, 256, mask=match)
    tl.store(hb_base + bins, local_hist_1)

    tl.debug_barrier()
    tl.atomic_add(sync_ptr + 1, 1)
    while tl.atomic_add(sync_ptr + 1, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_hist_b_ptr + i * 256 + bins)

    total_1 = tl.sum(counts)
    ps_1 = tl.cumsum(counts, axis=0)
    ss_1 = total_1 - ps_1 + counts
    pivot_1 = tl.max(tl.where(ss_1 >= remaining_k, bins, -1))
    ca_1 = tl.sum(tl.where(bins > pivot_1, counts, 0))
    remaining_k = remaining_k - ca_1
    match = match & (bucket_1 == pivot_1)

    # Iteration 2: byte 1, write to buf_A
    bucket_2 = (sortable >> 8) & 0xFF
    local_hist_2 = tl.histogram(bucket_2, 256, mask=match)
    tl.store(ha_base + bins, local_hist_2)

    tl.debug_barrier()
    tl.atomic_add(sync_ptr + 2, 1)
    while tl.atomic_add(sync_ptr + 2, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_hist_a_ptr + i * 256 + bins)

    total_2 = tl.sum(counts)
    ps_2 = tl.cumsum(counts, axis=0)
    ss_2 = total_2 - ps_2 + counts
    pivot_2 = tl.max(tl.where(ss_2 >= remaining_k, bins, -1))
    ca_2 = tl.sum(tl.where(bins > pivot_2, counts, 0))
    remaining_k = remaining_k - ca_2
    match = match & (bucket_2 == pivot_2)

    # Iteration 3: byte 0 (LSB), write to buf_B
    bucket_3 = sortable & 0xFF
    local_hist_3 = tl.histogram(bucket_3, 256, mask=match)
    tl.store(hb_base + bins, local_hist_3)

    tl.debug_barrier()
    tl.atomic_add(sync_ptr + 3, 1)
    while tl.atomic_add(sync_ptr + 3, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_hist_b_ptr + i * 256 + bins)

    total_3 = tl.sum(counts)
    ps_3 = tl.cumsum(counts, axis=0)
    ss_3 = total_3 - ps_3 + counts
    pivot_3 = tl.max(tl.where(ss_3 >= remaining_k, bins, -1))
    ca_3 = tl.sum(tl.where(bins > pivot_3, counts, 0))
    remaining_k = remaining_k - ca_3

    # Selection phase
    threshold = (pivot_0 << 24) | (pivot_1 << 16) | (pivot_2 << 8) | pivot_3
    above_total = TOP_K - remaining_k

    s_shifted = sortable ^ tl.full(sortable.shape, _SIGN_BIT, dtype=tl.int32)
    t_shifted = threshold ^ _SIGN_BIT

    above = (s_shifted > t_shifted) & valid
    equal = (sortable == threshold) & valid

    n_above = tl.sum(above.to(tl.int32))
    if n_above > 0:
        pa = tl.cumsum(above.to(tl.int32), axis=0)
        base_a = tl.atomic_add(counter_ptr, n_above)
        wp = base_a + pa - 1
        tl.store(
            indices_ptr + wp,
            offs.to(tl.int32),
            mask=above & (wp >= 0) & (wp < TOP_K),
        )

    n_equal = tl.sum(equal.to(tl.int32))
    if n_equal > 0:
        pe = tl.cumsum(equal.to(tl.int32), axis=0)
        base_e = tl.atomic_add(counter_ptr + 1, n_equal)
        wpe = above_total + base_e + pe - 1
        tl.store(
            indices_ptr + wpe,
            offs.to(tl.int32),
            mask=equal & ((base_e + pe - 1) < remaining_k) & (wpe >= 0) & (wpe < TOP_K),
        )

    # Zero shared state for next call
    if pid == 0:
        tl.store(sync_ptr + tl.arange(0, 4), tl.zeros([4], dtype=tl.int32))
        tl.store(counter_ptr, 0)
        tl.store(counter_ptr + 1, 0)


@triton.jit
def _topk_multi_block(
    logits_ptr,
    seq_len_ptr,
    pb_hist_ptr,
    sync_ptr,
    buf_val_ptr,
    buf_idx_ptr,
    counter_ptr,
    indices_ptr,
    stride1,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    BUF_SIZE: tl.constexpr,
):
    """Multi-block radix select for large vocab (>32K).

    Iteration 0: all blocks compute byte-3 histograms + barrier + reduce.
    Iterations 1-3: block-0 only, operating on a compacted buffer of
    elements matching the byte-3 pivot.  Avoids barrier overhead for
    high block counts (e.g. 32 blocks for vocab=129280).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    seq_len = tl.load(seq_len_ptr)
    valid = (offs < N) & (offs < seq_len)

    vals = tl.load(logits_ptr + offs * stride1, mask=valid, other=float("-inf"))
    sortable = _float_to_sortable(vals)

    # Iteration 0: all blocks compute byte-3 histogram
    bucket = (sortable >> 24) & 0xFF
    local_hist = tl.histogram(bucket, 256, mask=valid)

    bins = tl.arange(0, 256)
    h_base = pb_hist_ptr + pid * 256
    tl.store(h_base + bins, local_hist)

    tl.debug_barrier()
    tl.atomic_add(sync_ptr, 1)
    while tl.atomic_add(sync_ptr, 0) < NUM_BLOCKS:
        pass

    counts = tl.zeros([256], dtype=tl.int32)
    for i in tl.static_range(NUM_BLOCKS):
        counts += tl.load(pb_hist_ptr + i * 256 + bins)

    total = tl.sum(counts)
    ps = tl.cumsum(counts, axis=0)
    ss = total - ps + counts
    pivot_0 = tl.max(tl.where(ss >= TOP_K, bins, -1))
    count_above_0 = tl.sum(tl.where(bins > pivot_0, counts, 0))
    remaining_k = TOP_K - count_above_0

    above = (bucket > pivot_0) & valid
    match = (bucket == pivot_0) & valid

    # Write above-threshold indices directly to output
    n_above = tl.sum(above.to(tl.int32))
    if n_above > 0:
        pa = tl.cumsum(above.to(tl.int32), axis=0)
        base_a = tl.atomic_add(counter_ptr, n_above)
        wp = base_a + pa - 1
        tl.store(
            indices_ptr + wp,
            offs.to(tl.int32),
            mask=above & (wp >= 0) & (wp < TOP_K),
        )

    # Compact matching elements into buffer for block-0
    n_match = tl.sum(match.to(tl.int32))
    if n_match > 0:
        pm = tl.cumsum(match.to(tl.int32), axis=0)
        base_m = tl.atomic_add(counter_ptr + 1, n_match)
        bp = base_m + pm - 1
        tl.store(
            buf_val_ptr + bp,
            sortable,
            mask=match & (bp >= 0) & (bp < BUF_SIZE),
        )
        tl.store(
            buf_idx_ptr + bp,
            offs.to(tl.int32),
            mask=match & (bp >= 0) & (bp < BUF_SIZE),
        )

    # Iterations 1-3: block-0 processes compacted buffer
    tl.debug_barrier()
    tl.atomic_add(sync_ptr + 1, 1)
    if pid == 0:
        while tl.atomic_add(sync_ptr + 1, 0) < NUM_BLOCKS:
            pass

        buf_count = tl.atomic_add(counter_ptr + 1, 0)

        b_offs = tl.arange(0, BUF_SIZE)
        b_valid = b_offs < buf_count
        b_vals = tl.load(buf_val_ptr + b_offs, mask=b_valid, other=0)
        b_idxs = tl.load(buf_idx_ptr + b_offs, mask=b_valid, other=0)

        # Iteration 1: byte 2
        b_byte_1 = (b_vals >> 16) & 0xFF
        counts_1 = tl.histogram(b_byte_1, 256, mask=b_valid)
        total_1 = tl.sum(counts_1)
        ps_1 = tl.cumsum(counts_1, axis=0)
        ss_1 = total_1 - ps_1 + counts_1
        pivot_1 = tl.max(tl.where(ss_1 >= remaining_k, bins, -1))
        ca_1 = tl.sum(tl.where(bins > pivot_1, counts_1, 0))
        remaining_k = remaining_k - ca_1

        # Iteration 2: byte 1
        prefix_hi16 = (pivot_0 << 8) | pivot_1
        upper16 = (b_vals >> 16) & 0xFFFF
        b_match_2 = (upper16 == prefix_hi16) & b_valid
        b_bucket_2 = (b_vals >> 8) & 0xFF
        counts_2 = tl.histogram(b_bucket_2, 256, mask=b_match_2)
        total_2 = tl.sum(counts_2)
        ps_2 = tl.cumsum(counts_2, axis=0)
        ss_2 = total_2 - ps_2 + counts_2
        pivot_2 = tl.max(tl.where(ss_2 >= remaining_k, bins, -1))
        ca_2 = tl.sum(tl.where(bins > pivot_2, counts_2, 0))
        remaining_k = remaining_k - ca_2

        # Iteration 3: byte 0 (LSB)
        prefix_hi24 = (prefix_hi16 << 8) | pivot_2
        upper24 = (b_vals >> 8) & 0xFFFFFF
        b_match_3 = (upper24 == prefix_hi24) & b_valid
        b_bucket_3 = b_vals & 0xFF
        counts_3 = tl.histogram(b_bucket_3, 256, mask=b_match_3)
        total_3 = tl.sum(counts_3)
        ps_3 = tl.cumsum(counts_3, axis=0)
        ss_3 = total_3 - ps_3 + counts_3
        pivot_3 = tl.max(tl.where(ss_3 >= remaining_k, bins, -1))
        ca_3 = tl.sum(tl.where(bins > pivot_3, counts_3, 0))
        remaining_k = remaining_k - ca_3

        # Final selection from buffer
        threshold = (prefix_hi24 << 8) | pivot_3
        above_total = TOP_K - remaining_k

        s_sh = b_vals ^ tl.full(b_vals.shape, _SIGN_BIT, dtype=tl.int32)
        t_sh = threshold ^ _SIGN_BIT

        above_buf = (s_sh > t_sh) & b_valid
        equal_buf = (b_vals == threshold) & b_valid

        pa_b = tl.cumsum(above_buf.to(tl.int32), axis=0)
        wp_b = count_above_0 + pa_b - 1
        tl.store(
            indices_ptr + wp_b,
            b_idxs,
            mask=above_buf & (wp_b >= 0) & (wp_b < TOP_K),
        )

        pe_b = tl.cumsum(equal_buf.to(tl.int32), axis=0)
        wpe_b = above_total + pe_b - 1
        tl.store(
            indices_ptr + wpe_b,
            b_idxs,
            mask=equal_buf
            & ((pe_b - 1) < remaining_k)
            & (wpe_b >= 0)
            & (wpe_b < TOP_K),
        )

        tl.store(sync_ptr, 0)
        tl.store(sync_ptr + 1, 0)
        tl.store(counter_ptr, 0)
        tl.store(counter_ptr + 1, 0)


# Persistent scratch buffers, keyed by (device_index, dispatch_tier).
# Allocated once per device and reused across calls to avoid cudaMalloc overhead.
_cache = {}

# Dispatch thresholds for the three kernel tiers
_SINGLE_BLOCK_LIMIT = 8192
_MEDIUM_BLOCK_LIMIT = 32768
_MEDIUM_BLOCK_SIZE = 4096
_LARGE_BLOCK_SIZE = 4096
_LARGE_BUF_SIZE = 4096


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for decode phase of DeepSeek V4.

    Selects top_k indices from a single row of logits using radix-based
    selection. Only valid elements within [0, seq_lens[0]) are considered.

    Args:
        logits: [1, vocab_size] float32 tensor.
        next_n: number of next tokens (unused, kept for API compatibility).
        seq_lens: [1] int32 — valid range [0, seq_lens[0]).
        indices: [1, top_k] int32 — output buffer, filled with selected indices.
        num_rows: must be 1 (decode processes one row at a time).
        stride0: logits.stride(0).
        stride1: logits.stride(1).
        top_k: number of top elements to select.
    """
    logger.debug("GEMS TOP_K_PER_ROW_DECODE")

    assert num_rows == 1, "Only num_rows=1 supported in decode path"

    vocab_size = logits.shape[1]
    device = logits.device
    ind = indices.view(-1)

    if vocab_size <= _SINGLE_BLOCK_LIMIT // 2:
        # Small vocab: single block with BLOCK=4096
        _topk_single_block[(1,)](
            logits,
            seq_lens,
            ind,
            stride1,
            N=vocab_size,
            BLOCK=_SINGLE_BLOCK_LIMIT // 2,
            TOP_K=top_k,
            num_warps=8,
        )
    elif vocab_size <= _SINGLE_BLOCK_LIMIT:
        # Medium-small vocab: single block with BLOCK=8192
        _topk_single_block[(1,)](
            logits,
            seq_lens,
            ind,
            stride1,
            N=vocab_size,
            BLOCK=_SINGLE_BLOCK_LIMIT,
            TOP_K=top_k,
            num_warps=16,
        )
    elif vocab_size <= _MEDIUM_BLOCK_LIMIT:
        # Medium vocab: double-buffered all-blocks radix
        n_blocks = (vocab_size + _MEDIUM_BLOCK_SIZE - 1) // _MEDIUM_BLOCK_SIZE
        dev_idx = device.index if device.index is not None else 0
        key = (dev_idx, "med")
        if key not in _cache:
            max_nb = (
                _MEDIUM_BLOCK_LIMIT + _MEDIUM_BLOCK_SIZE - 1
            ) // _MEDIUM_BLOCK_SIZE
            pb_size = max_nb * 256
            pb_hist_a = torch.zeros(pb_size, dtype=torch.int32, device=device)
            pb_hist_b = torch.zeros(pb_size, dtype=torch.int32, device=device)
            sync = torch.zeros(4, dtype=torch.int32, device=device)
            counter = torch.zeros(2, dtype=torch.int32, device=device)
            _cache[key] = (pb_hist_a, pb_hist_b, sync, counter)
        pb_hist_a, pb_hist_b, sync, counter = _cache[key]

        _topk_medium_block[(n_blocks,)](
            logits,
            seq_lens,
            pb_hist_a,
            pb_hist_b,
            sync,
            counter,
            ind,
            stride1,
            N=vocab_size,
            NUM_BLOCKS=n_blocks,
            BLOCK=_MEDIUM_BLOCK_SIZE,
            TOP_K=top_k,
            num_warps=8,
        )
    else:
        # Large vocab: buffer-based multi-block radix
        n_blocks = (vocab_size + _LARGE_BLOCK_SIZE - 1) // _LARGE_BLOCK_SIZE
        dev_idx = device.index if device.index is not None else 0
        key = (dev_idx, "large")
        if key not in _cache:
            max_nb = 64
            pb_size = max_nb * 256
            total_sz = pb_size + 4
            scratch = torch.zeros(total_sz, dtype=torch.int32, device=device)
            buf = torch.empty(_LARGE_BUF_SIZE * 2, dtype=torch.int32, device=device)
            _cache[key] = (
                scratch[:pb_size],
                scratch[pb_size : pb_size + 2],
                buf[:_LARGE_BUF_SIZE],
                buf[_LARGE_BUF_SIZE:],
                scratch[pb_size + 2 : pb_size + 4],
            )
        pb_hist, sync, buf_val, buf_idx, counter = _cache[key]

        _topk_multi_block[(n_blocks,)](
            logits,
            seq_lens,
            pb_hist,
            sync,
            buf_val,
            buf_idx,
            counter,
            ind,
            stride1,
            N=vocab_size,
            NUM_BLOCKS=n_blocks,
            BLOCK=_LARGE_BLOCK_SIZE,
            TOP_K=top_k,
            BUF_SIZE=_LARGE_BUF_SIZE,
            num_warps=8,
        )
