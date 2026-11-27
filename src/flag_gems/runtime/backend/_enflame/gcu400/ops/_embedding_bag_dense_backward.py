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

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization-iteration driver script; only changes launch geometry and frequency counting,
# not semantics.
BLOCK_D_CAP = 512
NUM_WARPS = 4
USE_1D_GRID = 1
USE_FREQ_KERNEL = 1
FREQ_BLOCK = 128
USE_PSW_PLACEHOLDER = 1
# Threshold for enabling the grouped-aggregation path (removes per-element fp atomics): number of elements =
# mapped_samples * embedding_dim
GROUP_MIN_ELEMS = 16384
SCAN_BLOCK = 1024
SCAN_WARPS = 1
SCATTER_BLOCK = 128
REDUCE_WARPS = 4
# Block width of the built-in zero-fill kernel (replacing the fixed overhead of aten zeros)
ZERO_FILL_BLOCK = 4096
# Reduction kernel grid cap: with one program per row, the per-program fixed cost (launch/scalar chain) grows
# linearly with num_weights; switching to a FIXED_GRID strided row loop amortizes it; 0 means keep the per-row grid.
REDUCE_WORKERS = 48
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def _embedding_bag_zero_fill_kernel(out_ptr, num_elems, BLOCK: tl.constexpr):
    """In-place zeroing: aten zeros/zero_ measured ~9us per call on this machine, and following it
    immediately with a Triton kernel on the same stream adds another ~11us (likely a fallback-path
    sync), so the built-in narrow-grid zeroing kernel takes the same combination from 29.6us to 11.0us.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + offsets,
        tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty),
        mask=offsets < num_elems,
    )


@libentry()
@triton.jit
def _embedding_bag_freq_kernel(
    indices_ptr,
    freq_ptr,
    num_samples,
    num_weights,
    padding_idx,
    BLOCK: tl.constexpr,
):
    # Global occurrence count of each index; the main kernel needs one lookup for scale_grad_by_freq.
    sample_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index_values = tl.load(
        indices_ptr + sample_offsets, sample_offsets < num_samples, other=-1
    ).to(tl.int32)
    valid = (
        (sample_offsets < num_samples)
        & (index_values >= 0)
        & (index_values < num_weights)
        & (index_values != padding_idx)
    )
    tl.atomic_add(freq_ptr + index_values, 1, mask=valid)


@libentry()
@triton.jit
def _embedding_bag_dense_backward_kernel(
    grad_ptr,
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    grad_weight_ptr,
    freq_ptr,
    num_weights,
    padding_idx,
    mode: tl.constexpr,
    has_per_sample_weights: tl.constexpr,
    per_sample_weights_ptr,
    scale_grad_by_freq: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
    USE_1D_GRID: tl.constexpr,
    USE_FREQ_KERNEL: tl.constexpr,
):
    """Accumulate gradients into a dense weight-gradient table by (sample, dim tile).

    mode=0 (sum) adds the upstream gradient directly; mode=1 (mean) divides by bag_size first.
    padding_idx / out-of-range indices are skipped by masks; frequency scaling uses the precomputed freq table.
    """
    pid = tl.program_id(0)
    if USE_1D_GRID:
        pid_s = pid // NUM_D_TILES
        pid_d = pid % NUM_D_TILES
    else:
        pid_s = pid
        pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    idx = tl.load(indices_ptr + pid_s).to(tl.int32)
    bag_idx = tl.load(offset2bag_ptr + pid_s).to(tl.int32)
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    bag_size = tl.load(bag_size_ptr + bag_idx).to(tl.float32)
    bag_size = tl.where(bag_size == 0.0, 1.0, bag_size)

    go_ptrs = grad_ptr + bag_idx * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    if mode == 0:
        scale = 1.0
    else:
        scale = 1.0 / bag_size

    if has_per_sample_weights:
        pw = tl.load(per_sample_weights_ptr + pid_s).to(tl.float32)
        scale = scale * pw

    if scale_grad_by_freq:
        if USE_FREQ_KERNEL:
            # The freq table is precomputed by a separate kernel, avoiding a full rescan of indices per sample.
            freq = tl.load(freq_ptr + idx, valid, other=1).to(tl.float32)
            scale = scale / tl.maximum(freq, 1.0)
        else:
            n_samples = tl.num_programs(0)
            count = 0
            for i in range(n_samples):
                loaded_idx = tl.load(indices_ptr + i).to(tl.int32)
                count = count + (loaded_idx == idx)
            if count > 1:
                scale = scale / tl.cast(count, tl.float32)

    # ATen computes the per-sample scale and the contribution itself in the
    # input dtype (bf16/fp16), then accumulates; reproduce that so the 16-bit
    # comparison is exact.  For fp32 both casts are no-ops (bit-identical).
    element_ty = grad_ptr.dtype.element_ty
    scale = scale.to(element_ty).to(tl.float32)
    go = (go * scale).to(element_ty).to(tl.float32)
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    tl.atomic_add(gw_ptrs, go, mask=mask_d & valid)


MAX_BLOCK = 1024


@libentry()
@triton.jit
def _embedding_bag_dense_backward_max_kernel(
    grad_ptr,
    maximum_indices_ptr,
    grad_weight_ptr,
    n_max,
    num_bags,
    num_weights,
    stride_gb,
    stride_gd,
    padding_idx,
    EMBED_DIM: tl.constexpr,
    HAS_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Max-mode scatter: g[maximum_indices[b, d], d] += grad[b, d].

    ATen routes each (bag, dim) upstream gradient to the table row that
    produced the bag maximum; ``maximum_indices`` holds that row per (bag,
    dim). Padding-row and out-of-range targets are dropped.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_max
    row = tl.load(maximum_indices_ptr + offs, mask=mask, other=-1).to(tl.int64)
    d = offs % EMBED_DIM
    bag = offs // EMBED_DIM
    live = mask & (bag < num_bags) & (row >= 0) & (row < num_weights)
    if HAS_PAD:
        live = live & (row != padding_idx)
    value = tl.load(
        grad_ptr + bag * stride_gb + d * stride_gd, mask=live, other=0.0
    ).to(tl.float32)
    tl.atomic_add(grad_weight_ptr + row * EMBED_DIM + d, value, mask=live)


@libentry()
@triton.jit
def _embedding_bag_bucket_scan_kernel(
    counts_ptr,
    start_ptr,
    cursor_ptr,
    num_weights,
    NUM_CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Exclusive prefix sum counts -> start (single-CTA serial carry, nw=1 to dodge the wide-cumsum pathology).

    Both start and cursor are written in the same loop: start gives the reduction read ranges and
    cursor the scatter incrementing, saving a host-side copy launch.
    """
    carry = 0
    for chunk_idx in range(0, NUM_CHUNK):
        offs = chunk_idx * BLOCK + tl.arange(0, BLOCK)
        cnt = tl.load(counts_ptr + offs, mask=offs < num_weights, other=0)
        inclusive = tl.cumsum(cnt, axis=0)
        chunk_total = tl.sum(cnt, axis=0)
        exclusive = inclusive - cnt + carry
        live = offs < num_weights
        tl.store(start_ptr + offs, exclusive, mask=live)
        tl.store(cursor_ptr + offs, exclusive, mask=live)
        carry = carry + chunk_total
    tl.store(start_ptr + num_weights, carry)


@libentry()
@triton.jit
def _embedding_bag_bucket_scatter_kernel(
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    per_sample_weights_ptr,
    freq_ptr,
    cursor_ptr,
    perm_ptr,
    scale_ptr,
    num_samples,
    num_weights,
    padding_idx,
    mode: tl.constexpr,
    has_per_sample_weights: tl.constexpr,
    scale_grad_by_freq: tl.constexpr,
    SCALE_TRIVIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Counting-sort placement: each sample takes a slot inside its index bucket and its scale is computed too.

    perm[slot] = the grad row index of that sample (bag_idx, already computed by the scatter kernel).
    With SCALE_TRIVIAL (mode=0, no per_sample_weights and no frequency scaling) the scale is always 1,
    so the whole scale computation and store are cut, and the reduction kernel skips the scale read.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < num_samples
    idx = tl.load(indices_ptr + offs, mask=in_range, other=-1).to(tl.int32)
    valid = in_range & (idx >= 0) & (idx < num_weights) & (idx != padding_idx)

    bag_idx = tl.load(offset2bag_ptr + offs, mask=in_range, other=0).to(tl.int32)
    slot = tl.atomic_add(cursor_ptr + idx, 1, mask=valid)
    tl.store(perm_ptr + slot, bag_idx, mask=valid)

    if not SCALE_TRIVIAL:
        if mode != 0:
            bag_size = tl.load(bag_size_ptr + bag_idx, mask=in_range, other=1.0).to(
                tl.float32
            )
            bag_size = tl.where(bag_size == 0.0, 1.0, bag_size)
            scale = 1.0 / bag_size
        else:
            scale = tl.full([BLOCK], 1.0, tl.float32)
        if has_per_sample_weights:
            pw = tl.load(per_sample_weights_ptr + offs, mask=in_range, other=1.0).to(
                tl.float32
            )
            scale = scale * pw
        if scale_grad_by_freq:
            freq = tl.load(freq_ptr + idx, mask=valid, other=1).to(tl.float32)
            scale = scale / tl.maximum(freq, 1.0)
        tl.store(scale_ptr + slot, scale, mask=valid)


@libentry()
@triton.jit
def _embedding_bag_bucket_reduce_fixed_grid_kernel(
    perm_ptr,
    scale_ptr,
    start_ptr,
    grad_ptr,
    grad_weight_ptr,
    num_weights,
    NUM_WORKERS: tl.constexpr,
    SCALE_TRIVIAL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    """Fixed-grid segmented reduction: one program strides over several weight rows.

    The per-row grid (grid=num_weights) collapses once num_weights reaches the thousands because of
    per-program fixed cost (measured: on 512bag/256dim/w1000 the reduction is ~146us of 250us, and
    merely walking rows whose bucket has 0 samples already costs 0.079us/row). This caps the
    program count at NUM_WORKERS with an in-kernel row stride, keeping the per-bucket sample chain
    unchanged; the whole output table is overwritten, so no pre-zeroing is needed.
    With SCALE_TRIVIAL the sample chain reduces to one scalar perm read, and prefetching the next
    sample perm overlaps the current gradient row read, hiding scalar read latency; the output is
    stored directly in grad dtype (one fp32 accumulator rounding, bit-identical to the old path).
    """
    worker_id = tl.program_id(0)
    d_tile = tl.program_id(1)
    offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM
    element_ty = grad_weight_ptr.dtype.element_ty

    for weight_idx in range(worker_id, num_weights, NUM_WORKERS):
        lo = tl.load(start_ptr + weight_idx)
        hi = tl.load(start_ptr + weight_idx + 1)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        if SCALE_TRIVIAL:
            if hi > lo:
                # Prefetch the perm for the next slot: overlap the scalar read with the current gradient row read to
                # hide scalar-chain latency.
                bag_a = tl.load(perm_ptr + lo)
                for row in range(lo, hi):
                    nxt = tl.minimum(row + 1, hi - 1)
                    bag_b = tl.load(perm_ptr + nxt)
                    grad_vec = tl.load(
                        grad_ptr + bag_a * EMBED_DIM + offs_d,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    acc += grad_vec
                    bag_a = bag_b
        else:
            if hi > lo:
                bag_a = tl.load(perm_ptr + lo)
                scale_a = tl.load(scale_ptr + lo)
                for row in range(lo, hi):
                    nxt = tl.minimum(row + 1, hi - 1)
                    bag_b = tl.load(perm_ptr + nxt)
                    scale_b = tl.load(scale_ptr + nxt)
                    grad_vec = tl.load(
                        grad_ptr + bag_a * EMBED_DIM + offs_d,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    # ATen rounds scale and each contribution within the input dtype before accumulating; this is
                    # identity under fp32.
                    scale = scale_a.to(element_ty).to(tl.float32)
                    acc += (grad_vec * scale).to(element_ty).to(tl.float32)
                    bag_a = bag_b
                    scale_a = scale_b
        tl.store(
            grad_weight_ptr + weight_idx * EMBED_DIM + offs_d,
            acc.to(element_ty),
            mask=mask_d,
        )


def _zero_filled(shape, device, dtype):
    """torch.empty + a built-in narrow-grid zeroing kernel: dodges the fixed cost and sync of aten zeros."""
    tensor = torch.empty(shape, device=device, dtype=dtype)
    _embedding_bag_zero_fill_kernel[(triton.cdiv(tensor.numel(), ZERO_FILL_BLOCK),)](
        tensor, tensor.numel(), BLOCK=ZERO_FILL_BLOCK
    )
    return tensor


def _embedding_bag_dense_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    maximum_indices: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool,
    mode: int,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
) -> torch.Tensor:
    """Compute gradient for embedding_bag dense backward (GCU400 tuned)."""
    logger.debug("GEMS_ENFLAME _EMBEDDING_BAG_DENSE_BACKWARD")

    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    assert (
        grad.device.type == flag_gems.device
        and indices.device.type == flag_gems.device
        and offset2bag.device.type == flag_gems.device
        and bag_size.device.type == flag_gems.device
        and grad.device == indices.device
    ), "All inputs must be on the same device."

    device = grad.device
    num_bags = grad.shape[0]
    embedding_dim = grad.shape[1]
    num_samples = indices.numel()

    assert num_bags == bag_size.numel(), "num_bags must match bag_size length"

    # GCU's aten::_embedding_bag may return an empty offset2bag in SUM mode.
    # Under ATen semantics, only samples with a sample→bag mapping participate in accumulation; here the valid sample
    # count
    # is reduced to the mapping table length, avoiding out-of-bounds dereference of a null pointer in the kernel (which
    # triggers an async abort).
    mapped_samples = min(num_samples, offset2bag.numel())

    if mode == 2:
        # Max mode never reads offset2bag/bag_size: every (bag, dim) gradient
        # belongs to the single row recorded in maximum_indices.
        grad_weight = _zero_filled((num_weights, embedding_dim), device, torch.float32)
        n_max = maximum_indices.numel()
        if n_max > 0:
            _embedding_bag_dense_backward_max_kernel[(triton.cdiv(n_max, MAX_BLOCK),)](
                grad,
                maximum_indices,
                grad_weight,
                n_max,
                num_bags,
                num_weights,
                grad.stride(0),
                grad.stride(1) if grad.dim() > 1 else 1,
                padding_idx,
                EMBED_DIM=embedding_dim,
                HAS_PAD=0 <= padding_idx < num_weights,
                BLOCK=MAX_BLOCK,
            )
        if grad.dtype != torch.float32:
            return grad_weight.to(grad.dtype)
        return grad_weight

    if mapped_samples == 0:
        # No mapping means no gradient to accumulate, so return a zero-gradient table directly (trivial-case short
        # circuit).
        return _zero_filled((num_weights, embedding_dim), device, grad.dtype)

    # Use a narrower BLOCK_D for narrow embedding_dim: avoid masking out most of the 128 lanes.
    block_d = min(BLOCK_D_CAP, triton.next_power_of_2(max(embedding_dim, 1)))

    if per_sample_weights is not None:
        per_sample_weights_input = per_sample_weights.contiguous()
    else:
        per_sample_weights_input = None

    # Small/medium shapes: full scan in a single kernel (one launch, no intermediate buffer, no per-element fp atomics).
    if (
        mapped_samples * embedding_dim >= GROUP_MIN_ELEMS
        and 0 < num_weights <= 65535
        and block_d >= 16
    ):
        return _embedding_bag_bucket_backward(
            grad,
            indices,
            offset2bag,
            bag_size,
            per_sample_weights_input,
            num_weights,
            scale_grad_by_freq,
            mode,
            padding_idx,
            mapped_samples,
            block_d,
        )

    grad_weight = _embedding_bag_atomic_backward(
        grad,
        indices,
        offset2bag,
        bag_size,
        per_sample_weights_input,
        num_weights,
        scale_grad_by_freq,
        mode,
        padding_idx,
        mapped_samples,
        block_d,
    )

    if grad.dtype != torch.float32:
        return grad_weight.to(grad.dtype)
    return grad_weight


def _embedding_bag_atomic_backward(
    grad,
    indices,
    offset2bag,
    bag_size,
    per_sample_weights,
    num_weights,
    scale_grad_by_freq,
    mode,
    padding_idx,
    mapped_samples,
    block_d,
):
    """Original atomic accumulation path: a single launch for small shapes, winning on fixed cost."""
    device = grad.device
    embedding_dim = grad.shape[1]
    num_d_tiles = triton.cdiv(embedding_dim, block_d)

    grad_weight = _zero_filled((num_weights, embedding_dim), device, torch.float32)

    if USE_1D_GRID:
        grid = (mapped_samples * num_d_tiles,)
    else:
        grid = (mapped_samples, num_d_tiles)

    # Precompute frequencies when scale_grad_by_freq=True; otherwise reuse the output tensor as a placeholder pointer.
    freq_table = grad_weight
    if USE_FREQ_KERNEL and scale_grad_by_freq:
        freq_table = _zero_filled((num_weights,), device, torch.int32)
        _embedding_bag_freq_kernel[(triton.cdiv(mapped_samples, FREQ_BLOCK),)](
            indices,
            freq_table,
            mapped_samples,
            num_weights,
            padding_idx,
            BLOCK=FREQ_BLOCK,
        )

    if per_sample_weights is not None:
        per_sample_weights_input = per_sample_weights
    elif USE_PSW_PLACEHOLDER:
        # When HAS_PSW=False the kernel doesn't dereference this pointer, so reusing the output avoids an extra
        # allocation.
        per_sample_weights_input = grad_weight
    else:
        per_sample_weights_input = torch.ones(
            mapped_samples, device=device, dtype=torch.float32
        )

    _embedding_bag_dense_backward_kernel[grid](
        grad,
        indices,
        offset2bag,
        bag_size,
        grad_weight,
        freq_table,
        num_weights,
        padding_idx,
        mode,
        per_sample_weights is not None,
        per_sample_weights_input,
        scale_grad_by_freq,
        BLOCK_D=block_d,
        EMBED_DIM=embedding_dim,
        NUM_D_TILES=num_d_tiles,
        USE_1D_GRID=USE_1D_GRID,
        USE_FREQ_KERNEL=USE_FREQ_KERNEL,
        num_warps=NUM_WARPS,
    )
    return grad_weight


def _embedding_bag_bucket_backward(
    grad,
    indices,
    offset2bag,
    bag_size,
    per_sample_weights,
    num_weights,
    scale_grad_by_freq,
    mode,
    padding_idx,
    mapped_samples,
    block_d,
):
    """Grouped aggregation: histogram -> prefix sum -> counting sort -> segmented reduction (no per-element fp atomics).

    For small/medium num_weights the histogram + prefix sum are done in one single-CTA merged kernel
    (including counts zeroing, so counts may live in an uninitialized buffer); the whole output table
    is overwritten in grad dtype by the reduction kernel.
    """
    device = grad.device
    embedding_dim = grad.shape[1]
    num_d_tiles = triton.cdiv(embedding_dim, block_d)
    scale_trivial = (
        (mode == 0) and (per_sample_weights is None) and (not scale_grad_by_freq)
    )

    # One int32 workspace holds four tables: counts / start(nw+1) / cursor / perm(mapped),
    # reducing the number of small allocations; the entire output table is overwritten by the reduction kernel, so no
    # pre-zeroing is needed.
    scratch = torch.empty(
        3 * num_weights + 1 + mapped_samples, device=device, dtype=torch.int32
    )
    counts = scratch[0:num_weights]
    start = scratch[num_weights : 2 * num_weights + 1]
    cursor = scratch[2 * num_weights + 1 : 3 * num_weights + 1]
    perm = scratch[3 * num_weights + 1 : 3 * num_weights + 1 + mapped_samples]
    grad_weight = torch.empty(
        (num_weights, embedding_dim), device=device, dtype=grad.dtype
    )

    # 1) Histogram + exclusive prefix sum: start[w] = starting slot of bucket w in perm
    #    (counts also serves as the frequency table for scale_grad_by_freq; an inlined single-CTA histogram is
    #    bottlenecked by intra-CTA atomic throughput on GCU (measured 384us for 4096 samples vs multi-CTA
    #    frequency kernel at 21us), so keep the three-step chain "zero-fill kernel + multi-CTA frequency kernel +
    # single-CTA scan kernel".)
    _embedding_bag_zero_fill_kernel[(triton.cdiv(num_weights, ZERO_FILL_BLOCK),)](
        counts, num_weights, BLOCK=ZERO_FILL_BLOCK
    )
    _embedding_bag_freq_kernel[(triton.cdiv(mapped_samples, FREQ_BLOCK),)](
        indices,
        counts,
        mapped_samples,
        num_weights,
        padding_idx,
        BLOCK=FREQ_BLOCK,
    )
    scan_block = min(8192, max(64, triton.next_power_of_2(num_weights)))
    num_chunk = triton.cdiv(num_weights, scan_block)
    _embedding_bag_bucket_scan_kernel[(1,)](
        counts,
        start,
        cursor,
        num_weights,
        NUM_CHUNK=num_chunk,
        BLOCK=scan_block,
        num_warps=SCAN_WARPS,
    )

    if scale_trivial:
        # Scaling is always 1: the scale table is neither allocated nor written, and neither kernel dereferences this
        # placeholder pointer.
        sample_scale = grad_weight
    else:
        sample_scale = torch.empty(mapped_samples, device=device, dtype=torch.float32)

    # 2) Slot allocation (cursor is already written by the prefix-sum kernel, with the same value as start)
    _embedding_bag_bucket_scatter_kernel[(triton.cdiv(mapped_samples, SCATTER_BLOCK),)](
        indices,
        offset2bag,
        bag_size,
        per_sample_weights if per_sample_weights is not None else grad_weight,
        counts,
        cursor,
        perm,
        sample_scale,
        mapped_samples,
        num_weights,
        padding_idx,
        mode,
        per_sample_weights is not None,
        scale_grad_by_freq,
        SCALE_TRIVIAL=scale_trivial,
        BLOCK=SCATTER_BLOCK,
    )

    # 3) Segmented reduction + contiguous write-back
    _embedding_bag_bucket_reduce_fixed_grid_kernel[
        (min(num_weights, REDUCE_WORKERS), num_d_tiles)
    ](
        perm,
        sample_scale,
        start,
        grad,
        grad_weight,
        num_weights,
        NUM_WORKERS=REDUCE_WORKERS,
        SCALE_TRIVIAL=scale_trivial,
        BLOCK_D=block_d,
        EMBED_DIM=embedding_dim,
        num_warps=REDUCE_WARPS,
    )
    return grad_weight
