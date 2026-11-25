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
# Generated per round by the optimization driver script; only launch geometry and statistics method change, not
# semantics.
BLOCK_D_CAP = 512
NUM_WARPS = 4
USE_1D_GRID = 0
USE_FREQ_KERNEL = 1
FREQ_BLOCK = 128
USE_PSW_PLACEHOLDER = 1
# --- TUNING PARAMS END ---


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

    go = go * scale
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
    # Under ATen semantics only samples with a sample-to-bag mapping participate in accumulation; here the number of
    # valid samples is reduced to the mapping table length, avoiding out-of-bounds dereferences of a null pointer (async
    # abort).
    mapped_samples = min(num_samples, offset2bag.numel())

    grad_weight = torch.zeros(
        (num_weights, embedding_dim), device=device, dtype=torch.float32
    ).contiguous()

    if mode == 2:
        # Max mode never reads offset2bag/bag_size: every (bag, dim) gradient
        # belongs to the single row recorded in maximum_indices.
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
        # With no mapping there is no gradient to accumulate, so return a zero gradient table directly (trivial-case
        # short circuit).
        if grad.dtype != torch.float32:
            return grad_weight.to(grad.dtype)
        return grad_weight

    # Narrow embedding_dim uses a narrower BLOCK_D to avoid masking off most of the 128 lanes.
    block_d = min(BLOCK_D_CAP, triton.next_power_of_2(max(embedding_dim, 1)))
    num_d_tiles = triton.cdiv(embedding_dim, block_d)
    if USE_1D_GRID:
        grid = (mapped_samples * num_d_tiles,)
    else:
        grid = (mapped_samples, num_d_tiles)

    # Precompute frequencies when scale_grad_by_freq=True; otherwise reuse the output tensor as a placeholder pointer.
    freq_table = grad_weight
    if USE_FREQ_KERNEL and scale_grad_by_freq:
        freq_table = torch.zeros(
            (num_weights,), device=device, dtype=torch.int32
        ).contiguous()
        _embedding_bag_freq_kernel[(triton.cdiv(mapped_samples, FREQ_BLOCK),)](
            indices,
            freq_table,
            mapped_samples,
            num_weights,
            padding_idx,
            BLOCK=FREQ_BLOCK,
        )

    if per_sample_weights is not None:
        per_sample_weights_input = per_sample_weights.contiguous()
    elif USE_PSW_PLACEHOLDER:
        # With HAS_PSW=False the kernel does not dereference that pointer, so reusing the output avoids an extra
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

    if grad.dtype != torch.float32:
        return grad_weight.to(grad.dtype)
    return grad_weight
