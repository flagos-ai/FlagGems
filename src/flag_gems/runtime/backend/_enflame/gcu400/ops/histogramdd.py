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

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Device-side bins tensors are immutable and tiny, but building one costs a
# ~100us pageable H2D copy on GCU; cache them per (values, device).
_BINS_TENSOR_CACHE = {}


def _bins_device_tensor(bins, device):
    key = (tuple(int(b) for b in bins), device)
    cached = _BINS_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(bins, dtype=torch.int32, device=device)
        if len(_BINS_TENSOR_CACHE) < 64:
            _BINS_TENSOR_CACHE[key] = cached
    return cached


@libentry()
@triton.jit
def histogramdd_kernel(
    input_ptr,
    hist_ptr,
    mins_ptr,  # Per-dimension data minima (raw, not widened)
    maxs_ptr,  # Per-dimension data maxima (raw, not widened)
    edges_ptr,  # [D, max_bins + 1] output buffer for uniform bin edges
    N,
    D,
    bins_ptr,  # Array of bin counts per dimension
    total_cells,
    edges_row_stride,
    BLOCK_SIZE: tl.constexpr,
    CELLS: tl.constexpr,  # Padded power-of-2 cell count (onehot path)
    USE_ONEHOT: tl.constexpr,
    NREP: tl.constexpr,  # Number of histogram replicas (contention dilution)
    FP64_INPUT: tl.constexpr,
):
    """
    Compute D-dimensional histogram. Each program handles BLOCK_SIZE points
    in a vectorized fashion. Degenerate (all-equal) dimensions are widened
    to [v - 0.5, v + 0.5] and the uniform bin edges are written by program 0.

    Merge strategy (constexpr-selected):
    - USE_ONEHOT: per-program local counts via onehot + tl.sum(axis=0), then
      a single masked atomic per cell. Avoids the N-way same-address atomic
      serialization that dominates when the histogram has few cells.
    - NREP > 1: points of program pid atomically update replica pid % NREP,
      cutting per-cell contention by NREP; the host sums the replicas.
    - otherwise: direct atomic per point.
    """
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    linear_bin_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    in_range = mask
    stride = 1

    # Iterate dimensions in reverse order for row-major indexing
    for d in range(D - 1, -1, -1):
        # Vectorized load of this dimension for all points in the block
        val = tl.load(input_ptr + offs * D + d, mask=mask, other=float("nan"))

        # Load bin count for this dimension
        num_bins = tl.load(bins_ptr + d).to(tl.int32)

        # Load raw boundaries and widen degenerate (all-equal) dimensions
        vmin = tl.load(mins_ptr + d)
        vmax = tl.load(maxs_ptr + d)
        widen = tl.where(vmin == vmax, 0.5, 0.0)
        edge_min = vmin - widen
        edge_max = vmax + widen

        # Program 0 writes the uniform bin edges for this dimension
        # (edges_buf is always f32: the GCU backend miscompiles f32->f64
        # scalar casts on f64 stores, writing only half of each slot)
        if pid == 0:
            for k in range(0, num_bins + 1):
                e = edge_min + (edge_max - edge_min) * k / num_bins
                tl.store(edges_ptr + d * edges_row_stride + k, e)

        # Check if value is in range (val != val checks for NaN)
        # Note: edge_max is inclusive (values exactly at edge_max are valid)
        is_nan = val != val
        dim_in_range = ~((val < edge_min) | (val > edge_max) | is_nan)
        in_range = in_range & dim_in_range

        # Compute bin index using binary search approximation
        # For uniform bins: bin_idx = floor((val - min) / bin_width)
        bin_width = (edge_max - edge_min) / num_bins
        bin_idx = tl.floor((val - edge_min) / bin_width).to(tl.int32)

        # Clamp to valid range and handle right edge
        # Values exactly at edge_max should go into the last bin (num_bins - 1)
        bin_idx = tl.where(val >= edge_max, num_bins - 1, bin_idx)
        bin_idx = tl.maximum(0, tl.minimum(num_bins - 1, bin_idx))

        # Accumulate linear index (row-major order)
        linear_bin_idx += bin_idx * stride
        stride = stride * num_bins

    # Merge per-point counts into the histogram.
    # For fp64 inputs the histogram is accumulated in int32 (GCU atomic_rmw
    # has no f64 support) and cast back to fp64 on the host.
    if USE_ONEHOT:
        onehot = (linear_bin_idx[:, None] == tl.arange(0, CELLS)[None, :]) & in_range[
            :, None
        ]
        counts = tl.sum(onehot.to(tl.int32), axis=0)
        cell_ids = tl.arange(0, CELLS)
        cell_mask = cell_ids < total_cells
        if FP64_INPUT:
            tl.atomic_add(hist_ptr + cell_ids, counts, mask=cell_mask, sem="relaxed")
        else:
            tl.atomic_add(
                hist_ptr + cell_ids,
                counts.to(hist_ptr.dtype.element_ty),
                mask=cell_mask,
                sem="relaxed",
            )
    elif NREP > 1:
        rep_base = (pid % NREP) * total_cells
        if FP64_INPUT:
            tl.atomic_add(
                hist_ptr + rep_base + linear_bin_idx, 1, mask=in_range, sem="relaxed"
            )
        else:
            tl.atomic_add(
                hist_ptr + rep_base + linear_bin_idx, 1.0, mask=in_range, sem="relaxed"
            )
    else:
        if FP64_INPUT:
            tl.atomic_add(hist_ptr + linear_bin_idx, 1, mask=in_range, sem="relaxed")
        else:
            tl.atomic_add(hist_ptr + linear_bin_idx, 1.0, mask=in_range, sem="relaxed")


def histogramdd(input, bins, range=None, weight=None, density=False):
    """
    Compute D-dimensional histogram of input tensor.

    Args:
        input: Input tensor of shape (N, D) where N is number of points, D is dimensions
        bins: List of int specifying number of bins per dimension
        range: Optional list of [min0, max0, min1, max1, ...] for D dimensions
        weight: Optional weight tensor of shape (N,)
        density: If True, normalize to probability density

    Returns:
        tuple: (hist, bin_edges) where hist has shape (bins[0], bins[1], ..., bins[D-1])
               and bin_edges is a list of D edge tensors
    """
    logger.debug("GEMS HISTOGRAMDD")

    if input.dtype not in [torch.float32, torch.float64]:
        raise NotImplementedError(
            f"histogramdd not implemented for dtype {input.dtype}. "
            "Only float32 and float64 are supported."
        )

    if input.dim() != 2:
        raise ValueError(
            f"histogramdd requires 2D input (N, D), got shape {input.shape}"
        )

    N, D = input.shape
    if D < 2:
        raise ValueError(f"histogramdd requires at least 2 dimensions, got D={D}")

    if len(bins) != D:
        raise ValueError(f"bins length {len(bins)} must match input dimensions {D}")

    if weight is not None:
        if weight.shape != (N,):
            raise ValueError(f"weight must have shape ({N},), got {weight.shape}")
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight dtype {weight.dtype} must match input dtype {input.dtype}"
            )
        raise NotImplementedError("weighted histogramdd not yet implemented")

    # Ensure input is contiguous
    input = input.contiguous()

    # Store range parameter to avoid shadowing builtin range()
    range_vals = range

    # Empty input: default range [0, 1] per dimension, zero histogram
    if N == 0:
        bin_edges = [
            torch.linspace(
                0.0, 1.0, bins[d] + 1, dtype=input.dtype, device=input.device
            )
            for d in builtins.range(D)
        ]
        hist = torch.zeros(tuple(bins), dtype=input.dtype, device=input.device)
        return hist, bin_edges

    # GCU atomic_rmw has no f64 support: accumulate fp64 histograms in int32
    # (counts are exact integers) and cast back to fp64 afterwards.
    fp64_input = input.dtype == torch.float64
    acc_dtype = torch.int32 if fp64_input else input.dtype
    hist_shape = tuple(bins)

    if fp64_input:
        # The GCU backend has no vector f64 ALU either: run the kernel on an
        # f32 copy of the data. torch_gcu already computes fp64 aten ops
        # (min/max/linspace) with f32 kernels, so this keeps the platform's
        # own precision semantics for fp64.
        kernel_input = input.to(torch.float32)
    else:
        kernel_input = input

    # Per-dimension boundaries stay on device (no host synchronization);
    # torch.aminmax fuses the two column reductions into one kernel launch
    if range_vals is not None:
        kernel_mins = torch.tensor(
            [float(range_vals[2 * d]) for d in builtins.range(D)],
            dtype=kernel_input.dtype,
            device=input.device,
        )
        kernel_maxs = torch.tensor(
            [float(range_vals[2 * d + 1]) for d in builtins.range(D)],
            dtype=kernel_input.dtype,
            device=input.device,
        )
    else:
        kernel_mins, kernel_maxs = torch.aminmax(kernel_input, dim=0)

    total_cells = 1
    for num_bins in bins:
        total_cells *= num_bins

    max_bins = max(bins)
    # f32 edge buffer for every dtype (see kernel comment on f64 stores);
    # fp64 callers get per-dimension f64 copies of the views below.
    edges_buf = torch.empty(
        (D, max_bins + 1), dtype=kernel_input.dtype, device=input.device
    )
    bins_tensor = _bins_device_tensor(bins, input.device)

    # Merge-strategy dispatch (see kernel docstring):
    # - few cells, small grid -> onehot local counts (avoids hot-cell atomics)
    # - many points per cell -> replicate the histogram to dilute contention
    # - otherwise -> direct atomic per point
    use_onehot = False
    nrep = 1
    cells = 1
    block = 256
    if total_cells <= 128:
        cells = triton.next_power_of_2(total_cells)
        block = 1024 if cells <= 64 else 512
        use_onehot = triton.cdiv(N, block) <= 8
    if not use_onehot and N >= 16 * total_cells and total_cells <= 32768:
        nrep = 8

    if nrep > 1:
        hist = torch.zeros((nrep,) + hist_shape, dtype=acc_dtype, device=input.device)
    else:
        hist = torch.zeros(hist_shape, dtype=acc_dtype, device=input.device)

    grid = (triton.cdiv(N, block),)
    with torch_device_fn.device(input.device):
        histogramdd_kernel[grid](
            kernel_input,
            hist,
            kernel_mins,
            kernel_maxs,
            edges_buf,
            N,
            D,
            bins_tensor,
            total_cells,
            max_bins + 1,
            BLOCK_SIZE=block,
            CELLS=cells,
            USE_ONEHOT=use_onehot,
            NREP=nrep,
            FP64_INPUT=fp64_input,
        )

    if nrep > 1:
        hist = hist.sum(dim=0)
    if fp64_input:
        hist = hist.to(input.dtype)
        # 只做一次 dtype 转换，再从结果上切 view；逐维 .to() 会多出 D 次 launch。
        edges_out = edges_buf.to(input.dtype)
        bin_edges = [edges_out[d, : bins[d] + 1] for d in builtins.range(D)]
    else:
        bin_edges = [edges_buf[d, : bins[d] + 1] for d in builtins.range(D)]

    # Apply density normalization if requested
    if density:
        # Compute bin volumes
        bin_volumes = torch.ones_like(hist)
        for d in builtins.range(D):
            widths = bin_edges[d][1:] - bin_edges[d][:-1]
            # Broadcast widths along dimension d
            shape = [1] * D
            shape[d] = bins[d]
            widths_shaped = widths.reshape(shape)
            bin_volumes = bin_volumes * widths_shaped

        total_count = hist.sum()
        if total_count > 0:
            hist = hist / total_count / bin_volumes
        else:
            # All points were out of range; density is undefined (NaN)
            hist = hist / bin_volumes  # 0/vol = 0, will be NaN if needed

    return hist, bin_edges
