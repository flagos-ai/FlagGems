# Copyright 2026, The FlagOS Contributors.
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


@libentry()
@triton.jit
def histogramdd_kernel(
    input_ptr,  # (D, N) contiguous float input (transposed from the caller)
    hist_ptr,  # (num_bins,) fp32 histogram accumulator
    edge_mins_ptr,  # (D,) per-dimension lower edge
    edge_maxs_ptr,  # (D,) per-dimension upper edge
    bins_ptr,  # (D,) per-dimension bin count
    strides_ptr,  # (D,) row-major strides (stride[d] = prod(bins[d+1:]))
    N,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bin-major histogram: one program per output bin.

    The generic KernelGen kernel accumulates into the histogram with a scalar
    ``tl.atomic_add`` inside a per-point loop, which is unreliable on this XPU
    backend (cross-program atomics drop or double-count increments).  Instead each
    program owns one linear bin index ``b`` and scans the whole point cloud,
    computing every point's row-major linear bin index and counting the points that
    land in ``b``.  Counting uses a per-program int32 vector accumulator (the same
    ``grid = (bins,)`` scan pattern as ``histc``), so no cross-program atomic is
    ever issued.

    The input is transposed to ``(D, N)`` so each dimension's ``N`` coordinates are
    contiguous.  Accessing column ``d`` of the native ``(N, D)`` layout is a
    strided gather (``input[i*D + d]``, stride ``D``); on this backend a large
    strided gather hangs the card (``kl3_wait_for_noc_idle() timeout`` / NOC).
    Transposing turns every ``input_T[d*N + i]`` into a stride-1 block DMA load.
    """
    b = ext.program_id(0)
    # Single tile: BLOCK_SIZE is next_power_of_2(N), so every point is covered in
    # one shot.  A runtime outer loop (grid-stride over N) trips the XPU
    # TritonXPUMemoryInplace/dominance verifier on the nested D-loop, so the tile
    # is sized to hold the whole point cloud instead.
    offs = tl.arange(0, BLOCK_SIZE)
    # Clamp the address before the unmasked load: this backend ignores the
    # ``other`` fill of a masked load and reads neighbouring memory for the
    # out-of-range lanes, so the tail is handled with an explicit clamp plus an
    # ``ok`` mask instead.
    safe = tl.minimum(offs, N - 1)
    ok = (offs < N).to(tl.int32)

    linear = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    in_range = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)
    for d in range(D):
        val = tl.load(input_ptr + d * N + safe).to(tl.float32)
        edge_min = tl.load(edge_mins_ptr + d).to(tl.float32)
        edge_max = tl.load(edge_maxs_ptr + d).to(tl.float32)
        num_bins = tl.load(bins_ptr + d).to(tl.int32)
        stride = tl.load(strides_ptr + d).to(tl.int32)

        # A NaN fails every comparison; exclude it explicitly.
        cur_in = (val >= edge_min) & (val <= edge_max) & (val == val)
        in_range = in_range & cur_in.to(tl.int32)

        # Uniform-bin closed form: bin = floor((val - min) / width).  Values
        # exactly at the upper edge fold into the last bin.
        bin_width = (edge_max - edge_min) / num_bins.to(tl.float32)
        bin_idx = tl.floor((val - edge_min) / bin_width).to(tl.int32)
        bin_idx = tl.where(val >= edge_max, num_bins - 1, bin_idx)
        bin_idx = tl.maximum(0, tl.minimum(num_bins - 1, bin_idx))
        linear = linear + bin_idx * stride

    hit = (linear == b).to(tl.int32) & in_range & ok
    tl.store(hist_ptr + b, tl.sum(hit, axis=0).to(tl.float32))


@libentry()
@triton.jit
def histogramdd_range_kernel(
    input_ptr,  # (D, N) contiguous float input (transposed from the caller)
    mn_ptr,  # (D,) per-dimension minimum
    mx_ptr,  # (D,) per-dimension maximum
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per dimension: reduce its contiguous row to (min, max).

    Replaces ``input[:, d].min().item()`` / ``.max().item()``, which issue two
    device->host round trips per dimension.  Each program owns one row of the
    transposed ``(D, N)`` layout (stride-1 loads) and writes a scalar, so the
    host pays a single ``.tolist()`` sync per extremum regardless of ``D``.
    """
    d = ext.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    safe = tl.minimum(offs, N - 1)
    vals = tl.load(input_ptr + d * N + safe).to(tl.float32)
    # Out-of-range tail lanes are masked with the neutral element so they do not
    # pollute the reduction.
    mn = tl.min(tl.where(offs < N, vals, float("inf")), axis=0)
    mx = tl.max(tl.where(offs < N, vals, float("-inf")), axis=0)
    tl.store(mn_ptr + d, mn)
    tl.store(mx_ptr + d, mx)


def histogramdd(input, bins, range=None, weight=None, density=False):
    """Compute the D-dimensional histogram of a (N, D) input tensor.

    Functional override of the generic KernelGen ``histogramdd``: identical edge
    semantics and density normalisation, but the counting kernel is atomic-free
    (see :func:`histogramdd_kernel`).
    """
    logger.debug("GEMS_KUNLUNXIN HISTOGRAMDD")

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

    input = input.contiguous()

    # Transpose to (D, N) up front: each dimension's N coordinates become
    # contiguous, so both the min/max reduction and the counting kernel use
    # stride-1 block DMA loads instead of strided gathers (a strided gather at
    # large BLOCK_SIZE hangs this backend's NOC).
    input_t = input.t().contiguous()

    range_vals = range
    bin_edges = []
    edge_min_vals = []
    edge_max_vals = []

    # GPU per-dimension min/max: one program per dimension scans its contiguous
    # row and reduces to a scalar, then a single host sync per extremum.  This
    # replaces ``input[:, d].min().item()`` / ``.max().item()``, which issue two
    # device->host round trips per dimension.
    if range_vals is None and N > 0:
        mn_t = torch.empty(D, dtype=torch.float32, device=input.device)
        mx_t = torch.empty(D, dtype=torch.float32, device=input.device)
        with torch_device_fn.device(input.device):
            histogramdd_range_kernel[(D,)](
                input_t, mn_t, mx_t, N, BLOCK_SIZE=triton.next_power_of_2(N)
            )
        mn_host = mn_t.tolist()
        mx_host = mx_t.tolist()

    for d in builtins.range(D):
        num_bins = bins[d]

        if range_vals is not None:
            dim_min = float(range_vals[2 * d])
            dim_max = float(range_vals[2 * d + 1])
        elif N == 0:
            dim_min = 0.0
            dim_max = 1.0
        else:
            dim_min = float(mn_host[d])
            dim_max = float(mx_host[d])

        # Degenerate range (all values equal) folds into [v-0.5, v+0.5].
        if dim_min == dim_max:
            lo = dim_min - 0.5
            hi = dim_min + 0.5
        else:
            lo = dim_min
            hi = dim_max

        edges = torch.linspace(
            lo, hi, num_bins + 1, dtype=input.dtype, device=input.device
        )
        bin_edges.append(edges)
        edge_min_vals.append(lo)
        edge_max_vals.append(hi)

    hist_shape = tuple(bins)
    hist = torch.zeros(hist_shape, dtype=input.dtype, device=input.device)

    if N == 0:
        return hist, bin_edges

    bins_tensor = torch.tensor(bins, dtype=torch.int32, device=input.device)
    edge_mins = torch.tensor(edge_min_vals, dtype=input.dtype, device=input.device)
    edge_maxs = torch.tensor(edge_max_vals, dtype=input.dtype, device=input.device)

    # Row-major strides: stride[d] = prod(bins[d+1:]), last dim fastest.
    strides_host = [1] * D
    for d in builtins.range(D - 2, -1, -1):
        strides_host[d] = strides_host[d + 1] * bins[d + 1]
    strides = torch.tensor(strides_host, dtype=torch.int32, device=input.device)

    num_bins = 1
    for b in bins:
        num_bins *= b

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (num_bins,)

    with torch_device_fn.device(input.device):
        histogramdd_kernel[grid](
            input_t,
            hist,
            edge_mins,
            edge_maxs,
            bins_tensor,
            strides,
            N,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    if density:
        bin_volumes = torch.ones_like(hist)
        for d in builtins.range(D):
            widths = bin_edges[d][1:] - bin_edges[d][:-1]
            shape = [1] * D
            shape[d] = bins[d]
            widths_shaped = widths.reshape(shape)
            bin_volumes = bin_volumes * widths_shaped

        total_count = hist.sum()
        if total_count > 0:
            hist = hist / total_count / bin_volumes
        else:
            hist = hist / bin_volumes

    return hist, bin_edges
