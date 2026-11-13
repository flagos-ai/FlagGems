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

"""gcu400 specialization of ``histogram`` (aten::histogram.bin_ct /
histogram.bins_tensor).

The common implementation is a pure-torch host pipeline
(``torch.bucketize`` + ``where`` + ``scatter_add_``); on GCU its first
device op, ``torch.bucketize``, hard-aborts inside topsaten
(``Fatal Python error: Aborted``, see
compiler_problems/P21 .../sources/flaggems_histogram_bucketize_abort.md),
and the host also violates the host-isolation rule. This copy replaces the
whole pipeline with Triton kernels:

* ``histogram_bins_tensor_kernel``: grid-stride pass computing per-program
  partial histograms via onehot + ``tl.sum(axis=0)`` (deterministic,
  atomic-free per knowledge/coding_rules; ``tl.histogram`` itself is
  untrusted on this backend - see the "`n` triple-untrustworthy" row in
  knowledge/compiler_traps). Bin assignment is the CPU bucketize
  semantics: count the edges ``<= v``, drop out-of-range/NaN, clamp the
  last edge into the last bin. Invalid lanes are remapped to a padding bin
  ``NBINS - 1`` that the reduce stage masks away. The same kernel serves
  ``bin_ct`` (uniform bins) and ``bins_tensor`` (non-uniform bins).
* ``histogram_reduce_kernel``: single program sums the per-program partials
  and applies the optional density normalization (0/0 -> NaN matches CPU).
* The CPU reference bucketizes fp32 inputs against ``at::linspace`` values
  in the tensor dtype, so ``bin_ct`` builds its uniform edges with the host
  CPU ``torch.linspace`` (a device-side linspace differs by ~1 ulp) and
  caches the H2D copy per (min, max, bins, device); this makes the fp32
  assignment bit-exact.
* fp64 has no ALU/store support on gcu400 (P04 family: vector ``arith.extf``
  rejected, scalar f32->f64 store half-slot corruption). Device-stored fp64
  values are f32-representable (torch_gcu degrades fp64 storage), so fp64
  inputs and weights run on an f32 copy, the histogram counts in int32/f32
  buffers, and the host casts the results. The fp64 assignment uses
  PAIR_EDGES: the reference's f64 linspace edges are split on the host into
  f32 hi/lo pairs, and an f32-exact v satisfies ``v >= hi + lo`` exactly
  when ``(v > hi) | ((v == hi) & (lo <= 0))`` - pure comparisons, immune
  to the mul-sub residual miscompile family (P14) that a double-single
  arithmetic chain hit. The returned fp64 bin_edges keep the original CPU
  linspace values (moved to the device).
* range=None reuses the FlagGems Triton ``aminmax`` op (composition) and
  resolves left/right on the host with one ``.item()`` sync (host
  isolation: no aten compute; the hi/lo split is done in numpy). The
  degenerate min==max widening happens on the host, before the edges are
  built.
* Bins are capped at 2047 (padded NBINS <= 2048) so the onehot tile stays
  within the DSM budget proven by the histogramdd onehot path.
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.ops.aminmax import aminmax
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
MAX_BINS = 2047  # NBINS = next_pow2(bins + 1) <= 2048
# [BLOCK, NBINS] tile budget. The bin_ct kernel keeps one 2D onehot tile live
# (64K lanes proved to fit DSM); the bins_tensor kernel keeps two (the edge
# count compare plus the onehot), so it needs the halved budget (its first
# 64K-lane attempt required 1.2MB DSM vs the 896KB hardware limit).
ONEHOT_MAX_LANES = 65536
ONEHOT_MAX_LANES_DOUBLE_TILE = 32768


def _pick_block(nbins_padded, has_weight, max_lanes=ONEHOT_MAX_LANES):
    block = max(max_lanes // nbins_padded, 32)
    block = min(block, 4096)
    if has_weight:
        block = min(block, 1024)
    return block


@libentry()
@triton.jit(do_not_specialize=["n_elements", "nbins"])
def histogram_bins_tensor_kernel(
    input_ptr,
    weight_ptr,
    edges_ptr,
    edges_lo_ptr,
    partials_ptr,
    n_elements,
    nbins,
    HAS_WEIGHT: tl.constexpr,
    PAIR_EDGES: tl.constexpr,
    NBINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Assignment is the CPU bucketize semantics: count the edges `<= v`, drop
    # out-of-range/NaN, clamp the last edge into the last bin. PAIR_EDGES
    # serves fp64 bin_ct: device fp64 storage is f32-representable on gcu400
    # (torch_gcu degrades it), so v is exact in f32 while the reference's
    # f64 linspace edges need one extra f32 term. For an f32-exact value v and
    # a hi/lo edge pair, v >= edge is EXACTLY
    #   (v > hi) | ((v == hi) & (lo <= 0))
    # because |edge - hi| <= ulp(hi)/2. Only comparisons take part - no
    # multiply/subtract chains, which this backend's fusion pass miscompiles
    # (P14: sign-flipped residual in sum-tree tails).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    arange = tl.arange(0, BLOCK)
    bin_ids = tl.arange(0, NBINS)
    # Real edges at k in [0, nbins]; padded lanes get +inf so they never
    # satisfy `v >= edge` and stay out of the count.
    edges_all = tl.load(edges_ptr + bin_ids, mask=bin_ids <= nbins, other=float("inf"))
    if PAIR_EDGES:
        edges_lo_all = tl.load(edges_lo_ptr + bin_ids, mask=bin_ids <= nbins, other=0.0)
    e_first = tl.load(edges_ptr)
    e_last = tl.load(edges_ptr + nbins)
    if PAIR_EDGES:
        e_first_lo = tl.load(edges_lo_ptr)
        e_last_lo = tl.load(edges_lo_ptr + nbins)
    if HAS_WEIGHT:
        local = tl.zeros((NBINS,), dtype=tl.float32)
    else:
        local = tl.zeros((NBINS,), dtype=tl.int32)
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < n_elements
        v = tl.load(input_ptr + off, mask=mask, other=float("nan")).to(tl.float32)
        # cnt = number of edges <= v; for in-range values cnt in [1, nbins+1],
        # so cnt-1 is the left-closed bin index and v == last edge clamps to
        # the last bin. NaN fails both range checks and is dropped.
        if PAIR_EDGES:
            ge = (v[:, None] > edges_all[None, :]) | (
                (v[:, None] == edges_all[None, :]) & (edges_lo_all[None, :] <= 0.0)
            )
            cnt = tl.sum(ge.to(tl.int32), axis=1)
            valid = (
                mask
                & ((v > e_first) | ((v == e_first) & (e_first_lo <= 0.0)))
                & ((e_last > v) | ((e_last == v) & (e_last_lo >= 0.0)))
            )
        else:
            cnt = tl.sum((v[:, None] >= edges_all[None, :]).to(tl.int32), axis=1)
            valid = mask & (v >= e_first) & (v <= e_last)
        idx = tl.where(valid, tl.minimum(cnt - 1, nbins - 1), NBINS - 1)
        onehot = idx[:, None] == bin_ids[None, :]
        if HAS_WEIGHT:
            w = tl.load(weight_ptr + off, mask=mask, other=0.0).to(tl.float32)
            local += tl.sum(onehot.to(tl.float32) * w[:, None], axis=0)
        else:
            local += tl.sum(onehot.to(tl.int32), axis=0)
    tl.store(partials_ptr + pid * NBINS + bin_ids, local)


@libentry()
@triton.jit(do_not_specialize=["n_progs", "bins"])
def histogram_reduce_kernel(
    partials_ptr,
    hist_ptr,
    edges_ptr,
    n_progs,
    bins,
    DENSITY: tl.constexpr,
    NBINS: tl.constexpr,
    GRID_PAD: tl.constexpr,
):
    g = tl.arange(0, GRID_PAD)
    b = tl.arange(0, NBINS)
    part = tl.load(
        partials_ptr + g[:, None] * NBINS + b[None, :],
        mask=(g[:, None] < n_progs) & (b[None, :] < NBINS),
        other=0,
    )
    hist = tl.sum(part, axis=0)
    bm = b < bins
    if DENSITY:
        histf = hist.to(tl.float32)
        total = tl.sum(tl.where(bm, histf, 0.0))
        e_lo = tl.load(edges_ptr + b, mask=bm, other=0.0)
        e_hi = tl.load(edges_ptr + b + 1, mask=bm, other=0.0)
        width = e_hi - e_lo
        # total == 0 (all values out of range / empty input) gives 0/0 = NaN,
        # matching the CPU reference for density over an empty selection.
        out = histf / (total * width)
        tl.store(hist_ptr + b, out, mask=bm)
    else:
        tl.store(hist_ptr + b, hist.to(hist_ptr.dtype.element_ty), mask=bm)


def _histogram_launch_reduce(
    partials, edges_buf, nbins, nbins_padded, grid, density, device
):
    hist = torch.empty(nbins, dtype=torch.float32, device=device)
    grid_pad = triton.next_power_of_2(grid)
    with torch_device_fn.device(device):
        histogram_reduce_kernel[(1,)](
            partials,
            hist,
            edges_buf,
            grid,
            nbins,
            DENSITY=density,
            NBINS=nbins_padded,
            GRID_PAD=grid_pad,
            num_warps=4,
        )
    return hist


# Uniform-bin edges are immutable per (range, bins) but building one costs a
# ~100us pageable H2D copy on GCU; cache them per (min, max, bins, device).
_EDGES_CACHE = {}


def _edges_device_tensor(min_val, max_val, bins, device, fp64):
    key = (min_val, max_val, bins, str(device), fp64)
    cached = _EDGES_CACHE.get(key)
    if cached is None:
        # The CPU reference bucketizes against at::linspace(left, right,
        # nbins + 1) in the tensor dtype, so producing the same values on the
        # host CPU (device-side linspace differs by ~1 ulp) makes the f32
        # assignment bit-exact; the copy to the device is data movement only.
        # The hi/lo split is done in numpy so the host stays free of aten
        # compute. For fp64 the returned bin_edges keep the original CPU
        # linspace values (moved to the device) while the kernel consumes the
        # f32 hi/lo pair (see the PAIR_EDGES note in the kernel).
        if fp64:
            edges_cpu = torch.linspace(min_val, max_val, bins + 1, dtype=torch.float64)
            pair_np = edges_cpu.numpy()
            hi = pair_np.astype(np.float32)
            lo = (pair_np - hi.astype(np.float64)).astype(np.float32)
            pair = torch.from_numpy(np.stack([hi, lo]).copy()).to(device)
            edges_out = edges_cpu.to(device)
            cached = (pair, edges_out)
        else:
            cached = (
                torch.linspace(min_val, max_val, bins + 1, dtype=torch.float32).to(
                    device
                ),
                None,
            )
        if len(_EDGES_CACHE) < 64:
            _EDGES_CACHE[key] = cached
    return cached


def _histogram_common_args(self, weight, fp64):
    input_flat = self.contiguous().view(-1)
    if fp64:
        kernel_input = input_flat.to(torch.float32)
    else:
        kernel_input = input_flat
    if weight is not None:
        weight_flat = weight.contiguous().view(-1)
        if fp64:
            kernel_weight = weight_flat.to(torch.float32)
        else:
            kernel_weight = weight_flat
        has_weight = True
    else:
        kernel_weight = kernel_input
        has_weight = False
    return kernel_input, kernel_weight, has_weight


def histogram_bin_ct(self, bins=100, *, range=None, weight=None, density=False):
    """
    Compute histogram with uniform bins (integer bin count).

    Args:
        self: input tensor (any shape)
        bins: number of bins (int, default 100)
        range: tuple of (min, max) or None (default None = use data range)
        weight: optional weight tensor (same shape as self) or None
        density: if True, normalize to form probability density

    Returns:
        hist: histogram counts (shape: [bins])
        bin_edges: bin edges (shape: [bins+1])
    """
    logger.debug("GEMS_ENFLAME HISTOGRAM.BIN_CT")

    if self.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            f"histogram not implemented for dtype {self.dtype} on the enflame "
            "backend. Only float32 and float64 are supported."
        )
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if bins > MAX_BINS:
        raise NotImplementedError(
            f"histogram with more than {MAX_BINS} bins is not supported on "
            "the enflame backend (onehot tile DSM budget)."
        )

    fp64 = self.dtype == torch.float64
    kernel_input, kernel_weight, has_weight = _histogram_common_args(self, weight, fp64)
    n_elements = kernel_input.numel()

    # Range boundaries resolve to host scalars; for range=None the reduction
    # itself still runs through the FlagGems Triton aminmax op (host
    # isolation: no aten compute), with one .item() sync. The CPU reference
    # maps an empty range=None input to [0, 1].
    if range is not None:
        min_val, max_val = float(range[0]), float(range[1])
    elif n_elements == 0:
        min_val, max_val = 0.0, 1.0
    else:
        mins, maxs = aminmax(kernel_input)
        min_val, max_val = float(mins.item()), float(maxs.item())
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5

    # Bit-exact uniform edges from the host CPU linspace (cached H2D copy):
    # the reference bucketizes against at::linspace in the tensor dtype, and
    # a device-side linspace differs from the CPU one by ~1 ulp.
    edges_buf, edges_out = _edges_device_tensor(
        min_val, max_val, bins, self.device, fp64
    )

    nbins_padded = triton.next_power_of_2(bins + 1)
    # The edge-count kernel keeps two 2D tiles live (the compare plus the
    # onehot), so it uses the halved lane budget; the fp64 pair compare adds
    # a third boolean tile of the same shape.
    block = _pick_block(nbins_padded, has_weight, ONEHOT_MAX_LANES_DOUBLE_TILE)
    grid = max(1, min(triton.cdiv(n_elements, block), NUM_SIPS * 2))
    acc_dtype = torch.float32 if has_weight else torch.int32
    partials = torch.empty((grid, nbins_padded), dtype=acc_dtype, device=self.device)
    if fp64:
        # pair layout: row 0 = f32 hi parts, row 1 = f32 lo parts.
        edges_hi = edges_buf[0]
        edges_lo = edges_buf[1]
        with torch_device_fn.device(self.device):
            histogram_bins_tensor_kernel[(grid,)](
                kernel_input,
                kernel_weight,
                edges_hi,
                edges_lo,
                partials,
                n_elements,
                bins,
                HAS_WEIGHT=has_weight,
                PAIR_EDGES=True,
                NBINS=nbins_padded,
                BLOCK=block,
                num_warps=4,
            )
        hist = _histogram_launch_reduce(
            partials, edges_hi, bins, nbins_padded, grid, density, self.device
        )
        return hist.to(self.dtype), edges_out
    else:
        with torch_device_fn.device(self.device):
            histogram_bins_tensor_kernel[(grid,)](
                kernel_input,
                kernel_weight,
                edges_buf,
                edges_buf,
                partials,
                n_elements,
                bins,
                HAS_WEIGHT=has_weight,
                PAIR_EDGES=False,
                NBINS=nbins_padded,
                BLOCK=block,
                num_warps=4,
            )
        hist = _histogram_launch_reduce(
            partials, edges_buf, bins, nbins_padded, grid, density, self.device
        )
        return hist, edges_buf


def histogram_bins_tensor(self, bins, *, weight=None, density=False):
    """
    Compute histogram with non-uniform bins (tensor bin edges).

    Args:
        self: input tensor (any shape)
        bins: 1D tensor of bin edges (must be monotonically increasing)
        weight: optional weight tensor (same shape as self) or None
        density: if True, normalize to form probability density

    Returns:
        hist: histogram counts (shape: [bins.numel()-1])
        bin_edges: bin edges (a copy of bins)
    """
    logger.debug("GEMS_ENFLAME HISTOGRAM.BINS_TENSOR")

    if self.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            f"histogram not implemented for dtype {self.dtype} on the enflame "
            "backend. Only float32 and float64 are supported."
        )

    edges = bins.contiguous()
    num_edges = edges.numel()
    if num_edges < 2:
        raise ValueError("bins must have at least 2 elements")
    nbins = num_edges - 1
    if nbins > MAX_BINS:
        raise NotImplementedError(
            f"histogram with more than {MAX_BINS} bins is not supported on "
            "the enflame backend (onehot tile DSM budget)."
        )

    fp64 = self.dtype == torch.float64
    kernel_input, kernel_weight, has_weight = _histogram_common_args(self, weight, fp64)
    if fp64:
        kernel_edges = edges.to(torch.float32)
    else:
        kernel_edges = edges
    n_elements = kernel_input.numel()

    nbins_padded = triton.next_power_of_2(nbins + 1)
    block = _pick_block(nbins_padded, has_weight, ONEHOT_MAX_LANES_DOUBLE_TILE)
    grid = max(1, min(triton.cdiv(n_elements, block), NUM_SIPS * 2))
    acc_dtype = torch.float32 if has_weight else torch.int32
    partials = torch.empty((grid, nbins_padded), dtype=acc_dtype, device=self.device)

    with torch_device_fn.device(self.device):
        histogram_bins_tensor_kernel[(grid,)](
            kernel_input,
            kernel_weight,
            kernel_edges,
            kernel_edges,
            partials,
            n_elements,
            nbins,
            HAS_WEIGHT=has_weight,
            PAIR_EDGES=False,
            NBINS=nbins_padded,
            BLOCK=block,
            num_warps=4,
        )

    hist = _histogram_launch_reduce(
        partials, kernel_edges, nbins, nbins_padded, grid, density, self.device
    )

    if fp64:
        return hist.to(self.dtype), edges.clone()
    return hist, edges.clone()
