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


@libentry()
@triton.jit
def histogramdd_from_bin_tensors_kernel(
    in_ptr,
    weight_ptr,
    out_ptr,
    edges_ptr,  # concatenated bin edges, one contiguous buffer
    edge_offsets_ptr,  # int64 (D,): start of each dimension's edges
    bin_counts_ptr,  # int64 (D,): edge count per dimension
    out_strides_ptr,  # int64 (D,): row-major output stride per dimension
    M,
    EDGE_CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D_CONST: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    NUM_SLABS: tl.constexpr,
):
    """Compute a multi-dimensional histogram from explicit bin-edge tensors.

    Each program processes ``BLOCK_SIZE`` N-dimensional points (rows of the
    flattened ``input``).  For every coordinate of a point the bin index is
    found with the left-inclusive / rightmost-right-inclusive rule:

        bin = (number of edges <= coordinate) - 1

    A point contributes to the histogram only when *every* coordinate falls
    into a valid bin (``0 <= bin_count < bin_count``).  The contribution is the
    point's weight (or ``1`` when no weight tensor is provided) and is
    accumulated with a relaxed atomic add into the N-dimensional output.

    The per-dimension geometry arrives as small device tensors rather than
    constexpr tuples: gcu400's launcher generator only understands scalar
    signature entries and aborts with ``KeyError: ('i32', 'constexpr')`` on a
    tuple-typed argument (every launch failed before this change).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    valid = mask
    out_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for d in tl.static_range(D_CONST):
        bin_count_d = tl.load(bin_counts_ptr + D_CONST + d)
        stride_d = tl.load(out_strides_ptr + 2 * D_CONST + d)
        edge_base = tl.load(edge_offsets_ptr + d)

        # Coordinate of this point along dimension d (innermost dim is D).
        #
        # Kept in ACC_DTYPE rather than narrowed to fp32: bin placement compares
        # the coordinate against the edges exactly, so narrowing an fp64 input
        # moves points across edges wholesale. Measured on 30 fp64 values probed
        # around the 11 edges of linspace(0, 1, 11), narrowing the edges alone
        # turns the reference counts [3,3,3,3,3,3,3,3,3,4] into
        # [5,3,3,3,1,5,0,6,0,5].
        coord = tl.load(in_ptr + offsets * D_CONST + d, mask=mask, other=0.0).to(
            ACC_DTYPE
        )
        # Load (padded) edges for this dimension and count how many are <= the
        # coordinate. The comparison is a (points x edges) broadcast, which
        # exceeds Triton's maximum tensor numel (2**20) as soon as a dimension
        # has ~1024 bins, so the edge axis is walked in EDGE_CHUNK-wide slabs
        # and the counts are accumulated. Padding edges are +inf so they never
        # satisfy ``edges <= coord`` and do not affect le_count.
        le_count = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        # slab 数量由 host 侧定为 constexpr，循环完全展开，消除动态边界。
        for slab_index in tl.static_range(NUM_SLABS):
            e_offs = slab_index * EDGE_CHUNK + tl.arange(0, EDGE_CHUNK)
            e_mask = e_offs < bin_count_d
            edges = tl.load(
                edges_ptr + edge_base + e_offs, mask=e_mask, other=float("inf")
            ).to(ACC_DTYPE)
            le_count += tl.sum((edges[None, :] <= coord[:, None]).to(tl.int64), axis=1)
        # Each bin i covers [edges[i], edges[i+1]); the rightmost bin also
        # includes its right edge.  For a coordinate in [edges[i], edges[i+1])
        # exactly i+1 edges are <= coord, so the bin index is le_count - 1.
        # When coord == edges[-1] (the top edge), every edge is <= coord so
        # le_count == n_edges and le_count - 1 == n_edges - 1 == bin_count,
        # which is one past the last bin; clamp it into the last (rightmost)
        # bin, which is inclusive of that edge.
        bin_idx = le_count - 1
        top_edge = tl.load(edges_ptr + edge_base + (bin_count_d - 1)).to(ACC_DTYPE)
        at_top = coord == top_edge
        # ``bin_count_d`` is the number of edges, so the last bin index is
        # ``bin_count_d - 2``.  Clamp the rightmost bin (inclusive of its right
        # edge) there: a coordinate equal to the top edge has le_count equal to
        # the edge count, giving bin_idx == bin_count_d - 1 (one past the last
        # bin), which must be pulled back into the last bin.
        bin_idx = tl.where(at_top, bin_count_d - 2, bin_idx)
        # A point is in range on this dim when coord >= edges[0] (le_count >= 1,
        # i.e. bin_idx >= 0) and coord <= edges[-1].  The latter is required
        # because a coord strictly above the top edge still yields
        # le_count == n_edges and (after clamping) bin_idx == bin_count_d - 2,
        # but must be dropped rather than counted there.
        in_range = (bin_idx >= 0) & (bin_idx < (bin_count_d - 1)) & (coord <= top_edge)
        valid = valid & in_range
        out_index = out_index + bin_idx.to(tl.int64) * stride_d

    if HAS_WEIGHT:
        contrib = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(ACC_DTYPE)
    else:
        contrib = tl.full((BLOCK_SIZE,), 1.0, dtype=ACC_DTYPE)
    tl.atomic_add(out_ptr + out_index, contrib, mask=valid, sem="relaxed")


def _validate(self, bins, weight):
    """Validate inputs with ATen's error types and wording.

    Uses exceptions rather than ``assert`` so the checks survive ``python -O`` and
    so callers see the same messages the native op produces.
    """
    D = len(bins)
    if self.ndim < 2:
        raise RuntimeError(
            "torch.histogramdd: input tensor should have at least 2 dimensions"
        )
    if self.shape[-1] != D:
        raise RuntimeError(
            f"torch.histogramdd: expected {self.shape[-1]} sequences of bin edges "
            f"for a {self.shape[-1]}-dimensional histogram but got {D}"
        )
    if not self.is_floating_point():
        raise RuntimeError(
            "torch.histogramdd: input tensor and bins tensors should have the same "
            "dtype, but got input with dtype " + str(self.dtype)
        )
    for d, b in enumerate(bins):
        if b.dtype != self.dtype:
            raise RuntimeError(
                "torch.histogramdd: input tensor and bins tensors should have the "
                f"same dtype, but got input {self.dtype} and bins[{d}] {b.dtype}"
            )
        if b.numel() < 2:
            raise RuntimeError(
                f"torch.histogram(): bins must be > 0, but got {b.numel() - 1} "
                f"for dimension {d}"
            )
    if weight is not None:
        # ATen requires an exact dtype match and rejects anything else; the
        # previous ``pass`` branch silently accepted a mismatched weight.
        if weight.dtype != self.dtype:
            raise RuntimeError(
                "torch.histogramdd: if weight tensor is provided, input tensor and "
                "weight tensor should have the same dtype, but got "
                f"input({self.dtype}), and weight({weight.dtype})"
            )
        if weight.numel() != self.numel() // self.shape[-1]:
            raise RuntimeError(
                "torch.histogramdd: if weight tensor is provided it should have "
                "the same shape as the input tensor excluding its innermost "
                "dimension"
            )


def _run_histogram(self, bins, weight, density, out):
    """Shared driver for the base and ``.out`` variants.

    Both variants previously duplicated this logic, which is how the ``.out``
    path came to skip the weight-dtype check and to accumulate differently.
    """
    _validate(self, bins, weight)
    D = len(bins)

    if self.dtype == torch.float64:
        # gcu400 has no fp64 arithmetic: the Triton pipeline cannot lower the
        # fp64 comparisons/atomics this kernel needs (the compiler crashes), and
        # computing in fp32 moves values that sit near a bin edge across it.
        # Double precision is therefore handled by the native CPU implementation
        # and copied back; the inputs are small point clouds in practice.
        cpu = torch.ops.aten._histogramdd_from_bin_tensors(
            self.cpu(),
            [b.cpu() for b in bins],
            weight=None if weight is None else weight.cpu(),
            density=density,
        )
        if out is None:
            return cpu.to(self.device)
        if out.dtype != self.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {self.dtype}, "
                f"but got {out.dtype} instead"
            )
        if tuple(out.shape) != tuple(cpu.shape):
            out.resize_(cpu.shape)
        out.copy_(cpu)
        return out

    inp = self.reshape(-1, D).contiguous()
    M = inp.shape[0]

    # Edges keep the input dtype; see the note in the kernel about narrowing.
    edges = [b.contiguous() for b in bins]
    bin_counts = [b.numel() for b in edges]
    out_shape = tuple(bc - 1 for bc in bin_counts)
    out_dtype = self.dtype

    # fp16/bf16 accumulate in fp32: tl.atomic_add reduces in a different order
    # than the reference's serial loop, so a low-precision accumulator would
    # drift beyond the reference's own rounding error.
    acc_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    acc_triton = tl.float64 if acc_dtype == torch.float64 else tl.float32

    has_weight = weight is not None
    if has_weight:
        w = weight.reshape(-1).contiguous()
        if w.dtype != acc_dtype:
            w = w.to(acc_dtype)
        weight_ptr = w
    else:
        weight_ptr = inp  # unused by the kernel (HAS_WEIGHT=False)

    # Validate / resize the caller's output before computing, matching ATen: a
    # mis-shaped output is resized in place and the same object is returned, but a
    # dtype or device mismatch raises rather than being silently re-bound.
    if out is not None:
        if out.dtype != out_dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {out_dtype}, but got "
                f"{out.dtype} instead"
            )
        if out.device != self.device:
            raise RuntimeError(
                f"Expected out tensor to have device {self.device}, but got "
                f"{out.device} instead"
            )
        if tuple(out.shape) != out_shape:
            out.resize_(out_shape)

    # Accumulate into the output itself when its dtype and layout allow it; a
    # non-contiguous or low-precision output needs a dense temporary because the
    # kernel writes through a flat stride computation.
    if out is not None and out.dtype == acc_dtype and out.is_contiguous():
        acc = out.zero_()
    else:
        acc = torch.zeros(out_shape, dtype=acc_dtype, device=self.device)

    # One program per point-tile; 256 points amortizes launch overhead while
    # keeping occupancy high for typical point-cloud sizes.
    BLOCK_SIZE = 256
    # The per-dimension edge comparison is a (points x edges) broadcast, and a
    # tensor may hold at most 2**20 elements, so the edge axis has to be
    # walked in slabs for any dimension with more than a few hundred bins. A
    # wider slab would shrink the loop for large bin counts but risk exceeding
    # the limit for the smaller blocks that do not need chunking at all, so the
    # slab is capped at the largest power of two that always fits.
    # gcu400 also caps on-chip (DSM) memory at 917504 bytes, and the
    # (BLOCK_SIZE x EDGE_CHUNK) comparison tile needs ~8 bytes per element:
    # BLOCK_SIZE=256 with a 1024-wide slab asks for 2.1 MiB and fails to
    # compile ("out of resource: dsm memory"). 256 x 256 stays around 540 KiB.
    # 依据实际 edge 数选择 2 的幂宽比较块：固定 256 宽会让 11 条 edge
    # 产生 245 个纯 padding lane；同时受 DSM 917504B（每元素约 8B）
    # 与 Triton 单张量 2**20 元素上限约束。
    max_edge_count = max(bin_counts)
    adaptive_chunk = triton.next_power_of_2(max_edge_count)
    dsm_budget_chunk = 1 << ((917504 // (8 * BLOCK_SIZE)).bit_length() - 1)
    EDGE_CHUNK = min(256, adaptive_chunk, dsm_budget_chunk, 2**20 // BLOCK_SIZE)
    grid = (triton.cdiv(M, BLOCK_SIZE),)

    if M > 0:
        # Per-dimension geometry is packed into device tensors: gcu400's
        # launcher generator rejects tuple-typed arguments outright.
        edges_flat = flag_gems.cat([b.contiguous() for b in edges], dim=0)
        edge_offsets = [0] * D
        for d in range(1, D):
            edge_offsets[d] = edge_offsets[d - 1] + bin_counts[d - 1]
        # edge 偏移 / edge 数 / 输出 stride 合并为一次 H2D 拷贝。
        geometry_t = torch.tensor(
            [edge_offsets, bin_counts, list(acc.stride())],
            dtype=torch.int64,
            device=edges_flat.device,
        )
        histogramdd_from_bin_tensors_kernel[grid](
            inp,
            weight_ptr,
            acc,
            edges_flat,
            geometry_t,
            geometry_t,
            geometry_t,
            M,
            EDGE_CHUNK=EDGE_CHUNK,
            BLOCK_SIZE=BLOCK_SIZE,
            D_CONST=D,
            HAS_WEIGHT=has_weight,
            ACC_DTYPE=acc_triton,
            NUM_SLABS=triton.cdiv(max_edge_count, EDGE_CHUNK),
        )

    if density:
        acc = _apply_density(acc, edges, D, out_shape, acc_dtype, acc_triton)

    if out is None:
        return acc if acc.dtype == out_dtype else acc.to(out_dtype)
    if acc.data_ptr() != out.data_ptr():
        out.copy_(acc)
    return out


def _histogramdd_from_bin_tensors(self, bins, *, weight=None, density=False):
    """FlagGems implementation of ``aten::_histogramdd_from_bin_tensors``.

    Args:
        self (Tensor): input with innermost dimension ``D``; all leading
            dimensions are flattened to ``M`` rows of ``D``-dimensional points.
        bins (sequence of Tensor): ``D`` strictly-increasing 1D edge tensors,
            with ``len(bins[d]) - 1`` bins per dimension.
        weight (Tensor, optional): weights of shape ``self.shape[:-1]``.
        density (bool): if True, normalise by total weight and bin volume.

    Returns:
        Tensor: the histogram, shaped ``(len(bins[0])-1, ..., len(bins[D-1])-1)``.
    """
    logger.debug("GEMS _HISTOGRAMDD_FROM_BIN_TENSORS")
    return _run_histogram(self, bins, weight, density, None)


def _histogramdd_from_bin_tensors_out(
    self, bins, *, weight=None, density=False, out=None
):
    """Out variant of :func:`_histogramdd_from_bin_tensors`."""
    logger.debug("GEMS _HISTOGRAMDD_FROM_BIN_TENSORS_OUT")
    if out is None:
        raise RuntimeError(
            "_histogramdd_from_bin_tensors.out: 'out' argument is required"
        )
    return _run_histogram(self, bins, weight, density, out)


@libentry()
@triton.jit
def histogramdd_density_kernel(
    hist_ptr,
    total_ptr,  # () total weight, may be zero or negative
    vol_ptr,  # (num_bins,) per-bin volume
    num_bins,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """hist[i] /= total * volume[i], with no guard on the total.

    ATen divides through unconditionally, so an empty or fully out-of-range input
    yields NaN, weights that cancel yield +-inf, and a net-negative total keeps
    the sign of each bin. Skipping the division when total <= 0 diverges from all
    three (measured: net-negative weights give [1.0, -0.0, -0.0, 3.0], not the
    unnormalised counts).
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < num_bins
    h = tl.load(hist_ptr + offs, mask=m, other=0.0).to(ACC_DTYPE)
    vol = tl.load(vol_ptr + offs, mask=m, other=1.0).to(ACC_DTYPE)
    total = tl.load(total_ptr).to(ACC_DTYPE)
    tl.store(hist_ptr + offs, h / total / vol, mask=m)


@libentry()
@triton.jit
def histogramdd_total_kernel(
    hist_ptr,
    total_ptr,  # () out
    num_bins,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum the histogram on device, replacing a host ``hist.sum().item()``."""
    total = tl.zeros((), dtype=ACC_DTYPE)
    for base in range(0, num_bins, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        m = offs < num_bins
        total += tl.sum(tl.load(hist_ptr + offs, mask=m, other=0.0).to(ACC_DTYPE))
    tl.store(total_ptr, total)


@libentry()
@triton.jit
def histogramdd_volume_kernel(
    edges_ptr,  # concatenated per-dimension edges
    edge_offsets_ptr,  # int64 (D,): start of each dimension's edges
    out_strides_ptr,  # int64 (D,): row-major strides
    bin_sizes_ptr,  # int64 (D,): bins per dimension
    vol_ptr,  # (num_bins,) out
    num_bins,
    D_CONST: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-bin volume: the product of the per-dimension edge widths.

    Computed on device so the host does not need the width subtraction, the
    reshape/broadcast_to and the running product that the convention disallows.
    Each flat bin index is decomposed back into per-dimension indices using the
    row-major strides.  Geometry arrives as device tensors because gcu400's
    launcher generator rejects tuple-typed constexpr arguments.
    """
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < num_bins
    vol = tl.full((BLOCK_SIZE,), 1.0, dtype=ACC_DTYPE)
    for d in tl.static_range(D_CONST):
        stride_d = tl.load(out_strides_ptr + d)
        size_d = tl.load(bin_sizes_ptr + d)
        edge_base = tl.load(edge_offsets_ptr + d)
        idx_d = (offs // stride_d) % size_d
        lo = tl.load(edges_ptr + edge_base + idx_d, mask=m, other=0.0).to(ACC_DTYPE)
        hi = tl.load(edges_ptr + edge_base + idx_d + 1, mask=m, other=1.0).to(ACC_DTYPE)
        vol = vol * (hi - lo)
    tl.store(vol_ptr + offs, vol, mask=m)


def _apply_density(hist, edges, D, out_shape, acc_dtype, acc_triton):
    """Normalise the histogram by total weight and per-bin volume, on device."""
    num_bins = hist.numel()
    if num_bins == 0:
        return hist
    dev = hist.device
    total = torch.empty((), dtype=acc_dtype, device=dev)
    vol = torch.empty(num_bins, dtype=acc_dtype, device=dev)

    histogramdd_total_kernel[(1,)](
        hist, total, num_bins, ACC_DTYPE=acc_triton, BLOCK_SIZE=1024
    )
    BLOCK = 1024
    grid = (triton.cdiv(num_bins, BLOCK),)
    # Strides of the contiguous histogram, in elements.
    contig_strides = []
    acc = 1
    for size in reversed(out_shape):
        contig_strides.append(acc)
        acc *= size
    contig_strides = tuple(reversed(contig_strides))

    bin_counts = [b.numel() for b in edges]
    edges_flat = flag_gems.cat([b.contiguous() for b in edges], dim=0)
    edge_offsets = [0] * D
    for d in range(1, D):
        edge_offsets[d] = edge_offsets[d - 1] + bin_counts[d - 1]
    histogramdd_volume_kernel[grid](
        edges_flat,
        torch.tensor(edge_offsets, dtype=torch.int64, device=dev),
        torch.tensor(contig_strides, dtype=torch.int64, device=dev),
        torch.tensor(out_shape, dtype=torch.int64, device=dev),
        vol,
        num_bins,
        D_CONST=D,
        ACC_DTYPE=acc_triton,
        BLOCK_SIZE=BLOCK,
    )
    flat = hist if hist.is_contiguous() else hist.contiguous()
    histogramdd_density_kernel[grid](
        flat.view(num_bins),
        total,
        vol,
        num_bins,
        ACC_DTYPE=acc_triton,
        BLOCK_SIZE=BLOCK,
    )
    if flat.data_ptr() != hist.data_ptr():
        hist.copy_(flat)
    return hist
