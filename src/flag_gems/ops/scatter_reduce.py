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

"""Triton implementation of torch.scatter_reduce for FlagGems.

Supports all reduce modes: sum, prod, mean, amax, amin.
Inputs above five dimensions are canonicalized to an equivalent 3D problem
before entering the fixed-width coordinate decoders.

Vendor compatibility:
  - NVIDIA: native atomic_max/min for amax/amin reduce
  - Iluvatar: CAS-based fallback for atomic_max/min (no native support)
  - Metax: larger BLOCK=256 for better occupancy

Performance notes:
  - Sum/mean use tl.atomic_add with relaxed semantics for throughput
  - Prod uses a bitwise CAS loop (no tl.atomic_mul exists)
  - Vendor backends can reuse the deterministic output-centric prod scan
    when their CAS lowering is not reliable under contention
  - All offset arithmetic uses int64 to avoid overflow for N > 2^31
  - LOOP=4: each program processes LOOP*BLOCK elements to amortize launch overhead
  - 2D fast path: specialized kernels for 2D tensors avoid 5D coordinate decoding
"""

import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_CANONICALIZE_5D_MIN_ELEMENTS = 1 << 23
# At this size, reducing 5D extrema through the 3D decoder amortizes the view
# setup and avoids the fixed-width coordinate overhead. Cross-backend probes on
# all supported floating dtypes showed a win or parity at 2**23 elements; keep
# smaller tensors and other reductions on their existing tuned paths.


def heur_block(args):
    """Vendor-aware block size heuristic.

    Metax and Iluvatar GPUs benefit from larger blocks (256) for better
    occupancy. NVIDIA GPUs default to 128 which balances occupancy and
    register pressure.
    """
    if flag_gems.vendor_name in ["metax", "iluvatar"]:
        return 256
    return 128


def heur_loop(args):
    """Loop unrolling factor.

    Each program processes LOOP*BLOCK elements to amortize kernel launch
    overhead. LOOP=4 is optimal for Iluvatar BI-V150.
    """
    return 4


def heur_prod_block(args):
    # The lock path uses one lane per program so same-address acquisition is
    # serialized by the device-wide atomic unit rather than by vector lanes.
    if args["USE_LOCK"]:
        return 1
    return heur_block(args)


def _prod_grid(N, max_grid_x=None):
    def grid(meta):
        programs = triton.cdiv(N, meta["BLOCK"] * meta["LOOP"])
        if max_grid_x is not None:
            grid_x = min(programs, max_grid_x)
            return (grid_x, triton.cdiv(programs, grid_x))
        return (programs,)

    return grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad5(lst, fill):
    """Pad a list to exactly 5 elements from the left with `fill`.

    This enables uniform 5D coordinate decoding in kernels regardless
    of the actual tensor dimensionality (1D-5D). Shapes are padded with 1,
    strides with 0.
    """
    return [fill] * (5 - len(lst)) + lst if len(lst) < 5 else lst


def _needs_cas_fallback():
    """Check if the current vendor needs CAS-based fallback for atomic_max/min.

    Iluvatar GPUs lack native tl.atomic_max/min, so we fall back to a
    CAS (Compare-And-Swap) loop for amax/amin reduce modes.
    """
    return flag_gems.vendor_name in ["iluvatar"]


def _scan_block(args):
    return 128


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit
def _locked_multiply(
    out_ptr,
    lock_ptr,
    out_offsets,
    src_value,
    mask,
    out_numel,
    BLOCK: tl.constexpr,
):
    """Multiply through an int32 lock when float bit-pattern CAS is unavailable."""
    pending = mask
    block_pending = tl.sum(pending.to(tl.int32)) > 0
    lock_zeros = tl.full((BLOCK,), 0, dtype=tl.int32)
    lock_ones = tl.full((BLOCK,), 1, dtype=tl.int32)
    while block_pending:
        # The extra lock is a safe target for inactive lanes on backends whose
        # atomic_cas implementation does not accept a mask argument.
        lock_offsets = tl.where(pending, out_offsets, out_numel)
        acquired = (
            tl.atomic_cas(
                lock_ptr + lock_offsets,
                lock_zeros,
                lock_ones,
                sem="acq_rel",
            )
            == 0
        ) & pending
        current = tl.load(out_ptr + out_offsets, mask=acquired, other=1.0).to(
            tl.float32
        )
        tl.store(out_ptr + out_offsets, current * src_value, mask=acquired)
        release_offsets = tl.where(acquired, out_offsets, out_numel)
        tl.atomic_cas(
            lock_ptr + release_offsets,
            lock_ones,
            lock_zeros,
            sem="acq_rel",
        )
        pending &= ~acquired
        block_pending = tl.sum(pending.to(tl.int32)) > 0


@libentry()
@triton.heuristics({"BLOCK": _scan_block})
@triton.jit(do_not_specialize=["out_numel"])
def scatter_reduce_prod_scan_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    out_numel,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    MATERIALIZE_PRODUCT: tl.constexpr,
    scan_shape: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    idx_shape_0,
    idx_shape_1,
    idx_shape_2,
    idx_shape_3,
    idx_shape_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_shape_0,
    out_shape_1,
    out_shape_2,
    out_shape_3,
    out_shape_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
):
    """Assign one program to each output and reduce matching source values."""
    pid = tl.program_id(axis=0).to(tl.int64)
    in_bounds = pid < out_numel

    remaining = pid
    coord0 = remaining // (out_shape_1 * out_shape_2 * out_shape_3 * out_shape_4)
    remaining %= out_shape_1 * out_shape_2 * out_shape_3 * out_shape_4
    coord1 = remaining // (out_shape_2 * out_shape_3 * out_shape_4)
    remaining %= out_shape_2 * out_shape_3 * out_shape_4
    coord2 = remaining // (out_shape_3 * out_shape_4)
    remaining %= out_shape_3 * out_shape_4
    coord3 = remaining // out_shape_4
    coord4 = remaining % out_shape_4

    out_offset = (
        coord0 * out_stride_0
        + coord1 * out_stride_1
        + coord2 * out_stride_2
        + coord3 * out_stride_3
        + coord4 * out_stride_4
    )
    idx_full_offset = (
        coord0 * idx_stride_0
        + coord1 * idx_stride_1
        + coord2 * idx_stride_2
        + coord3 * idx_stride_3
        + coord4 * idx_stride_4
    )
    src_full_offset = (
        coord0 * src_stride_0
        + coord1 * src_stride_1
        + coord2 * src_stride_2
        + coord3 * src_stride_3
        + coord4 * src_stride_4
    )

    if DIM == 0:
        target = coord0
        idx_base = idx_full_offset - coord0 * idx_stride_0
        src_base = src_full_offset - coord0 * src_stride_0
        idx_scan_stride = idx_stride_0
        src_scan_stride = src_stride_0
        valid_other = (
            (coord1 < idx_shape_1)
            & (coord2 < idx_shape_2)
            & (coord3 < idx_shape_3)
            & (coord4 < idx_shape_4)
            & (coord1 < src_shape_1)
            & (coord2 < src_shape_2)
            & (coord3 < src_shape_3)
            & (coord4 < src_shape_4)
        )
    elif DIM == 1:
        target = coord1
        idx_base = idx_full_offset - coord1 * idx_stride_1
        src_base = src_full_offset - coord1 * src_stride_1
        idx_scan_stride = idx_stride_1
        src_scan_stride = src_stride_1
        valid_other = (
            (coord0 < idx_shape_0)
            & (coord2 < idx_shape_2)
            & (coord3 < idx_shape_3)
            & (coord4 < idx_shape_4)
            & (coord0 < src_shape_0)
            & (coord2 < src_shape_2)
            & (coord3 < src_shape_3)
            & (coord4 < src_shape_4)
        )
    elif DIM == 2:
        target = coord2
        idx_base = idx_full_offset - coord2 * idx_stride_2
        src_base = src_full_offset - coord2 * src_stride_2
        idx_scan_stride = idx_stride_2
        src_scan_stride = src_stride_2
        valid_other = (
            (coord0 < idx_shape_0)
            & (coord1 < idx_shape_1)
            & (coord3 < idx_shape_3)
            & (coord4 < idx_shape_4)
            & (coord0 < src_shape_0)
            & (coord1 < src_shape_1)
            & (coord3 < src_shape_3)
            & (coord4 < src_shape_4)
        )
    elif DIM == 3:
        target = coord3
        idx_base = idx_full_offset - coord3 * idx_stride_3
        src_base = src_full_offset - coord3 * src_stride_3
        idx_scan_stride = idx_stride_3
        src_scan_stride = src_stride_3
        valid_other = (
            (coord0 < idx_shape_0)
            & (coord1 < idx_shape_1)
            & (coord2 < idx_shape_2)
            & (coord4 < idx_shape_4)
            & (coord0 < src_shape_0)
            & (coord1 < src_shape_1)
            & (coord2 < src_shape_2)
            & (coord4 < src_shape_4)
        )
    else:
        target = coord4
        idx_base = idx_full_offset - coord4 * idx_stride_4
        src_base = src_full_offset - coord4 * src_stride_4
        idx_scan_stride = idx_stride_4
        src_scan_stride = src_stride_4
        valid_other = (
            (coord0 < idx_shape_0)
            & (coord1 < idx_shape_1)
            & (coord2 < idx_shape_2)
            & (coord3 < idx_shape_3)
            & (coord0 < src_shape_0)
            & (coord1 < src_shape_1)
            & (coord2 < src_shape_2)
            & (coord3 < src_shape_3)
        )

    lanes = tl.arange(0, BLOCK)
    if MATERIALIZE_PRODUCT:
        # Some Ascend runtimes can replay a program. Writing the same factor
        # is idempotent, while updating the destination would apply it again.
        acc = 1.0
    else:
        acc = tl.load(out_ptr + out_offset, mask=in_bounds, other=1.0).to(tl.float32)
    has_contribution = False

    for start in range(0, scan_shape, BLOCK):
        scan = start + lanes
        valid = in_bounds & valid_other & (scan < scan_shape)
        index_value = tl.load(
            index_ptr + idx_base + scan * idx_scan_stride,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        match = valid & (index_value == target)
        src_value = tl.load(
            src_ptr + src_base + scan * src_scan_stride,
            mask=valid,
            other=1.0,
        ).to(tl.float32)
        factors = tl.where(match, src_value, 1.0)
        acc *= tl.reduce(factors, axis=0, combine_fn=_multiply)
        has_contribution |= tl.sum(match.to(tl.int32)) > 0

    tl.store(out_ptr + out_offset, acc, mask=in_bounds)
    if USE_MASK:
        tl.store(
            mask_ptr + out_offset,
            has_contribution.to(tl.int32),
            mask=in_bounds,
        )


def _scatter_reduce_prod_scan(
    inp,
    dim,
    index,
    src,
    include_self,
    *,
    materialize_product=False,
):
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim %= inp.ndim
    padded_dim = dim + (5 - inp.ndim)

    inp_f32 = inp.to(torch.float32).contiguous()
    if index.numel() == 0:
        return inp_f32.to(inp.dtype).clone()

    if materialize_product or not include_self:
        kernel_out = torch.ones_like(inp_f32)
    else:
        kernel_out = inp_f32.clone()

    src = src.contiguous()
    index = index.contiguous()
    idx_shapes = [int(value) for value in _pad5(list(index.shape), 1)]
    src_shapes = [int(value) for value in _pad5(list(src.shape), 1)]
    out_shapes = [int(value) for value in _pad5(list(kernel_out.shape), 1)]
    src_strides = [int(value) for value in _pad5(list(src.stride()), 0)]
    idx_strides = [int(value) for value in _pad5(list(index.stride()), 0)]
    out_strides = [int(value) for value in _pad5(list(kernel_out.stride()), 0)]

    use_mask = not include_self
    reduced_mask = torch.zeros(kernel_out.shape, dtype=torch.int32, device=inp.device)
    dummy_mask = torch.empty(1, dtype=torch.int32, device=inp.device)
    mask_ptr = reduced_mask if use_mask else dummy_mask

    with torch_device_fn.device(inp.device):
        scatter_reduce_prod_scan_kernel[(kernel_out.numel(),)](
            index,
            src,
            kernel_out,
            mask_ptr,
            kernel_out.numel(),
            padded_dim,
            use_mask,
            materialize_product,
            index.size(dim),
            src_strides[0],
            src_strides[1],
            src_strides[2],
            src_strides[3],
            src_strides[4],
            idx_shapes[0],
            idx_shapes[1],
            idx_shapes[2],
            idx_shapes[3],
            idx_shapes[4],
            src_shapes[0],
            src_shapes[1],
            src_shapes[2],
            src_shapes[3],
            src_shapes[4],
            idx_strides[0],
            idx_strides[1],
            idx_strides[2],
            idx_strides[3],
            idx_strides[4],
            out_shapes[0],
            out_shapes[1],
            out_shapes[2],
            out_shapes[3],
            out_shapes[4],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            out_strides[3],
            out_strides[4],
        )

    if materialize_product and include_self:
        out = inp_f32 * kernel_out
    elif use_mask:
        out = torch.where(reduced_mask == 0, inp_f32, kernel_out)
    else:
        out = kernel_out
    return out.to(inp.dtype)


def _scatter_reduce_high_rank(
    inp,
    dim,
    index,
    src,
    reduce,
    include_self,
    *,
    use_prod_scan=False,
    materialize_product=False,
    deduplicate_programs=False,
    use_product_lock=False,
    product_grid_limit=None,
):
    """Reduce an arbitrary-rank scatter through an equivalent 3D problem.

    ``index`` defines the active prefix domain. Only that domain is read from
    ``src`` and updated in ``inp``; values outside it remain unchanged. After
    narrowing those prefixes, every valid scatter can be represented as
    ``(outer, scatter_dim, inner)`` and handled by the existing kernels.
    """
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim %= inp.ndim

    index_shape = tuple(int(size) for size in index.shape)
    active_inp = inp
    active_src = src
    for axis, index_size in enumerate(index_shape):
        active_src = active_src.narrow(axis, 0, index_size)
        if axis != dim:
            active_inp = active_inp.narrow(axis, 0, index_size)

    outer = math.prod(index_shape[:dim])
    inner = math.prod(index_shape[dim + 1 :])
    active_shape = tuple(active_inp.shape)
    inp_3d = active_inp.contiguous().reshape(outer, inp.size(dim), inner)
    index_3d = index.contiguous().reshape(outer, index_shape[dim], inner)
    src_3d = active_src.contiguous().reshape(outer, index_shape[dim], inner)

    if use_prod_scan:
        active_result = _scatter_reduce_prod_scan(
            inp_3d,
            1,
            index_3d,
            src_3d,
            include_self,
            materialize_product=materialize_product,
        )
    else:
        active_result = scatter_reduce(
            inp_3d,
            1,
            index_3d,
            src_3d,
            reduce,
            include_self=include_self,
            _deduplicate_programs=deduplicate_programs,
            _use_product_lock=use_product_lock,
            _product_grid_limit=product_grid_limit,
        )

    active_domain_is_full = all(
        axis == dim or index_size == inp.size(axis)
        for axis, index_size in enumerate(index_shape)
    )
    if active_domain_is_full:
        return active_result.reshape(active_shape)

    result = inp.contiguous().clone()
    result_active = result
    for axis, index_size in enumerate(index_shape):
        if axis != dim:
            result_active = result_active.narrow(axis, 0, index_size)
    result_active.copy_(active_result.reshape(active_shape))
    return result


def _should_canonicalize_5d(inp, dim, index, src, reduce):
    if (
        inp.ndim != 5
        or index.ndim != inp.ndim
        or src.ndim != inp.ndim
        or reduce not in ("amax", "amin")
        or inp.numel() < _CANONICALIZE_5D_MIN_ELEMENTS
    ):
        return False
    return all(
        axis == dim or index.size(axis) == inp.size(axis) for axis in range(inp.ndim)
    )


# ---------------------------------------------------------------------------
# 2D Fast Path Kernels with LOOP
# Specialized for 2D tensors to avoid 5D coordinate decoding overhead.
# Uses 1D grid with LOOP=4 to amortize kernel launch overhead.
# ---------------------------------------------------------------------------


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_sum_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    claim_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CLAIM: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if USE_CLAIM:
        program_claimed = tl.atomic_cas(
            claim_ptr + pid,
            0,
            1,
            sem="acq_rel",
        )
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        if USE_CLAIM:
            mask &= program_claimed == 0

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)
        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_prod_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_prod_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    claim_ptr,
    lock_ptr,
    N,
    out_numel,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CLAIM: tl.constexpr,
    USE_LOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)
    if USE_CLAIM:
        program_claimed = tl.atomic_cas(
            claim_ptr + pid,
            0,
            1,
            sem="acq_rel",
        )
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        if USE_CLAIM:
            mask &= program_claimed == 0

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        if USE_LOCK:
            _locked_multiply(
                out_ptr,
                lock_ptr,
                out_offsets,
                src_val,
                mask,
                out_numel,
                BLOCK,
            )
        else:
            # CAS on the raw float32 bit pattern is portable across the
            # backends that select this path.
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            out_ptr_i32 = (out_ptr + out_offsets).to(
                tl.pointer_type(tl.int32, 1), bitcast=True
            )
            while not block_stop:
                cur_bits = tl.load(out_ptr_i32, mask=mask, other=0)
                cur_val = cur_bits.to(tl.float32, bitcast=True)
                new_val = tl.where(stop, cur_val, cur_val * src_val)
                new_bits = new_val.to(tl.int32, bitcast=True)
                cas_res = tl.atomic_cas(out_ptr_i32, cur_bits, new_bits, sem="acq_rel")
                stop |= cur_bits == cas_res
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_mean_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    count_ptr,
    mask_ptr,
    claim_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CLAIM: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if USE_CLAIM:
        program_claimed = tl.atomic_cas(
            claim_ptr + pid,
            0,
            1,
            sem="acq_rel",
        )
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        if USE_CLAIM:
            mask &= program_claimed == 0

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
        ones_f = tl.full((BLOCK,), 1.0, dtype=tl.float32)
        tl.atomic_add(count_ptr + out_offsets, ones_f, mask=mask, sem="relaxed")

        if USE_MASK:
            ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_amax_2d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    idx_ncols,
    src_ncols,
    out_ncols,
    DIM: tl.constexpr,
    IS_AMAX: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CAS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        row = offsets // idx_ncols
        col = offsets % idx_ncols

        if DIM == 0:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = idx * out_ncols + col
        else:
            idx_offsets = row * idx_ncols + col
            src_offsets = row * src_ncols + col
            idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
            out_offsets = row * out_ncols + idx

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur_val = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)
                if IS_AMAX:
                    new_val = tl.maximum(cur_val, src_val)
                else:
                    new_val = tl.minimum(cur_val, src_val)
                cas_res = tl.atomic_cas(
                    out_ptr + out_offsets, cur_val, new_val, sem="relaxed"
                )
                stop |= cur_val == cas_res
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if IS_AMAX:
                tl.atomic_max(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# Generic 5D Kernels with LOOP optimization
# For tensors with ndim != 2.
# ---------------------------------------------------------------------------


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_sum_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    claim_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CLAIM: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if USE_CLAIM:
        program_claimed = tl.atomic_cas(
            claim_ptr + pid,
            0,
            1,
            sem="acq_rel",
        )
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        if USE_CLAIM:
            mask &= program_claimed == 0

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)
        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_prod_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_prod_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    claim_ptr,
    lock_ptr,
    N,
    out_numel,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CLAIM: tl.constexpr,
    USE_LOCK: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)
    if USE_CLAIM:
        program_claimed = tl.atomic_cas(
            claim_ptr + pid,
            0,
            1,
            sem="acq_rel",
        )
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        if USE_CLAIM:
            mask &= program_claimed == 0

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        if USE_LOCK:
            _locked_multiply(
                out_ptr,
                lock_ptr,
                out_offsets,
                src_val,
                mask,
                out_numel,
                BLOCK,
            )
        else:
            # CAS on bits preserves NaNs and avoids floating-point CAS
            # differences between the backends that select this path.
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            out_ptr_i32 = (out_ptr + out_offsets).to(
                tl.pointer_type(tl.int32, 1), bitcast=True
            )
            while not block_stop:
                cur_bits = tl.load(out_ptr_i32, mask=mask, other=0)
                cur_val = cur_bits.to(tl.float32, bitcast=True)
                new_val = tl.where(stop, cur_val, cur_val * src_val)
                new_bits = new_val.to(tl.int32, bitcast=True)
                cas_res = tl.atomic_cas(out_ptr_i32, cur_bits, new_bits, sem="acq_rel")
                stop |= cur_bits == cas_res
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_mean_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    count_ptr,
    mask_ptr,
    claim_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CLAIM: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if USE_CLAIM:
        program_claimed = tl.atomic_cas(
            claim_ptr + pid,
            0,
            1,
            sem="acq_rel",
        )
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N
        if USE_CLAIM:
            mask &= program_claimed == 0

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        tl.atomic_add(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
        ones_f = tl.full((BLOCK,), 1.0, dtype=tl.float32)
        tl.atomic_add(count_ptr + out_offsets, ones_f, mask=mask, sem="relaxed")

        if USE_MASK:
            ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": heur_block, "LOOP": heur_loop})
@triton.jit(do_not_specialize=["N"])
def scatter_reduce_amax_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    mask_ptr,
    N,
    out_stride_dim,
    src_stride_dim,
    src_shape_dim,
    out_shape_dim,
    DIM: tl.constexpr,
    IS_AMAX: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_CAS: tl.constexpr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    src_stride_4,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    src_shape_3,
    src_shape_4,
    idx_stride_0,
    idx_stride_1,
    idx_stride_2,
    idx_stride_3,
    idx_stride_4,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = pid * BLOCK * LOOP + tl.arange(0, BLOCK)

    for i in range(LOOP):
        offsets = (base_offsets + i * BLOCK).to(tl.int64)
        mask = offsets < N

        remaining = offsets
        coord0 = remaining // (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_1 * src_shape_2 * src_shape_3 * src_shape_4)
        coord1 = remaining // (src_shape_2 * src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_2 * src_shape_3 * src_shape_4)
        coord2 = remaining // (src_shape_3 * src_shape_4)
        remaining = remaining % (src_shape_3 * src_shape_4)
        coord3 = remaining // src_shape_4
        coord4 = remaining % src_shape_4

        idx_offsets = (
            coord0 * idx_stride_0
            + coord1 * idx_stride_1
            + coord2 * idx_stride_2
            + coord3 * idx_stride_3
            + coord4 * idx_stride_4
        )
        src_offsets = (
            coord0 * src_stride_0
            + coord1 * src_stride_1
            + coord2 * src_stride_2
            + coord3 * src_stride_3
            + coord4 * src_stride_4
        )

        idx = tl.load(index_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)

        if DIM == 0:
            out_offsets = (
                idx * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 1:
            out_offsets = (
                coord0 * out_stride_0
                + idx * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 2:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + idx * out_stride_2
                + coord3 * out_stride_3
                + coord4 * out_stride_4
            )
        elif DIM == 3:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + idx * out_stride_3
                + coord4 * out_stride_4
            )
        else:
            out_offsets = (
                coord0 * out_stride_0
                + coord1 * out_stride_1
                + coord2 * out_stride_2
                + coord3 * out_stride_3
                + idx * out_stride_4
            )

        src_val = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur_val = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)
                if IS_AMAX:
                    new_val = tl.maximum(cur_val, src_val)
                else:
                    new_val = tl.minimum(cur_val, src_val)
                cas_res = tl.atomic_cas(
                    out_ptr + out_offsets, cur_val, new_val, sem="relaxed"
                )
                stop |= cur_val == cas_res
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if IS_AMAX:
                tl.atomic_max(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out_ptr + out_offsets, src_val, mask=mask, sem="relaxed")

        if USE_MASK:
            ones = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(mask_ptr + out_offsets, ones, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------


def scatter_reduce(
    inp,
    dim,
    index,
    src,
    reduce,
    *,
    include_self=True,
    _deduplicate_programs=False,
    _use_product_lock=False,
    _product_grid_limit=None,
):
    """Triton-accelerated scatter_reduce operation.

    Scatters src values into the output tensor at positions determined by index,
    applying the specified reduction. Supports sum, prod, mean, amax, amin.

    Args:
        inp: Input tensor.
        dim: Dimension along which to scatter.
        index: Index tensor mapping source elements to output positions.
        src: Source tensor containing values to scatter.
        reduce: Reduction mode - "sum", "prod", "mean", "amax", or "amin".
        include_self: If True, include inp values in the reduction.

    Returns:
        Output tensor with same shape and dtype as inp.
    """
    logger.debug("GEMS SCATTER_REDUCE_TWO")

    assert reduce in (
        "sum",
        "prod",
        "mean",
        "amax",
        "amin",
    ), f"Unsupported reduce: {reduce}"
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim %= inp.ndim

    if inp.ndim > 5 or _should_canonicalize_5d(inp, dim, index, src, reduce):
        return _scatter_reduce_high_rank(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self,
            deduplicate_programs=_deduplicate_programs,
            use_product_lock=_use_product_lock,
            product_grid_limit=_product_grid_limit,
        )

    padded_dim = dim + (5 - inp.ndim)

    out_stride_dim = inp.stride(dim)
    out_shape_dim = inp.size(dim)
    src_stride_dim = src.stride(dim)
    src_shape_dim = src.size(dim)
    N = index.numel()

    # Avoid double clone: merge contiguous + float32 cast
    inp_f32 = inp.to(torch.float32).contiguous()

    if include_self:
        out = inp_f32.clone()
    else:
        if reduce in ("sum", "mean"):
            out = torch.zeros_like(inp_f32)
        elif reduce == "prod":
            out = torch.ones_like(inp_f32)
        elif reduce == "amax":
            out = torch.full(
                inp_f32.shape,
                float("-inf"),
                dtype=inp_f32.dtype,
                device=inp_f32.device,
            )
        elif reduce == "amin":
            out = torch.full(
                inp_f32.shape,
                float("inf"),
                dtype=inp_f32.dtype,
                device=inp_f32.device,
            )

    if N == 0:
        return inp_f32.to(inp.dtype).clone()

    use_mask = not include_self
    if use_mask:
        reduced_mask = torch.zeros(out.shape, dtype=torch.int32, device=inp.device)

    if reduce == "mean":
        if include_self:
            count = torch.ones_like(out, dtype=torch.float32)
        else:
            count = torch.zeros_like(out, dtype=torch.float32)

    src = src.contiguous()
    index = index.contiguous()

    # Convert strides/shapes to int64 to avoid overflow in kernel arithmetic
    idx_shapes = [int(x) for x in _pad5(list(index.shape), 1)]
    src_strides_p = [int(x) for x in _pad5(list(src.stride()), 0)]
    idx_strides_p = [int(x) for x in _pad5(list(index.stride()), 0)]
    out_strides_p = [int(x) for x in _pad5(list(out.stride()), 0)]

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]),)
    prod_grid = _prod_grid(N, _product_grid_limit)

    dummy_mask = torch.empty(1, dtype=torch.int32, device=inp.device)
    mask_ptr = reduced_mask if use_mask else dummy_mask
    use_claim = _deduplicate_programs and reduce in ("sum", "prod", "mean")
    claim_ptr = (
        torch.zeros(N + 1, dtype=torch.int32, device=inp.device)
        if use_claim
        else dummy_mask
    )
    use_product_lock = _use_product_lock and reduce == "prod"
    lock_ptr = (
        torch.zeros(out.numel() + 1, dtype=torch.int32, device=inp.device)
        if use_product_lock
        else dummy_mask
    )

    # Use 2D fast path for 2D tensors (most common case)
    use_2d = inp.ndim == 2

    # For 2D kernels, use raw dim (0 or 1) instead of padded_dim
    dim_2d = dim

    with torch_device_fn.device(inp.device):
        if reduce == "sum":
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_sum_2d_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    claim_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    use_mask,
                    use_claim,
                )
            else:
                scatter_reduce_sum_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    claim_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    use_mask,
                    use_claim,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )
        elif reduce == "prod":
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_prod_2d_kernel[prod_grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    claim_ptr,
                    lock_ptr,
                    N,
                    out.numel(),
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    use_mask,
                    use_claim,
                    use_product_lock,
                )
            else:
                scatter_reduce_prod_kernel[prod_grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    claim_ptr,
                    lock_ptr,
                    N,
                    out.numel(),
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    use_mask,
                    use_claim,
                    use_product_lock,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )
        elif reduce == "mean":
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_mean_2d_kernel[grid](
                    index,
                    src,
                    out,
                    count,
                    mask_ptr,
                    claim_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    use_mask,
                    use_claim,
                )
            else:
                scatter_reduce_mean_kernel[grid](
                    index,
                    src,
                    out,
                    count,
                    mask_ptr,
                    claim_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    use_mask,
                    use_claim,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )
            has_contributions = count > 0
            count = torch.clamp(count, min=1.0)
            out = out / count
            out = torch.where(has_contributions, out, inp_f32)
        elif reduce in ("amax", "amin"):
            use_cas = _needs_cas_fallback()
            if use_2d:
                idx_ncols = index.shape[1]
                src_ncols = src.shape[1]
                out_ncols = out.shape[1]
                scatter_reduce_amax_2d_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    idx_ncols,
                    src_ncols,
                    out_ncols,
                    dim_2d,
                    reduce == "amax",
                    use_mask,
                    use_cas,
                )
            else:
                scatter_reduce_amax_kernel[grid](
                    index,
                    src,
                    out,
                    mask_ptr,
                    N,
                    out_stride_dim,
                    src_stride_dim,
                    src_shape_dim,
                    out_shape_dim,
                    padded_dim,
                    reduce == "amax",
                    use_mask,
                    use_cas,
                    src_strides_p[0],
                    src_strides_p[1],
                    src_strides_p[2],
                    src_strides_p[3],
                    src_strides_p[4],
                    idx_shapes[0],
                    idx_shapes[1],
                    idx_shapes[2],
                    idx_shapes[3],
                    idx_shapes[4],
                    idx_strides_p[0],
                    idx_strides_p[1],
                    idx_strides_p[2],
                    idx_strides_p[3],
                    idx_strides_p[4],
                    out_strides_p[0],
                    out_strides_p[1],
                    out_strides_p[2],
                    out_strides_p[3],
                    out_strides_p[4],
                )

    if use_mask and reduce != "mean":
        unreduced = reduced_mask == 0
        out = torch.where(unreduced, inp_f32, out)

    return out.to(inp.dtype)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    """In-place variant of scatter_reduce. Modifies inp in-place."""
    logger.debug("GEMS SCATTER_REDUCE_TWO_")

    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    """Out-variant of scatter_reduce. Writes result to out tensor if provided."""
    logger.debug("GEMS SCATTER_REDUCE_TWO_OUT")

    if out is not None and out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    if out is not None:
        out.copy_(result)
        return out
    return result
