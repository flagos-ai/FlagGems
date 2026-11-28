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
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# namedtuple class construction takes ~87us and would dominate small-shape kernel time per call, so reuse one at module
# level.
MaxOut = namedtuple("max", ["values", "indices"])


@libentry()
@triton.jit
def max_kernel_1_simple(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    tl.store(mid + pid, tl.max(inp_val))


@libentry()
@triton.jit(do_not_specialize=["M"])
def max_kernel_1_grid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    dtype = inp.type.element_ty
    min_val = get_dtype_min(dtype)

    acc = tl.full([BLOCK_SIZE], value=min_val, dtype=dtype)

    block_id = pid
    while block_id * BLOCK_SIZE < M:
        offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        vals = tl.load(inp + offset, mask=mask, other=min_val)
        acc = tl.where(vals > acc, vals, acc)
        block_id += num_progs

    tl.store(mid + pid, tl.max(acc))


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def max_kernel_inner_1d(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        # max only compares and selects values, so keeping f16/bf16 instead
        # of widening to f32 is bit-exact.  It also avoids a predicate that is
        # computed through an f16 -> f32 widening and then drives the i32 index
        # select, which GCU400 miscompiles: on (4096, 256) dim=1 the value is
        # right but the index points at a non-maximum element for ~46% of the
        # rows.  fp32 input, which is not widened, is unaffected.
        acc_dtype = dtype
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    for row_id in tl.range(pid, M, step, num_stages=num_stages):
        max_vals = tl.full([BLOCK_N], value=get_dtype_min(acc_dtype), dtype=acc_dtype)
        max_idxs = tl.zeros([BLOCK_N], dtype=tl.int32)

        base = row_id * N
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            mask = n_offset < N
            x = tl.load(inp + base + n_offset, mask=mask, other=min_value).to(acc_dtype)

            update = x > max_vals
            max_vals = tl.where(update, x, max_vals)
            max_idxs = tl.where(update, n_offset, max_idxs)

        final_max = tl.max(max_vals, axis=0)
        eq_mask = max_vals == final_max
        max_idxs_masked = tl.where(eq_mask, max_idxs, 2147483647)
        final_idx = tl.min(max_idxs_masked, axis=0)

        tl.store(out_value + row_id, final_max.to(dtype))
        tl.store(out_index + row_id, final_idx)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K", "num_k_tiles"])
def max_kernel_non_inner(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    num_k_tiles,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    total_work = M * num_k_tiles

    dtype = inp.type.element_ty
    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        # See max_kernel_inner_1d: bit-exact for max, and avoids the predicate
        # computed through an f16 -> f32 widening that GCU400 miscompiles.
        acc_dtype = dtype
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype
    min_value = get_dtype_min(acc_dtype)
    orig_min = get_dtype_min(dtype)

    for work_id in tl.range(pid, total_work, num_prog, num_stages=num_stages):
        m = work_id // num_k_tiles
        k_tile = work_id % num_k_tiles
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        max_vals = tl.full([BLOCK_K], dtype=acc_dtype, value=min_value)
        max_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

        base = m * N * K
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            offset = base + n_idx[:, None] * K + k_offset[None, :]
            mask = (n_idx[:, None] < N) & k_mask[None, :]
            vals = tl.load(inp + offset, mask=mask, other=orig_min).to(acc_dtype)

            # tl.max promotes f16/bf16 to f32; cast back so the index select
            # below still sees an f16 predicate.  The result is one of the
            # inputs, so this is exact.
            local_max = tl.max(vals, axis=0).to(acc_dtype)
            eq = vals == local_max[None, :]
            idx_matrix = tl.where(eq, n_idx[:, None], 2147483647)
            local_argmax = tl.min(idx_matrix, axis=0)

            update = local_max > max_vals
            max_vals = tl.where(update, local_max, max_vals)
            max_idxs = tl.where(update, local_argmax, max_idxs)

        out_offset = m * K + k_offset
        tl.store(out_value + out_offset, max_vals.to(dtype), k_mask)
        tl.store(out_index + out_offset, max_idxs, k_mask)


def max(inp):
    logger.debug("GEMS_ENFLAME MAX")

    if torch.is_grad_enabled() and inp.requires_grad:
        return MaxFunction.apply(inp)

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M <= 10 * 1024 * 1024:
        bsize = 16384
        num_programs = min(triton.cdiv(M, bsize), 48)
        block_mid = triton.next_power_of_2(num_programs)
        mid = torch.empty((num_programs,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            max_kernel_1_grid[(num_programs,)](
                inp, mid, M, BLOCK_SIZE=bsize, num_warps=1
            )
            max_kernel_2[(1,)](mid, out, num_programs, block_mid, num_warps=1)
    else:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            max_kernel_1_simple[(mid_size, 1, 1)](inp, mid, M, block_size, num_warps=1)
            max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, num_warps=1)
    return out


# gcu400 miscompiles a reduction whose lane vector reaches 512 B: the reduce
# silently covers only part of the axis (the value becomes the max over a
# subset and the index matches that wrong value).  Measured on
# max_kernel_inner_1d, (4096, 256) dim=1: BLOCK_N=128 with fp32 and
# BLOCK_N=256 with fp16/bf16 are wrong for ~94% of the rows, while 64 fp32
# lanes and 128 fp16 lanes are exact.  Cap the reduction width at 256 B.
_MAX_REDUCTION_BYTES = 256


def _reduction_cap(inp):
    # NB: the module-level ``max`` op shadows the builtin, so no max() here.
    cap = _MAX_REDUCTION_BYTES // inp.element_size()
    return cap if cap > 16 else 16


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME MAX_DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if torch.is_grad_enabled() and inp.requires_grad:
        values, indices = MaxDimFunction.apply(inp, dim, keepdim)
        return MaxOut(values=values, indices=indices)

    shape = inp.shape
    dim = dim % inp.ndim

    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    if not inp.is_contiguous():
        inp = inp.contiguous()

    shape_list = list(shape)
    if keepdim:
        shape_list[dim] = 1
    else:
        # Both kernels write back in a flat m*K+k layout, so allocate the final shape directly,
        # avoiding device sync and Python overhead from two torch.squeeze calls.
        shape_list.pop(dim)
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    # GCU400 Triton int32 indices store only 4 bytes per lane; writing directly into an int64
    # output tensor yields many zeros, so use an int32 buffer for kernel writes and convert to int64 once.
    out_index = torch.empty(shape_list, dtype=torch.int32, device=inp.device)

    if K == 1:
        BLOCK_N = min(triton.next_power_of_2(N), _reduction_cap(inp))
        grid_m = min(M, 48)
        num_stages = 3 if M > grid_m else 1
        with torch_device_fn.device(inp.device):
            max_kernel_inner_1d[(grid_m,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=1,
            )
    else:
        # BLOCK_K is the width of the loop-carried accumulator (output
        # columns; the reduction walks BLOCK_N).  GCU400 keeps such a vector
        # in at most 512B of OACC: 128 int64 lanes need 1024B and the
        # reduction then silently covers only part of the axis -- the value is
        # the max over a subset and the index matches that wrong value.
        # int64 dim=0 on (4096, 256) gets 219/256 columns wrong, while int32
        # at the same 128 lanes (512B) is exact.  The wide tile is also ~12x
        # slower.
        _acc_bpes = {
            torch.float64: 8,
            torch.int64: 8,
            torch.float32: 4,
            torch.int32: 4,
            torch.int16: 4,
            torch.int8: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        BLOCK_K = min(
            triton.next_power_of_2(K), min(512 // _acc_bpes.get(inp.dtype, 8), 128)
        )
        BLOCK_N = min(triton.next_power_of_2(N), _reduction_cap(inp))

        num_k_tiles = triton.cdiv(K, BLOCK_K)
        total_work = M * num_k_tiles
        grid_size = min(total_work, 48)

        with torch_device_fn.device(inp.device):
            max_kernel_non_inner[(grid_size,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                K,
                num_k_tiles,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
                num_stages=1,
                # num_warps=1 miscompiles the second grid-stride iteration
                # (total_work > 48): the last work items return the max over a
                # subset (measured on (200, 2560, 3) dim=0 fp32: k = 6144..7678
                # wrong, max err 2.0).  4 warps is exact.
                num_warps=4,
            )

    return MaxOut(values=out_value, indices=out_index.to(torch.int64))


class MaxFunction(torch.autograd.Function):
    """Autograd wrapper for ``torch.max(inp)`` on GCU400.

    ``forward`` runs the backend reduction, ``backward`` the evenly distributed
    formula that ATen keeps in the unregistered C++ helper
    ``evenly_distribute_backward``. The derivative is first order only.
    """

    @staticmethod
    def forward(ctx, inp):
        inp = inp.contiguous()
        out = max(inp)
        ctx.save_for_backward(inp, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None
        inp, result = ctx.saved_tensors
        return evenly_distribute_backward(grad_output, inp, result)


class MaxDimFunction(torch.autograd.Function):
    """Autograd wrapper for ``torch.max(inp, dim, keepdim)`` on GCU400.

    ``indices`` is an integer output and is marked non-differentiable, matching
    the native contract. ``backward`` scatters the values gradient onto the
    argmax positions saved by the forward pass.
    """

    @staticmethod
    def forward(ctx, inp, dim, keepdim):
        inp = inp.contiguous()
        values, indices = max_dim(inp, dim, keepdim)
        ctx.save_for_backward(inp, values, indices)
        ctx.dim = dim % inp.ndim
        ctx.mark_non_differentiable(indices)
        return values, indices

    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        inp, values, indices = ctx.saved_tensors
        return max_backward(grad_values, inp, values, indices, ctx.dim), None, None


# GCU400 limits the generic implementation runs into: grid.x/grid.y stop at 65535
# and grid.z at 255 (a 512000-row last-dim slab or a 256-way split raises
# OutOfResources), and an int64 lane comparison never matches, which zeroed every
# gradient. The kernels below take an int32 index and cap the grid, striding
# inside the kernel instead of growing a grid axis.
_MAX_GRID = 65535
_INDEX_SAFE_LIMIT = 0x7FFFFFFF
_SHARE_BLOCK = 2048


@libentry()
@triton.jit
def max_backward_fill_kernel(
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    """Zero the gradient slab with plain streaming stores.

    GCU400 runs a masked select-store over the whole slab at roughly a quarter of
    the streaming write bandwidth (measured 211 GB/s vs 780 GB/s for the same
    shape), so the gradient is produced the way ATen does it: one full-speed
    zero fill, then a scatter of the (tiny) argmax positions.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    zero = tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty)
    for start in tl.range(pid * BLOCK, n_elements, num_pids * BLOCK):
        offsets = start + lane
        if ALIGNED:
            tl.store(out_ptr + offsets, zero)
        else:
            tl.store(out_ptr + offsets, zero, mask=offsets < n_elements)


@libentry()
@triton.jit
def max_backward_scatter_lastdim_kernel(
    grad_ptr,
    index_ptr,
    out_ptr,
    reduction,
    rows,
    BLOCK: tl.constexpr,
):
    """Scatter the values gradient onto the argmax of each row (innermost dim).

    One lane per row, so the whole launch moves ``rows`` elements regardless of
    how wide the reduction is.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    for start in tl.range(pid * BLOCK, rows, num_pids * BLOCK):
        row = start + lane
        keep = row < rows
        index = tl.load(index_ptr + row, mask=keep, other=0)
        grad = tl.load(grad_ptr + row, mask=keep, other=0.0)
        tl.store(out_ptr + row * reduction + index, grad, mask=keep)


@libentry()
@triton.jit
def max_backward_scatter_tile_kernel(
    grad_ptr,
    index_ptr,
    out_ptr,
    outer,
    reduction,
    inner,
    BLOCK: tl.constexpr,
):
    """Scatter the values gradient when the reduced dim is interior.

    One lane per ``(plane, inner)`` position: ``out[plane, index, j] = grad[plane,
    j]``. The plane walk is strided inside the kernel so the grid stays inside the
    axis limit.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    for plane in tl.range(pid, outer, num_pids):
        base = plane * inner
        target = out_ptr + plane * (reduction * inner)
        for start in tl.range(0, inner, BLOCK):
            column = start + lane
            keep = column < inner
            index = tl.load(index_ptr + base + column, mask=keep, other=0)
            grad = tl.load(grad_ptr + base + column, mask=keep, other=0.0)
            tl.store(target + index * inner + column, grad, mask=keep)


@libentry()
@triton.jit
def evenly_distribute_count_kernel(
    inp_ptr,
    result_ptr,
    count_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Count the positions holding the maximum (a NaN maximum matches every NaN)."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    result = tl.load(result_ptr)
    nan_result = result != result
    for start in tl.range(pid * BLOCK, n_elements, num_pids * BLOCK):
        offsets = start + lane
        mask = offsets < n_elements
        values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        hits = ((values == result) | ((values != values) & nan_result)) & mask
        tl.atomic_add(count_ptr, tl.sum(hits.to(tl.int32), axis=0))


@libentry()
@triton.jit
def evenly_distribute_fill_kernel(
    inp_ptr,
    grad_ptr,
    result_ptr,
    count_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Write one equal share of the upstream gradient on every matching position."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    result = tl.load(result_ptr)
    nan_result = result != result
    count = tl.load(count_ptr).to(tl.int32)
    # The quotient is evaluated in float32, which keeps an even split exact for
    # the 16-bit and 32-bit dtypes that GCU400 supports natively.
    share = tl.load(grad_ptr).to(tl.float32) / count.to(tl.float32)
    zero = tl.zeros_like(share)
    for start in tl.range(pid * BLOCK, n_elements, num_pids * BLOCK):
        offsets = start + lane
        mask = offsets < n_elements
        values = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        match = ((values == result) | ((values != values) & nan_result)) & mask
        tl.store(
            out_ptr + offsets,
            tl.where(match, share, zero).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _backward_geometry(inp, dim):
    """Return ``(outer, reduction, inner)`` of ``dim`` in ``inp``'s layout."""
    ndim = inp.ndim
    dim = dim + ndim if dim < 0 else dim
    assert 0 <= dim < ndim, f"Invalid dim {dim} for a {ndim}-D tensor"
    shape = inp.shape
    outer = math.prod(shape[:dim])
    reduction = shape[dim]
    inner = math.prod(shape[dim + 1 :])
    return outer, reduction, inner


def _fill_config(n_elements, itemsize):
    """Streaming tile for the zero fill, tuned per element size on GCU400.

    A 2-byte stream needs the widest tile (131072 lanes) to keep the write
    engines busy, while a 4-byte stream peaks at 32768 lanes with two warps.
    """
    if itemsize == 2:
        if n_elements > (1 << 21):
            return 131072, 4
        return 16384, 4
    if n_elements > (1 << 21):
        return 32768, 2
    return 16384, 4


def max_backward(grad_output, self, result, indices, dim):
    """Gradient of ``torch.max(self, dim, keepdim)`` with respect to ``self``.

    ``grad_output`` holds the upstream gradient of the values output (already
    squeezed when the forward used ``keepdim=True``) and ``indices`` the int64
    argmax positions saved by the forward pass, so the result is exactly ATen's
    ``zeros_like(self).scatter_(dim, indices, grad_output)``: ties follow the
    single index the forward selected and a negative ``dim`` is normalized.
    ``result`` belongs to the ATen signature but the backward math never needs it.

    Inputs above the int32 offset range, and float64 inputs, keep the generic
    implementation; those shapes are beyond what GCU400 runs anyway.
    """
    logger.debug("GEMS_ENFLAME MAX_BACKWARD")
    outer, reduction, inner = _backward_geometry(self, dim)
    if self.numel() == 0:
        return torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if self.numel() > _INDEX_SAFE_LIMIT or self.dtype == torch.float64:
        from flag_gems.ops.max import max_backward as generic_max_backward

        return generic_max_backward(grad_output, self, result, indices, dim)

    grad = grad_output.contiguous()
    # The argmax is int64, but GCU400 mis-narrows an int64 lane vector inside the
    # kernel (measured: [0, 1, 5, 7] came back as [0, 5, <garbage>, <garbage>]),
    # so the index is narrowed on the host instead. The index tensor holds
    # outer * inner elements, far fewer than the slab the kernel writes.
    index = indices.contiguous().to(torch.int32)
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    n_elements = out.numel()
    with torch_device_fn.device(self.device):
        block, num_warps = _fill_config(n_elements, out.element_size())
        fill_grid = min(triton.cdiv(n_elements, block), _MAX_GRID)
        max_backward_fill_kernel[(fill_grid,)](
            out,
            n_elements,
            BLOCK=block,
            ALIGNED=n_elements % block == 0,
            num_warps=num_warps,
        )
        if inner == 1:
            rows = outer
            scatter_block = 1024
            scatter_grid = min(triton.cdiv(rows, scatter_block), _MAX_GRID)
            max_backward_scatter_lastdim_kernel[(scatter_grid,)](
                grad,
                index,
                out,
                reduction,
                rows,
                BLOCK=scatter_block,
            )
        else:
            scatter_block = triton.next_power_of_2(inner)
            if scatter_block > 1024:
                scatter_block = 1024
            scatter_grid = min(outer, _MAX_GRID)
            max_backward_scatter_tile_kernel[(scatter_grid,)](
                grad,
                index,
                out,
                outer,
                reduction,
                inner,
                BLOCK=scatter_block,
            )
    return out


def evenly_distribute_backward(grad_output, self, result):
    """Gradient of a full reduction such as ``torch.max(self)``.

    Every position holding the maximum receives an equal share of the upstream
    gradient; ties and (when the maximum is NaN) every NaN are matched with
    ATen's ``(self == result) | (self.isnan() & result.isnan())`` predicate, so
    the count is exactly the number of matched elements. The count is accumulated
    with an atomic add, which keeps both launches on a capped grid.
    """
    logger.debug("GEMS_ENFLAME EVENLY_DISTRIBUTE_BACKWARD")
    inp = self if self.is_contiguous() else self.contiguous()
    n_elements = inp.numel()
    out = torch.empty_like(inp)
    if n_elements == 0:
        return out
    if n_elements > _INDEX_SAFE_LIMIT or inp.dtype == torch.float64:
        from flag_gems.ops.max import evenly_distribute_backward as generic_evenly

        return generic_evenly(grad_output, self, result)

    grad = grad_output.contiguous().reshape(())
    value = result.contiguous().reshape(())
    count = torch.zeros((1,), dtype=torch.int32, device=inp.device)
    grid = min(triton.cdiv(n_elements, _SHARE_BLOCK), _MAX_GRID)
    with torch_device_fn.device(inp.device):
        evenly_distribute_count_kernel[(grid,)](
            inp, value, count, n_elements, BLOCK=_SHARE_BLOCK
        )
        evenly_distribute_fill_kernel[(grid,)](
            inp, grad, value, count, out, n_elements, BLOCK=_SHARE_BLOCK
        )
    return out
