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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

MinResult = namedtuple("min", ["values", "indices"])


@libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


@libentry()
@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = ext.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    # you just cannot create a function that return a tl.dtype in triton lang
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmin = tl.argmin(inp_vals, 1)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def min(inp):
    if torch.is_grad_enabled() and inp.requires_grad and inp.is_floating_point():
        return MinFunction.apply(inp)
    logger.debug("GEMS MIN")
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        min_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        min_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def min_dim(inp, dim=None, keepdim=False):
    if torch.is_grad_enabled() and inp.requires_grad and inp.is_floating_point():
        values, indices = MinDimFunction.apply(inp, dim, keepdim)
        return MinResult(values, indices)
    logger.debug("GEMS MIN_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        min_kernel[grid](inp, out_value, out_index, M, N)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out


# ---------------------------------------------------------------------------
# Backward
#
# Neither `aten::min(Tensor)` nor `aten::min.dim` has an independently
# registered backward kernel: both adjoints come from the autograd engine
# (`evenly_distribute_backward` and `value_selecting_reduction_backward`
# respectively).  We therefore attach the adjoint with `torch.autograd.Function`
# and run it with Triton:
#
#   * global `min(x)` splits `grad_output` evenly between every entry that is
#     equal to the minimum, so the tie count is a global reduction that has to
#     run inside the backward;
#   * `min(x, dim)` scatters `grad_output` onto the single saved argmin index
#     (`indices` is a non-differentiable `int64` tensor), so the adjoint only
#     needs the reduced layout.
# ---------------------------------------------------------------------------


@libentry(cache_binding=True)
@triton.jit
def min_backward_global_count_kernel(
    inp,
    result,
    partials,
    n_elements,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INT64_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_OFFSET:
        pid = pid.to(tl.int64)
    min_value = tl.load(result)
    value_is_nan = min_value != min_value
    total = tl.zeros((1,), dtype=tl.int32)
    for start in range(pid * BLOCK_SIZE, n_elements, NUM_PROGRAMS * BLOCK_SIZE):
        offset = start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements
        value = tl.load(inp + offset, mask=mask, other=0)
        is_min = (value == min_value) | ((value != value) & value_is_nan)
        total += tl.sum((is_min & mask).to(tl.int32))
    # One partial sum per program: the buffer is freshly allocated, so it needs
    # no pre-zeroing and the whole global adjoint costs two launches.
    tl.store(partials + pid, tl.sum(total, axis=0))


@libentry(cache_binding=True)
@triton.jit
def min_backward_global_fill_kernel(
    inp,
    grad_output,
    result,
    partials,
    out,
    n_elements,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INT64_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_OFFSET:
        pid = pid.to(tl.int64)
    dtype = inp.type.element_ty
    acc_dtype = tl.float64 if dtype is tl.float64 else tl.float32
    min_value = tl.load(result)
    value_is_nan = min_value != min_value
    count = tl.sum(tl.load(partials + tl.arange(0, NUM_PARTIALS)).to(tl.int64), axis=0)
    scale = tl.load(grad_output).to(acc_dtype) / count.to(acc_dtype)
    for start in range(pid * BLOCK_SIZE, n_elements, tl.num_programs(0) * BLOCK_SIZE):
        offset = start + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements
        value = tl.load(inp + offset, mask=mask, other=0)
        is_min = (value == min_value) | ((value != value) & value_is_nan)
        grad_input = tl.where(is_min, scale, 0.0)
        tl.store(out + offset, grad_input.to(dtype), mask=mask)


@libentry(cache_binding=True)
@triton.jit
def min_backward_dim_kernel(
    grad_output,
    indices,
    out,
    n_rows,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INT64_OFFSET: tl.constexpr,
):
    # Tile counts are constexpr arithmetic: passing them as separate launch
    # arguments costs binding time on every call.
    n_tiles: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    k_tiles: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K

    pid = tl.program_id(0)
    if INT64_OFFSET:
        pid = pid.to(tl.int64)
    tile_n = pid % n_tiles
    rest = pid // n_tiles
    tile_k = rest % k_tiles
    first_row = (rest // k_tiles) * ROWS

    dim_offset = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner_offset = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    dim_mask = dim_offset < N
    inner_mask = inner_offset < K
    store_mask = dim_mask[:, None] & inner_mask[None, :]

    for step in range(ROWS):
        row = first_row + step
        # A row tile may overshoot the last row of the tensor; the guard has to
        # cover the loads as well as the store.
        row_mask = row < n_rows
        reduced_offset = row * K + inner_offset
        load_mask = inner_mask & row_mask
        index = tl.load(indices + reduced_offset, mask=load_mask, other=-1)
        grad = tl.load(grad_output + reduced_offset, mask=load_mask, other=0)
        selected = (dim_offset[:, None] == index[None, :]) & dim_mask[:, None]
        # A zero of the loaded dtype keeps the select/store pipeline in the input
        # dtype instead of promoting the tile to fp32: same bits, half the
        # register footprint.
        zero = tl.zeros([BLOCK_N, BLOCK_K], dtype=grad.dtype)
        grad_input = tl.where(selected, grad[None, :], zero)
        offset = row * (N * K) + dim_offset[:, None] * K + inner_offset[None, :]
        tl.store(out + offset, grad_input, mask=store_mask & row_mask)


# Launch-shape caches: the backward is host-bound at small sizes, so the tile
# arithmetic must not be recomputed on every call.
_GLOBAL_PLAN_CACHE = {}
_DIM_PLAN_CACHE = {}
_PLAN_CACHE_LIMIT = 64


def _global_plan(n_elements):
    plan = _GLOBAL_PLAN_CACHE.get(n_elements)
    if plan is None:
        count_block = 4096
        # `tl.arange` needs a power of two, and the count buffer is sized from
        # it, so round up; programs with an empty range just store 0.
        count_grid = triton.next_power_of_2(triton.cdiv(n_elements, count_block))
        # The public `min` below shadows the builtin inside this module.
        if count_grid > 1024:
            count_grid = 1024
        fill_block = 4096
        fill_grid = triton.cdiv(n_elements, fill_block)
        if fill_grid > 65535:
            fill_grid = 65535
        wide = n_elements >= 2147483648
        plan = (count_grid, count_block, fill_grid, fill_block, wide)
        if len(_GLOBAL_PLAN_CACHE) < _PLAN_CACHE_LIMIT:
            _GLOBAL_PLAN_CACHE[n_elements] = plan
    return plan


def _dim_launch_config(outer_size, n_reduce, inner_size):
    """Tile choice for the value-selecting adjoint.

    ``BLOCK_K`` is the contiguous run length of a single store: whenever the
    inner axis fits one program's tile (``inner_size <= 1024``) it is taken
    whole, so a program's ``BLOCK_N x BLOCK_K`` store is one fully contiguous
    slab instead of a strided set of ``BLOCK_K``-sized pieces.  For
    ``inner_size == 1`` this is the plain "one program per row tile" shape.

    A wide inner axis gets the larger 8192-element tile with eight warps: the
    store is then 16 KiB per program, which halves the program count for the
    multi-GiB cases where the store stream is the whole workload.
    """
    block_k = triton.next_power_of_2(inner_size)
    wide = block_k > 256
    if block_k > 1024:
        block_k = 1024
    elems = 8192 if wide else 2048
    max_block_n = triton.next_power_of_2(n_reduce)
    block_n = elems // block_k
    if block_n > max_block_n:
        block_n = max_block_n
    if block_n < 1:
        block_n = 1
    rows = elems // (block_n * block_k)
    if rows < 1:
        rows = 1
    n_tiles = triton.cdiv(n_reduce, block_n)
    k_tiles = triton.cdiv(inner_size, block_k)
    # Keep enough programs in flight for the very small tensors.
    while rows > 1 and triton.cdiv(outer_size, rows) * n_tiles * k_tiles < 264:
        rows //= 2
    grid = (triton.cdiv(outer_size, rows) * n_tiles * k_tiles,)
    num_warps = 8 if wide else 4
    return grid, rows, block_n, block_k, n_tiles, k_tiles, num_warps


def _dim_plan(shape, dim):
    plan = _DIM_PLAN_CACHE.get((shape, dim))
    if plan is not None:
        return plan
    n_reduce = shape[dim]
    inner_size = 1
    for size in shape[dim + 1 :]:
        inner_size *= size
    outer_size = 1
    for size in shape[:dim]:
        outer_size *= size
    grid, rows, block_n, block_k, n_tiles, k_tiles, num_warps = _dim_launch_config(
        outer_size, n_reduce, inner_size
    )
    wide = outer_size * n_reduce * inner_size >= 2147483648
    plan = (
        grid,
        rows,
        block_n,
        block_k,
        num_warps,
        outer_size,
        n_reduce,
        inner_size,
        wide,
    )
    if len(_DIM_PLAN_CACHE) < _PLAN_CACHE_LIMIT:
        _DIM_PLAN_CACHE[(shape, dim)] = plan
    return plan


def _min_backward_global(grad_output, inp, result):
    """Adjoint of the global reduction: `evenly_distribute_backward`."""
    n_elements = inp.numel()
    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if n_elements == 0:
        return out
    # The forward kernels index `inp` linearly, so the adjoint inherits the same
    # contiguous-input requirement.
    source = inp if inp.is_contiguous() else inp.contiguous()
    count_grid, count_block, fill_grid, fill_block, wide = _global_plan(n_elements)
    partials = torch.empty(count_grid, dtype=torch.int32, device=inp.device)
    with torch_device_fn.device(inp.device):
        min_backward_global_count_kernel[(count_grid,)](
            source,
            result,
            partials,
            n_elements,
            NUM_PROGRAMS=count_grid,
            BLOCK_SIZE=count_block,
            INT64_OFFSET=wide,
        )
        min_backward_global_fill_kernel[(fill_grid,)](
            source,
            grad_output,
            result,
            partials,
            out,
            n_elements,
            NUM_PARTIALS=count_grid,
            BLOCK_SIZE=fill_block,
            INT64_OFFSET=wide,
        )
    return out


def _min_backward_dim(grad_output, inp, indices, dim):
    """Adjoint of `min.dim`: scatter `grad_output` at the saved argmin index."""
    if dim < 0:
        dim += inp.ndim
    (
        grid,
        rows,
        block_n,
        block_k,
        num_warps,
        outer_size,
        n_reduce,
        inner_size,
        wide,
    ) = _dim_plan(inp.shape, dim)

    out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if out.numel() == 0:
        return out

    grad = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    index = indices if indices.is_contiguous() else indices.contiguous()
    # Positional constexpr arguments bind measurably faster than keywords.
    args = (
        grad,
        index,
        out,
        outer_size,
        n_reduce,
        inner_size,
        rows,
        block_n,
        block_k,
        wide,
    )
    with torch_device_fn.device(inp.device):
        if num_warps == 8:
            min_backward_dim_kernel[grid](*args, num_warps=num_warps)
        else:
            min_backward_dim_kernel[grid](*args)
    return out


class MinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        result = min(inp)
        ctx.save_for_backward(inp, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        inp, result = ctx.saved_tensors
        return _min_backward_global(grad_output, inp, result)


class MinDimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, keepdim):
        values, indices = min_dim(inp, dim, keepdim)
        ctx.save_for_backward(inp, indices)
        ctx.dim = dim
        ctx.mark_non_differentiable(indices)
        return values, indices

    @staticmethod
    def backward(ctx, grad_values, grad_indices):
        if grad_values is None:
            return None, None, None
        inp, indices = ctx.saved_tensors
        return _min_backward_dim(grad_values, inp, indices, ctx.dim), None, None


def min_backward(grad_output, inp, result, indices=None, dim=None):
    """Return the min gradient; the dim overload requires its saved indices."""
    if dim is None:
        return _min_backward_global(grad_output, inp, result)
    if indices is None:
        raise ValueError("min_backward requires indices when dim is specified")
    return _min_backward_dim(grad_output, inp, indices, dim)
