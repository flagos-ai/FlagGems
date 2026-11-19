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
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Ascend's local buffer is smaller than the tile selected by the generic
# sqrt-based reduction heuristic for very large tensors. Keep each reduction
# tile bounded and let the second stage consume multiple tiles when needed.
_ASCEND_SUM_MAX_BLOCK_SIZE = 4096
_ASCEND_MAX_CORE_DIM = 65535


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    for pid in range(ext.program_id(0), tl.cdiv(M, BLOCK_SIZE), ext.num_programs(0)):
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inp_ptrs = inp + offset
        mask = offset < M

        inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
        sum_val = tl.sum(inp_val)
        mid_ptr = mid + pid
        tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    sum_val = tl.zeros((), dtype=cdtype)
    for start in range(0, mid_size, BLOCK_MID):
        offset = start + tl.arange(0, BLOCK_MID)
        mid_ptrs = mid + offset
        mask = offset < mid_size
        mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
        sum_val += tl.sum(mid_val)
    tl.store(out, sum_val)


def _reduction_block_size(numel):
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(numel)))
    return min(block_size, _ASCEND_SUM_MAX_BLOCK_SIZE)


def _reduction_mid_block_size(mid_size):
    return min(triton.next_power_of_2(mid_size), _ASCEND_SUM_MAX_BLOCK_SIZE)


def _reduction_grid_size(num_tiles):
    # Bound the total launch size, including flattened multidimensional grids.
    # Kernels traverse the remaining logical tiles with a grid-stride loop.
    return min(num_tiles, _ASCEND_MAX_CORE_DIM)


def _sum_dtype(inp, dtype, out=None):
    if out is not None:
        if dtype is not None and dtype != out.dtype:
            raise RuntimeError("sum: dtype must match out.dtype")
        return out.dtype
    if dtype is not None:
        return dtype
    if not inp.dtype.is_floating_point and not inp.dtype.is_complex:
        return torch.int64
    return inp.dtype


def sum(inp, *, dtype=None):
    logger.debug("GEMS_ASCEND SUM")
    dtype = _sum_dtype(inp, dtype)
    out = torch.empty([], dtype=dtype, device=inp.device)
    return sum_out(inp, dtype=dtype, out=out)


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_ASCEND SUM_OUT")
    dtype = _sum_dtype(inp, dtype, out)
    inp = inp.contiguous()
    M = inp.numel()
    out.resize_([])
    if M == 0:
        zero_(out)
        return out
    if inp.dtype is torch.bool:
        inp = inp.to(torch.int64)
    block_size = _reduction_block_size(M)
    mid_size = triton.cdiv(M, block_size)
    block_mid = _reduction_mid_block_size(mid_size)
    grid_size = _reduction_grid_size(mid_size)

    # Keep partial sums in FP32 so low-precision outputs do not round or
    # overflow before the second stage can combine cancelling partial sums.
    mid_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    mid = torch.empty((mid_size,), dtype=mid_dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        sum_kernel_1[(grid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def sum_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    # Ascend flattens grid dimensions into coreDim. Flatten explicitly before
    # capping the launch so even M * ceil(K / TILE_K) > 65535 stays valid.
    num_k_tiles = tl.cdiv(K, TILE_K)
    for pid in range(ext.program_id(0), M * num_k_tiles, ext.num_programs(0)):
        pid_m = pid // num_k_tiles
        pid_k = pid % num_k_tiles
        k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

        if ONE_TILE_PER_CTA:
            n_offsets = tl.arange(0, TILE_N)[:, None]
            inp_offset = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offset, mask=mask, other=0).to(cdtype)
            out = tl.sum(inp, axis=0, keep_dims=True)
        else:
            # Reset the accumulator for every logical output tile.
            sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)
            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
                inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
                mask = (n_offsets < N) & (k_offsets < K)
                inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
                sum += inp
            out = tl.sum(sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def sum_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    for pid_m in range(ext.program_id(0), M, ext.num_programs(0)):
        if ONE_TILE_PER_CTA:
            n_offsets = tl.arange(0, TILE_N)
            inp_offset = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offset, mask=mask, other=0).to(cdtype)
            out = tl.sum(inp, axis=0)
        else:
            sum = tl.zeros([TILE_N], dtype=cdtype)
            for start_n in range(0, N, TILE_N):
                n_offsets = start_n + tl.arange(0, TILE_N)
                inp_offsets = pid_m * N + n_offsets
                mask = n_offsets < N
                inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
                sum += inp
            out = tl.sum(sum, axis=0)
        tl.store(output_ptr + pid_m, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def sum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    for tile in range(ext.program_id(0), tl.cdiv(M, BLOCK_M), ext.num_programs(0)):
        pid = tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        row_mask = pid < M
        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask & col_mask

            a = tl.load(inp + pid * N + cols, mask, other=0).to(cdtype)
            _sum += a
        sum = tl.sum(_sum, axis=1)[:, None]
        tl.store(out + pid, sum, row_mask)


def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    dtype = _sum_dtype(inp, dtype, out)
    if dim is None or dim == [] or dim == ():
        if out is not None:
            sum_out(inp, dtype=dtype, out=out)
            if keepdim:
                out.resize_([1] * inp.ndim)
            return out
        result = sum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)
    dim = [dim] if isinstance(dim, int) else dim
    dim = [d % inp.ndim for d in dim]

    # Handle both empty reductions and empty outputs before dividing by M/N
    # or launching a reduction. Both public dim entry points share this guard.
    if inp.numel() == 0:
        dim_set = set(dim)
        out_shape = (
            [1 if i in dim_set else size for i, size in enumerate(shape)]
            if keepdim
            else [size for i, size in enumerate(shape) if i not in dim_set]
        )
        if out is None:
            out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        else:
            out.resize_(out_shape)
        if out.numel() > 0:
            zero_(out)
        return out

    if len(dim) == 1:
        dim = dim[0]
        N = inp.shape[dim]
        M = reduce(lambda x, y: x * y, shape[:dim], 1)
        inp = inp.contiguous()
        K = inp.numel() // M // N
        shape[dim] = 1
        _out_provided = out is not None
        if _out_provided:
            # Resize out to the expected output shape, matching native PyTorch
            # sum.out behavior. The caller (e.g. logsumexp) may pass a
            # zero-size placeholder that needs to be resized before use.
            if keepdim:
                out.resize_(shape)
            else:
                out.resize_(shape[:dim] + shape[dim + 1 :])
        else:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (
                    _reduction_grid_size(M * triton.cdiv(K, meta["TILE_K"])),
                )
                sum_dim_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                grid = (_reduction_grid_size(M),)
                sum_dim_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        if not keepdim and not _out_provided:
            out = out.squeeze(dim=dim)
        return out
    else:
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        _out_provided = out is not None
        if _out_provided:
            dim_set = set(dim)
            if keepdim:
                out.resize_(shape)
            else:
                out.resize_([s for i, s in enumerate(shape) if i not in dim_set])
        else:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (_reduction_grid_size(triton.cdiv(M, meta["BLOCK_M"])),)
        with torch_device_fn.device(inp.device):
            sum_dim_kernel[grid](inp, out, M, N)
        if not keepdim and not _out_provided:
            for d in sorted(dim, reverse=True):
                out = out.squeeze(dim=d)
        return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ASCEND SUM_DIM")
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_ASCEND SUM_DIM_OUT")
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype, out=out)
