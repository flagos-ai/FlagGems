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


def _sum_acc_dtype(dtype):
    """Sum output / Accumulation dtype: Consistent with native PyTorch, bool and integer types with width less than 64 
    are promoted to int64 (int64/uint64 remain unchanged), floating-point types remain unchanged."""
    if dtype in (torch.bool, torch.int8, torch.uint8, torch.int16, torch.uint16, torch.int32, torch.uint32):
        return torch.int64
    return dtype


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

    pid = ext.program_id(0)
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

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def sum(inp, *, dtype=None):
    logger.debug("GEMS SUM")
    inp = inp.contiguous()
    M = inp.numel()

    if M == 0:
        if dtype is None:
            dtype = inp.dtype
        out = torch.zeros([], dtype=dtype, device=inp.device)
        return out

    if dtype is None:
        dtype = _sum_acc_dtype(inp.dtype)
        if dtype != inp.dtype:
            # Integer Promotion: Elevate the input to the accumulation precision, so that the kernel accumulates in int64, which is consistent with the native semantics
            inp = inp.to(dtype)
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS SUM_OUT")
    M = inp.numel()
    if dtype is None:
        dtype = _sum_acc_dtype(inp.dtype)
        if dtype != inp.dtype:
            inp = inp.to(dtype)
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
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

    pid_m = ext.program_id(0) + ext.program_id(1) * ext.num_programs(0)
    pid_k = ext.program_id(2)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]
    row_mask = pid_m < M

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = row_mask & (n_offsets < N) & (k_offsets < K)
        input_ptrs = input_ptr + inp_offset
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=row_mask & (k_offsets < K))
    else:
        sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = row_mask & (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum += inp
        out = tl.sum(sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=row_mask & (k_offsets < K))


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

    pid_m = ext.program_id(0) + ext.program_id(1) * ext.num_programs(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + inp_offset
        mask = (pid_m < M) & (n_offsets < N)
        inp = tl.load(input_ptrs, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=pid_m < M)
    else:
        sum = tl.zeros(
            [
                TILE_N,
            ],
            dtype=cdtype,
        )
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = (pid_m < M) & (n_offsets < N)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=0).to(cdtype)
            sum += inp
        out = tl.sum(sum, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=pid_m < M)


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

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    sum = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, sum, row_mask)


def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = _sum_acc_dtype(inp.dtype)
        if dtype != inp.dtype:
            # Integer Promotion: Elevate the input to the accumulation precision, so the kernel accumulates in int64, which is consistent with the native semantics
            inp = inp.to(dtype)

    # A scalar accepts axes 0/-1, but has no physical dimension to index.
    # Normalize and validate axes before using them in shape arithmetic.
    if dim is not None:
        dim = [dim] if isinstance(dim, int) else list(dim)
        rank = max(inp.ndim, 1)
        normalized_dim = []
        for d in dim:
            if d < -rank or d >= rank:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[{-rank}, {rank - 1}], but got {d})"
                )
            axis = d % rank
            if axis in normalized_dim:
                raise RuntimeError(
                    f"dim {axis} appears multiple times in the list of dims"
                )
            normalized_dim.append(axis)
        dim = normalized_dim

    # Handle both public overloads here. In particular, out= must never launch
    # a reduction on empty storage (K may be zero even when M and N are not).
    if inp.ndim == 0 or inp.numel() == 0:
        dims = list(range(inp.ndim)) if not dim else dim
        if keepdim:
            out_shape = [1 if d in dims else size for d, size in enumerate(inp.shape)]
        else:
            out_shape = [size for d, size in enumerate(inp.shape) if d not in dims]
        if out is None:
            out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        else:
            out.resize_(out_shape)
        if out.numel() == 0:
            return out
        if inp.numel() == 0:
            return zero_(out)
        # A nonempty rank-zero tensor contains exactly one value. Keep its
        # output rank zero for both keepdim settings and apply dtype on store.
        with torch_device_fn.device(inp.device):
            sum_dim_kernel_inner[(1, 1, 1)](out, inp, 1, 1)
        return out

    if dim is None:
        result = torch.sum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    if dim == [] or dim == ():
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)

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
                # HIP limits gridDim * blockDim to less than 2**32 per axis.
                # Split large row counts over X/Y (K tiles move to Z) rather
                # than overflowing X — same treatment as sum_dim_kernel_inner.
                grid_x = min(M, 65535)
                grid = lambda meta: (
                    grid_x,
                    triton.cdiv(M, grid_x),
                    triton.cdiv(K, meta["TILE_K"]),
                )
                sum_dim_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                # HIP limits gridDim * blockDim to less than 2**32 per axis.
                # Split large row counts over X/Y rather than overflowing X.
                grid_x = min(M, 65535)
                grid = (grid_x, triton.cdiv(M, grid_x), 1)
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

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            sum_dim_kernel[grid](inp, out, M, N)
        if not keepdim and not _out_provided:
            for d in sorted(dim, reverse=True):
                out = out.squeeze(dim=d)
        return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS SUM_DIM")
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS SUM_DIM_OUT")
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype, out=out)
