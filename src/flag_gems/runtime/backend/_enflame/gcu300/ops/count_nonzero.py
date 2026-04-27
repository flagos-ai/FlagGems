import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def count_nonzero_kernel_1(
    x_ptr, out_ptr, numel, ng, ng_per_block: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid = pid_x * ng_per_block
    sum = 0
    for start_ng in range(0, ng_per_block, 1):
        if pid < ng:
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            x = tl.load(x_ptr + offsets, mask=mask, other=0)
            is_nonzero = (x != 0).to(tl.int32)
            nonzero_count = tl.sum(is_nonzero, axis=0)
            sum = sum + nonzero_count
        pid = pid + 1

    tl.store(out_ptr + pid_x, sum)


@libentry()
@triton.jit
def count_nonzero_kernel_1_atomic_add(
    in_ptr, out_ptr, NP: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    sum = 0
    for start_n in range(0, NP, BLOCK_SIZE):
        offset = start_n + tl.arange(0, BLOCK_SIZE)
        mask = offset < NP
        x = tl.load(in_ptr + offset, mask=mask, other=0)
        sum += tl.sum(x)
    tl.store(out_ptr, sum)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_kernel(
    x_ptr, out_ptr, N, numel, ng, ng_per_block: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid = pid_x * ng_per_block
    for start_ng in range(0, ng_per_block, 1):
        if pid < ng:
            nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
            for start_n in range(0, N, BLOCK_SIZE):
                cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
                offset = pid * N + cols_offsets
                mask = offset < numel and cols_offsets < N
                x = tl.load(x_ptr + offset, mask=mask, other=0)
                is_nonzero = (x != 0).to(tl.int32)
                nonzero_count += tl.sum(is_nonzero)

            tl.store(out_ptr + pid, nonzero_count)
        pid = pid + 1


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_combin_kernel_1(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = offset < numel and cols_offsets < N
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        nonzero_count += tl.sum(x)
    tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@triton.jit
def count_nonzero_combin_kernel(
    x_ptr, combin_ptr, N, combin_N, numel, BLOCK_SIZE: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = offset < numel and cols_offsets < N
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int32)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


def count_nonzero(x, dim=None):
    logger.debug("GEMS COUNT NONZERO")
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = x.shape
        BLOCK_SIZE = 2048
        numel = x.numel()
        x = dim_compress(x, dim)
        x = x.contiguous().flatten()
        combin_shape = list(shape)
        combin_shape[dim] = triton.cdiv(combin_shape[dim], BLOCK_SIZE)
        if combin_shape[dim] != 1:
            combin = torch.zeros(combin_shape, dtype=torch.int32, device=x.device)

            grid = (triton.cdiv(numel, shape[dim]), combin_shape[dim], 1)

            count_nonzero_combin_kernel[grid](
                x, combin, shape[dim], combin_shape[dim], numel, BLOCK_SIZE
            )
            x = combin
            shape = x.shape
            numel = x.numel()
            out_shape = list(shape)
            del out_shape[dim]
            out = torch.zeros(out_shape, dtype=torch.int32, device=x.device)

            grid = lambda meta: (triton.cdiv(numel, shape[dim]),)

            count_nonzero_combin_kernel_1[grid](x, out, shape[dim], numel)
            out = out.to(torch.int64)
            return out
        out_shape = list(shape)
        del out_shape[dim]
        out = torch.zeros(out_shape, dtype=torch.int32, device=x.device)

        ng = triton.cdiv(numel, shape[dim])
        if ng > MAX_GRID_DIM:
            ng_per_block = triton.cdiv(ng, MAX_GRID_DIM)
            grid = (MAX_GRID_DIM,)
        else:
            ng_per_block = 1
            grid = (ng,)
        count_nonzero_kernel[grid](
            x, out, shape[dim], numel, ng, ng_per_block=ng_per_block
        )

        out = out.to(torch.int64)
        return out
    else:
        x = x.contiguous().flatten()
        numel = x.numel()

        BLOCK_SIZE = 1024
        ng = triton.cdiv(numel, BLOCK_SIZE)
        if ng > MAX_GRID_DIM:
            out_ng = MAX_GRID_DIM
            ng_per_block = triton.cdiv(ng, MAX_GRID_DIM)
            grid = (MAX_GRID_DIM,)
        else:
            out_ng = ng
            ng_per_block = 1
            grid = (ng,)

        out_t = torch.zeros(out_ng, dtype=torch.int32, device=x.device)

        count_nonzero_kernel_1[grid](
            x, out_t, numel, ng, ng_per_block=ng_per_block, BLOCK_SIZE=BLOCK_SIZE
        )

        out = torch.zeros(1, dtype=torch.int32, device=x.device)
        count_nonzero_kernel_1_atomic_add[(1,)](
            out_t, out, NP=out_ng, BLOCK_SIZE=BLOCK_SIZE
        )

        out = out.to(torch.int64)
        return out[0]
