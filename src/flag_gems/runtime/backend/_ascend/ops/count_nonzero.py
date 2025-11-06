import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int32)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_kernel(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    n_workers = tle.num_programs(0)
    pid = tle.program_id(0)

    n_tasks = tl.cdiv(numel, N)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    for task_index in range(tasks_per_worker):
        task_id = pid + task_index * n_workers
        nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
        for start_n in range(0, N, BLOCK_SIZE):
            cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
            offset = task_id * N + cols_offsets
            mask = offset < numel and cols_offsets < N
            x = tl.load(x_ptr + offset, mask=mask, other=0)
            is_nonzero = (x != 0).to(tl.int64)
            nonzero_count += tl.sum(is_nonzero)

        tl.store(out_ptr + task_id, nonzero_count)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.jit
def count_nonzero_combin_kernel_1(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = tle.program_id(0)
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
    pid_x = tle.program_id(0)
    pid_y = tle.program_id(1)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = offset < numel and cols_offsets < N
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


def count_nonzero(x, dim=None):
    logger.debug("GEMS_ASCEND COUNT NONZERO")
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = x.shape
        BLOCK_SIZE = 8192
        numel = x.numel()
        x = dim_compress(x, dim)
        x = x.contiguous().flatten()
        combin_shape = list(shape)
        combin_shape[dim] = triton.cdiv(combin_shape[dim], BLOCK_SIZE)
        if combin_shape[dim] != 1:
            combin = torch.zeros(combin_shape, dtype=torch.int64, device=x.device)
            grid = (triton.cdiv(numel, shape[dim]), combin_shape[dim], 1)
            count_nonzero_combin_kernel[grid](
                x, combin, shape[dim], combin_shape[dim], numel, BLOCK_SIZE
            )
            x = combin
            shape = x.shape
            numel = x.numel()
            out_shape = list(shape)
            del out_shape[dim]
            out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)
            grid = lambda meta: (triton.cdiv(numel, shape[dim]),)
            count_nonzero_combin_kernel_1[grid](x, out, shape[dim], numel)
            return out
        out_shape = list(shape)
        del out_shape[dim]
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)

        def grid(meta):
            axis0 = triton.cdiv(numel, shape[dim])
            axis0 = axis0 if axis0 < 240 else 240
            return (axis0,)

        count_nonzero_kernel[grid](x, out, shape[dim], numel)
        return out
    else:
        x = x.contiguous().flatten()
        numel = x.numel()

        out = torch.zeros(1, dtype=torch.int32, device=x.device)

        BLOCK_SIZE = 8192
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

        count_nonzero_kernel_1[grid](x, out, numel, BLOCK_SIZE=BLOCK_SIZE)

        return out[0].to(torch.int64)
