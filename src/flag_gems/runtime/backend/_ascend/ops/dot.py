import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum = tl.sum(x * y)
    tl.store(out_ptr, sum)


@libentry()
@triton.jit
def dot_kernel_1(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    n_workers = tle.num_programs(0)
    pid = tle.program_id(0)

    n_tasks = tl.cdiv(N, BLOCK_SIZE)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)
    for task_index in range(tasks_per_worker):
        task_id = pid + task_index * n_workers

        block_start = task_id * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        partial_sum = tl.sum(x * y)
        tl.store(mid_ptr + task_id, partial_sum)


@libentry()
@triton.jit
def dot_kernel_2(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr):
    n_tasks = tl.cdiv(M, BLOCK_MID)
    for task_index in range(n_tasks):
        offset = task_index * BLOCK_MID + tl.arange(0, BLOCK_MID)
        mid = mid_ptr + offset
        mask = offset < M
        mid_val = tl.load(mid, mask=mask, other=0.0)
        out_val = tl.sum(mid_val)
        tl.store(out_ptr, out_val)


def dot(x, y):
    logger.debug("GEMS_ASCEND DOT")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]

    # Only when N is less than TRITON_MAX_TENSOR_NUMEL can it be processed with a single kernel,
    # and performance is better when N < 4096
    if N >= 4096:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(N)))
        block_size = 12032 if block_size > 12032 else block_size
        mid_size = triton.cdiv(N, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        block_mid = 16384 if block_mid > 16384 else block_mid

        grid_1 = (mid_size if mid_size < 240 else 240, 1, 1)
        grid_2 = (1, 1, 1)

        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel_1[grid_1](x, y, mid, N, block_size)
            dot_kernel_2[grid_2](mid, out, mid_size, block_mid)

    else:
        block_size = triton.next_power_of_2(N)

        grid = (1, 1, 1)

        out = torch.empty([], dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel[grid](x, y, out, N, block_size)
            out = out.to(x.dtype)

    return out
