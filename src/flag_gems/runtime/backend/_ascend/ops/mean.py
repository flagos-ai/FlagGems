import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(pid * BLOCK_SIZE, M, num_jobs * BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=0.0)
        _sum += inp_val.to(tl.float32)
    tl.store(mid + pid, tl.sum(_sum, axis=0))


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid + offset, mask=mask, other=0.0)
    mean_val = tl.sum(mid_val) / M
    tl.store(out, mean_val)


def mean(inp, *, dtype=None):
    logger.debug("GEMS MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = min(block_size, 1024)
    mid_size = triton.cdiv(M, block_size)
    mid_size = min(mid_size, 4096)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out.to(dtype)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mean"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    workers = tle.num_programs(0)
    pid = tle.program_id(0)
    total_workloads = tl.cdiv(M, BLOCK_M)
    workloads = tl.cdiv(total_workloads, workers)

    for w in range(workloads):
        work_id = pid + w * workers
        rows = work_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + rows * N
        Mean_ptr = Mean + rows
        row_mask = rows < M

        _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=1) / N
        mean = mean[:, None]
        tl.store(Mean_ptr, mean, row_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS MEAN DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)

    def grid(meta):
        axis0 = triton.cdiv(M, meta["BLOCK_M"])
        axis0 = axis0 if axis0 < 4096 else 4096
        return (axis0,)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out
