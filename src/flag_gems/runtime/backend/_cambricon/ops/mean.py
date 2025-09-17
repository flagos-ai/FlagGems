import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner

from ..utils import TOTAL_CORE_NUM, cfggen_reduce_op

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@libtuner(
    configs=cfggen_reduce_op(), key=["M"], strategy=["log"], reset_to_zero=["out"]
)
@triton.jit
def mean_kernel_1(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    block_start = block_start.to(tl.int64)
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=0.0)
        _tmp = inp_val + _tmp

    mean_val = tl.sum(_tmp, axis=0) / M
    tl.atomic_add(out, mean_val)


def mean(inp, *, dtype=None):
    logger.debug("GEMS_CAMBRICON MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    out = torch.zeros([], dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[grid](inp, out, M)
    return out.to(dtype)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mean"),
    key=["M", "N"],
    strategy=["log", "log"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[
            :, None
        ]
        X_ptr = X + pid * N
        Mean_ptr = Mean + pid
        row_mask = pid < M

        # Compute mean
        _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _mean += a
        _mean /= N
        mean = tl.sum(_mean, axis=1)[:, None]
        tl.store(Mean_ptr, mean, row_mask)


def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS_CAMBRICON MEAN DIM")

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
    grid = lambda META: (min(triton.cdiv(M, META["BLOCK_M"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out
