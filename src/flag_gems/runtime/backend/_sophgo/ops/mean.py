import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from ..utils.shape_utils import dim_compress

logger = logging.getLogger(__name__)

# sophgo grid-cap for the stage-1 full reduction (see sum.py): <=MAX_GRID fat
# programs grid-stride over the tiles and accumulate partial SUMS in fp32;
# stage-2 sums the <=MAX_GRID partials and divides by M (unchanged), so the
# tiling does not affect the result.
MAX_GRID = 64
RED_BLOCK = 4096


@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    TPB: tl.constexpr,
):
    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for t in range(TPB):
        offset = (pid + t * nprog) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        acc += tl.load(inp + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(mid + pid, tl.sum(acc, axis=0))


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


def mean(inp, *, dtype=None):
    logger.debug("GEMS MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    num_tiles = triton.cdiv(M, RED_BLOCK)
    grid = min(num_tiles, MAX_GRID)
    tpb = triton.cdiv(num_tiles, grid)
    mid_size = grid
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(grid, 1, 1)](inp, mid, M, RED_BLOCK, tpb)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
    share="naive_reduction",
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)


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
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch_device_fn.device(x.device):
        mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out
