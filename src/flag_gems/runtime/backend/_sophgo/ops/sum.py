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

# sophgo grid-cap for the stage-1 full reduction: instead of launching
# ceil(sqrt(M)) tiny programs (~1024 for M=1M), launch at most MAX_GRID
# programs that grid-stride over the tiles and accumulate in-register, so
# stage-2 only reduces <=MAX_GRID partials. Same insight as the pointwise
# overrides (fewer, fatter programs win on this TPU).
MAX_GRID = 64
RED_BLOCK = 4096


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    TPB: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    nprog = tle.num_programs(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=cdtype)
    for t in range(TPB):
        offset = (pid + t * nprog) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        acc += tl.load(inp + offset, mask=mask, other=0).to(cdtype)
    tl.store(mid + pid, tl.sum(acc))


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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
    share="naive_reduction",
)
@triton.jit
def sum_kernel(
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
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
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


def sum(inp, *, dtype=None):
    logger.debug("GEMS SUM")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    num_tiles = triton.cdiv(M, RED_BLOCK)
    grid = min(num_tiles, MAX_GRID)
    tpb = triton.cdiv(num_tiles, grid)
    mid_size = grid
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[(grid, 1, 1)](inp, mid, M, RED_BLOCK, tpb)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS SUM_OUT")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    num_tiles = triton.cdiv(M, RED_BLOCK)
    grid = min(num_tiles, MAX_GRID)
    tpb = triton.cdiv(num_tiles, grid)
    mid_size = grid
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        sum_kernel_1[(grid, 1, 1)](inp, mid, M, RED_BLOCK, tpb)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS SUM DIM")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS SUM_DIM_OUT")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N)
    if not keepdim:
        out.squeeze_(dim=dim)
    return out
