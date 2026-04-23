import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

_min = builtins.min
_max = builtins.max
MAX_GRID = 48


@libentry()
@triton.jit(do_not_specialize=["M"])
def sum_global_kernel_1(
    X,
    Mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog, num_stages=num_stages):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        x = tl.load(X + offset, mask=mask, other=0.0).to(cdtype)
        sum_val = tl.sum(x, axis=0)
        tl.store(Mid + tile_id, sum_val)


@libentry()
@triton.jit(do_not_specialize=["M"])
def sum_single_kernel(X, Out, M, BLOCK_SIZE: tl.constexpr):
    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    x = tl.load(X + offset, mask=mask, other=0.0).to(cdtype)
    sum_val = tl.sum(x)
    tl.store(Out, sum_val)


@libentry()
@triton.jit(do_not_specialize=["MID_SIZE"])
def sum_global_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    if tl.constexpr(Mid.dtype.element_ty == tl.float16) or tl.constexpr(
        Mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(Mid + offset, mask=mask, other=0.0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(Out, sum_val)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_dim_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    if tl.constexpr(X.dtype.element_ty == tl.float16) or tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid_m = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid_m, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            x = tl.load(X + offset, mask, other=0.0).to(cdtype)
            _sum += x

        total_sum = tl.sum(_sum, axis=1)
        tl.store(Out + m_offset, total_sum, row_mask)


def _launch_global_sum(inp, out, M):
    is_fp32 = inp.dtype == torch.float32
    max_single = 32768 if is_fp32 else 65536

    if M <= max_single:
        block_size = _max(1024, triton.next_power_of_2(M))
        with torch_device_fn.device(inp.device):
            sum_single_kernel[(1,)](inp, out, M, BLOCK_SIZE=block_size, num_warps=4)
        return

    block_size = 131072 if is_fp32 else 65536
    mid_size = triton.cdiv(M, block_size)
    grid_size = _min(mid_size, MAX_GRID)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty([mid_size], dtype=inp.dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_global_kernel_1[(grid_size, 1, 1)](
            inp, mid, M,
            BLOCK_SIZE=block_size,
            num_stages=1,
            num_warps=2,
        )
        sum_global_kernel_2[(1, 1, 1)](
            mid, out, mid_size, block_mid,
            num_warps=1,
        )


def _launch_dim_sum(inp, out, M, N):
    BLOCK_N = _min(triton.next_power_of_2(N), 2048)
    BLOCK_M = _max(1, _min(128, 65536 // BLOCK_N))
    grid_m = _min(triton.cdiv(M, BLOCK_M), MAX_GRID)

    with torch_device_fn.device(inp.device):
        sum_dim_kernel[(grid_m,)](
            inp, out, M, N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=1,
            num_warps=1,
        )


def sum(inp, *, dtype=None):
    logger.debug("GEMS SUM GCU400")
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    inp = inp.contiguous()
    M = inp.numel()

    out = torch.empty([], dtype=dtype, device=inp.device)
    _launch_global_sum(inp, out, M)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS SUM_OUT GCU400")
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    inp = inp.contiguous()
    M = inp.numel()
    _launch_global_sum(inp, out, M)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS SUM_DIM GCU400")
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if dim is None:
            out_shape = [1] * len(out_shape) if keepdim else []
        elif isinstance(dim, (list, tuple)) and len(dim) == 0:
            out_shape = [1] * len(out_shape) if keepdim else []
        else:
            dims_to_reduce = dim if isinstance(dim, (list, tuple)) else [dim]
            if keepdim:
                for d in dims_to_reduce:
                    out_shape[d % inp.ndim] = 1
            else:
                for d in sorted(dims_to_reduce, key=lambda x: x % inp.ndim, reverse=True):
                    out_shape.pop(d % inp.ndim)
        return torch.zeros(out_shape, dtype=dtype, device=inp.device)

    if dim is None:
        result = sum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    if dim == []:
        result = sum(inp, dtype=dtype)
        if keepdim:
            return torch.reshape(result, [1] * inp.ndim)
        return result

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)
    _launch_dim_sum(inp, out, M, N)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS SUM_DIM_OUT GCU400")
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    if dim == []:
        return sum_out(inp, dtype=dtype, out=out)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    _launch_dim_sum(inp, out, M, N)

    if not keepdim:
        out.squeeze_(dim=dim)
    return out
