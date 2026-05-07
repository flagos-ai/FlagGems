import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner

from ..utils import MAX_GRID_SIZE_X, TOTAL_CORE_NUM, cfggen_reduce_op
from .zeros import zero_

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@libtuner(
    configs=cfggen_reduce_op(), key=["M"], strategy=["log"], reset_to_zero=["out"]
)
@triton.jit
def sum_kernel_1(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = tl.zeros([BLOCK_SIZE], dtype=cdtype)
    block_start = block_start.to(tl.int64)
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=0.0)
        _tmp = inp_val + _tmp

    sum_val = tl.sum(_tmp)
    tl.atomic_add(out, sum_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sum"),
    key=["M", "N"],
    strategy=["log", "log"],
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
    elif tl.constexpr(inp.dtype.element_ty == tl.int1):
        cdtype = tl.int32
    else:
        cdtype = inp.dtype.element_ty
    prog_num = tl.num_programs(0).to(tl.uint64)
    sub_pid = tl.program_id(0).to(tl.uint64)
    task_num = tl.cdiv(M, BLOCK_M).to(tl.uint64)
    while sub_pid < task_num:
        # Map the program id to the row of inp it should compute.
        pid = sub_pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        inp_ = inp + pid * N
        out_ = out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(inp_ + cols, mask, other=0).to(cdtype)
            _sum += a
        sum = tl.sum(_sum, axis=1)[:, None]
        tl.store(out_, sum, row_mask)
        sub_pid += prog_num


def sum(inp, *, dtype=None):
    logger.debug("GEMS_CAMBRICON SUM")
    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    out = torch.zeros([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[grid](inp, out, M)
    return out.to(dtype)


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_CAMBRICON SUM_OUT")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32

    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[grid](inp, out, M)
    return out.to(dtype)


def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim is None:
        result = torch.sum(inp, dtype=dtype)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

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
    _out_provided = out is not None
    if _out_provided:
        dim_set = set(dim)
        if keepdim:
            out.resize_(shape)
        else:
            out.resize_([s for i, s in enumerate(shape) if i not in dim_set])
    else:
        out = torch.empty(shape, dtype=dtype, device=inp.device)
    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_M"]), MAX_GRID_SIZE_X // 4),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N)
    if not keepdim and not _out_provided:
        for d in sorted(dim, reverse=True):
            out = out.squeeze(dim=d)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_CAMBRICON SUM DIM")
    # support dim = 0, which are consistent with PyTorch
    if inp.numel() == 0:
        if dtype is None:
            dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

        out_shape = list(inp.shape)
        if dim is None:
            if keepdim:
                out_shape = [1] * len(out_shape)
            else:
                out_shape = []
        elif isinstance(dim, (list, tuple)) and len(dim) == 0:
            if keepdim:
                out_shape = [1] * len(out_shape)
            else:
                out_shape = []
        else:
            dims_to_reduce = dim if isinstance(dim, (list, tuple)) else [dim]
            if keepdim:
                for d in dims_to_reduce:
                    out_shape[d % inp.ndim] = 1
            else:
                sorted_dims_to_remove = sorted(
                    dims_to_reduce, key=lambda x: x % inp.ndim, reverse=True
                )
                for d in sorted_dims_to_remove:
                    index_to_remove = d % inp.ndim
                    out_shape.pop(index_to_remove)
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        zero_(out)
        return out
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_CAMBRICON SUM_DIM_OUT")
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype, out=out)
