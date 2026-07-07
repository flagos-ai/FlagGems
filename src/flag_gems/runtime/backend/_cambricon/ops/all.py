import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

from ..utils import TOTAL_CORE_NUM, cfggen_reduce_op2

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("all"), key=["M", "N"])
@triton.jit
def all_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)


@libentry()
@triton.autotune(configs=cfggen_reduce_op2(), key=["M"])
@triton.jit
def all_kernel_1(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
    ITER_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = tl.full([BLOCK_SIZE], value=1, dtype=tl.int1)
    block_start = block_start.to(tl.int64)
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=1.0)
        _tmp = _tmp and (inp_val != 0)

    # Reset to original reduce programming mode after optimizing the tl.reduce.
    for x in tl.static_range(1, int(ITER_NUM), 1):
        _tmp[: BLOCK_SIZE // (2**x)] = (
            _tmp[: BLOCK_SIZE // (2**x)]
            and _tmp[BLOCK_SIZE // (2**x) : (BLOCK_SIZE // (2**x)) * 2]
        )

    tl.atomic_and(out, _tmp[0].to(tl.int32))


def all(inp):
    logger.debug("GEMS_CAMBRICON ALL")
    M = inp.numel()
    grid = lambda meta: (min(triton.cdiv(M, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    out = torch.ones([], dtype=torch.int32, device=inp.device)

    with torch_device_fn.device(inp.device):
        all_kernel_1[grid](inp, out, M)

    return out.to(torch.bool)


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_CAMBRICON ALL DIM")
    shape = list(inp.shape)
    if dim is None:
        out = all(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=torch.bool, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            all_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_CAMBRICON ALL DIMS")

    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
