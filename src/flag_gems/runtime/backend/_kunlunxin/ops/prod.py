import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@libentry()
@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
    prod_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_mul)
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    logger.debug("GEMS PROD")
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = get_block_size_1d(M, inp.element_size())
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](
            inp, mid, M, block_size, buffer_size_limit=2048
        )
        if mid_size == 1:
            return mid.reshape([])
        prod_kernel_result[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=2048
        )
    return out


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def prod_kernel(
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
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _prod = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        tmp = _prod + a
        _prod = tl.where(mask, tmp, _prod)

    prod = tl.reduce(_prod, axis=1, combine_fn=reduce_mul)[:, None]
    tl.store(out, prod, row_mask)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS prod DIM")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    shape = list(inp.shape)
    dim = [dim % inp.ndim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch.cuda.device(inp.device):
        prod_kernel[grid](inp, out, M, N, buffer_size_limit=2048)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
