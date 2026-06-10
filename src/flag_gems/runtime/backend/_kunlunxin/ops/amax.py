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


@libentry()
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    amax_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amax_val)


@libentry()
@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("amax"), key=["M", "N"])
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _all = tl.full([BLOCK_M, BLOCK_N], value=-float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=-float("inf")).to(tl.float32)
        a = tl.where(mask, a, -float("inf"))
        _all = tl.maximum(_all, a)
    all = tl.max(_all, axis=1)[:, None]
    tl.store(out, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMAX")
    if dim is None or len(dim) == 0:
        M = inp.numel()
        # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        block_size = get_block_size_1d(M, inp.element_size())
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amax_kernel_1[(mid_size, 1)](
                inp, mid, M, block_size, buffer_size_limit=2048
            )
            amax_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid, buffer_size_limit=2048
            )  # max block size is 128k, so mid does not requires int64 index
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

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
            amax_kernel[grid](inp, out, M, N, buffer_size_limit=2048)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out
