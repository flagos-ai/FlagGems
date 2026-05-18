import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@triton.jit
def reduce_any(a, b):
    return a | b


# ========== Global any (optimized: grid-stride + bool→u8) ==========

@libentry()
@triton.jit(do_not_specialize=["N_total"])
def any_global_kernel(
    inp_ptr, mid_ptr, N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int1)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        val = tl.load(inp_ptr + off, mask=mask, other=0)
        acc = acc | (val != 0)
    result = tl.reduce(acc, axis=0, combine_fn=reduce_any)
    tl.store(mid_ptr + pid, result)


@libentry()
@triton.jit
def any_reduce_kernel(mid_ptr, out_ptr, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0).to(tl.int1)
    result = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out_ptr, result)


# ========== Dim any (kept close to generic, with bool→u8 + fixed &/|) ==========

def _keep_config(conf):
    bm = conf.kwargs["BLOCK_M"]
    bn = conf.kwargs["BLOCK_N"]
    if bm * bn > 131072:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(_keep_config, runtime.get_tuned_config("naive_reduction"))),
    key=["M", "N"],
)
@triton.jit
def any_kernel_dim(
    inp, out, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any_result = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any_result[:, None], row_mask)


def _to_u8_if_bool(inp):
    if inp.dtype == torch.bool:
        return inp.view(torch.uint8)
    return inp


def any(inp):
    logger.debug("GEMS ANY GCU400")
    inp = _to_u8_if_bool(inp)
    N_total = inp.numel()
    if N_total <= 4096:
        BLOCK = max(triton.next_power_of_2(N_total), 1024)
    elif N_total <= 65536:
        BLOCK = 4096
    else:
        BLOCK = 65536
    num_blocks = triton.cdiv(N_total, BLOCK)
    grid_size = min(num_blocks, NUM_SIPS * 2)

    mid = torch.empty((grid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        any_global_kernel[(grid_size,)](inp, mid, N_total, BLOCK)
        block_mid = triton.next_power_of_2(grid_size)
        any_reduce_kernel[(1,)](mid, out, grid_size, block_mid)

    return out


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIM GCU400")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        inp = _to_u8_if_bool(inp)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=torch.bool, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            any_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIMS GCU400")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    inp = _to_u8_if_bool(inp)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        any_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
