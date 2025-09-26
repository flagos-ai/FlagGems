import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.max import max_kernel_1, max_kernel_2
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
thread_num = core_num * cluster_num
buf_len_per_core = 2048
vector_size = 16


def get_block(n: int) -> int:
    if n < cluster_num:
        res = cluster_num
    else:
        res = cluster_num * triton.cdiv(n, cluster_num)
    return res


def heur_m_block_size(args):
    return triton.next_power_of_2(min(triton.cdiv(args["M"], cluster_num), core_num))


def heur_n_block_size(args):
    return triton.next_power_of_2(min(args["N"], triton.cdiv(buf_len_per_core, 4)))


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("any"), key=["M", "N"])
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def any_kernel_dim(
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

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any[:, None], row_mask)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def max_kernel_dim(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    xoffset = tl.program_id(0) * BLOCK_M
    xindex = xoffset + tl.arange(0, BLOCK_M)[:, None]
    xmask = xindex < M
    rbase = tl.arange(0, BLOCK_N)[None, :]
    _max = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
    for roffset in range(0, N, BLOCK_N):
        rindex = roffset + rbase
        rmask = rindex < N
        r1 = rindex
        inp = tl.load(
            in_ptr + (r1 + (N * xindex)), rmask & xmask, other=float("-inf")
        ).to(tl.float32)
        inpb = tl.broadcast_to(inp, [BLOCK_M, BLOCK_N])
        _max = tl.maximum(_max, inpb)
    tmp2 = tl.max(_max, axis=1, return_indices=False)[:, None]
    tl.store(out_ptr + xindex, tmp2, xmask)


@libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_any)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val)


@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.int1)
    any_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_any)
    tl.store(out, any_val)


def any(inp):
    logger.debug("GEMS ANY")
    n_elements = inp.numel()
    block_size = min(
        triton.cdiv(get_block(n_elements), cluster_num),
        triton.cdiv(buf_len_per_core * core_num, 4),
    )
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    if n_elements >= vector_size * thread_num:
        # according to api, op == any, use max to calculate
        inpf = inp.to(torch.float)
        midf = torch.empty((mid_size,), dtype=torch.float, device=inp.device)
        outf = torch.empty([], dtype=torch.float, device=inp.device)

        with torch_device_fn.device(inp.device):
            max_kernel_1[(mid_size, 1)](
                inpf, midf, n_elements, block_size, buffer_size_limit=2048
            )
            if mid_size == 1:
                return midf.to(torch.bool).reshape([])
            max_kernel_2[(1, 1)](
                midf, outf, mid_size, block_mid, buffer_size_limit=2048
            )
        out = outf.to(torch.bool)
    else:
        mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
        out = torch.empty([], dtype=torch.bool, device=inp.device)

        with torch_device_fn.device(inp.device):
            any_kernel_1[(mid_size, 1)](
                inp, mid, n_elements, block_size, buffer_size_limit=2048
            )
            if mid_size == 1:
                return mid.reshape([])
            any_kernel_2[(1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)

    return out


def any_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIM")
    shape = list(inp.shape)
    if dim is None:
        out = any(inp)
        if keepdim:
            out = torch.reshape(out, [1] * inp.ndim)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        if N >= vector_size * vector_size:
            # according to api, op == any, use max to calculate
            inpf = inp.to(torch.float)
            outf = torch.empty(shape, dtype=torch.float, device=inp.device)

            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
            with torch_device_fn.device(inp.device):
                max_kernel_dim[grid](inpf, outf, M, N, buffer_size_limit=2048)
            out = outf.to(torch.bool)
        else:
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
            with torch_device_fn.device(inp.device):
                any_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS ANY DIMS")

    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if N >= vector_size * core_num:
        # according to api, op == any, use max to calculate
        inpf = inp.to(torch.float)
        outf = torch.empty(shape, dtype=torch.float, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            max_kernel_dim[grid](inpf, outf, M, N, buffer_size_limit=2048)
        out = outf.to(torch.bool)
    else:
        out = torch.empty(shape, dtype=torch.bool, device=inp.device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            any_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
