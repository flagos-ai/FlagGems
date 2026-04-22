import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.

cluster_num = 12
core_num = 64
buf_len_per_core = 2048
vector_size = 16


def heur_m_block_size(args):
    return triton.next_power_of_2(min(triton.cdiv(args["M"], cluster_num), core_num))


def heur_n_block_size(args):
    return triton.next_power_of_2(min(args["N"], 512))


@triton.jit
def reduce_all(a, b):
    return a and b


@libentry()
@triton.jit
def all_global_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Global all over all elements. C++ handler replaces with api::all<T,bool>.
    Triton fallback: single program loops over chunks of BLOCK_SIZE."""
    _all = tl.full([BLOCK_SIZE], value=1, dtype=tl.int1)
    for off in range(0, n_elements, BLOCK_SIZE):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements
        val = tl.load(inp + offset, mask=mask, other=1.0)
        _all = _all and (val != 0)
    result = tl.reduce(_all, axis=0, combine_fn=reduce_all)
    tl.store(out, result)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
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
    pid = tle.program_id(0)
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


def all(inp):
    logger.debug("GEMS ALL")
    n_elements = inp.numel()
    # BLOCK_SIZE must fit in XPU per-core local buffer so the Triton fallback
    # kernel always compiles.  The C++ handler (api::all<T,bool>) ignores this
    # value and handles any n_elements internally.
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), buf_len_per_core)
    out = torch.empty([], dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_global_kernel[(1, 1)](
            inp, out, n_elements, BLOCK_SIZE, buffer_size_limit=2048
        )
    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ALL DIM")
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

        if N == 1:
            # N==1: each row has a single element; avoid kernel dispatch for
            # trivial case that some hardware configs cannot handle.
            out = (inp.reshape(M) != 0).reshape(shape)
        else:
            out = torch.empty(shape, dtype=torch.bool, device=inp.device)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
            with torch_device_fn.device(inp.device):
                all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS ALL DIMS")

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

    if N == 1:
        out = (inp.reshape(M) != 0).reshape(shape)
    else:
        out = torch.empty(shape, dtype=torch.bool, device=inp.device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            all_kernel_dim[grid](inp, out, M, N, buffer_size_limit=2048)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
