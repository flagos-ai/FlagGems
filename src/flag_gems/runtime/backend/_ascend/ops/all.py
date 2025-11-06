import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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
    workers = tle.num_programs(0)
    pid = tle.program_id(0)

    total_workloads = tl.cdiv(M, BLOCK_M)
    workloads = tl.cdiv(total_workloads, workers)

    for w in range(workloads):
        work_id = pid + w * workers
        rows = work_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ninp = inp + rows * N
        nout = out + rows
        row_mask = rows < M

        _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(ninp + cols, mask, other=1.0)
            _all = _all and (a != 0)
        all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
        tl.store(nout, all[:, None], row_mask)


@libentry()
@triton.jit
def all_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def all(inp):
    logger.debug("GEMS_ASCEND ALL")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
        all_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


def all_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ASCEND ALL DIM")
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

        def grid(meta):
            axis0 = triton.cdiv(M, meta["BLOCK_M"])
            axis0 = axis0 if axis0 < 40 else 40
            return (axis0,)

        with torch_device_fn.device(inp.device):
            all_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def all_dims(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ASCEND ALL DIMS")

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

    def grid(meta):
        axis0 = triton.cdiv(M, meta["BLOCK_M"])
        axis0 = axis0 if axis0 < 40 else 40
        return (axis0,)

    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
