import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

# torch.all: Tests if all elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_all(a, b):
    return a and b


@triton.jit
def reduce_any(a, b):
    return a or b


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
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Use all(x) = !any(!x) to workaround arith.andi not supported in Triton-to-PPL
    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        _any = _any or (a == 0.0)  # Check for zeros instead of non-zeros
    has_zero = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    # Convert to int32 to avoid type issues, then invert
    has_zero_int = has_zero.to(tl.int32)
    # If has_zero (1), result is 0; if no zero (0), result is 1
    all = 1 - has_zero_int
    tl.store(out, all[:, None], row_mask)


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
    # Use all(x) = !any(!x) to workaround arith.andi not supported in Triton-to-PPL
    # Check for zeros and use OR reduction (supported by Triton-to-PPL)
    has_zero = tl.reduce(inp_val == 0.0, axis=0, combine_fn=reduce_any)
    # Convert to int32 first to avoid type issues, then cast result
    has_zero_int = has_zero.to(tl.int32)
    # If has_zero (1), result should be 0 (False); if no zero (0), result is 1 (True)
    # This is: all_val = 1 - has_zero
    all_val = 1 - has_zero_int
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1)
    # Use all(x) = !any(!x) to workaround arith.andi not supported in Triton-to-PPL
    # mid_val is 0 or 1: check if any are 0 (False)
    has_false = tl.reduce(mid_val == 0, axis=0, combine_fn=reduce_any)
    # Convert to int32 to avoid type mismatch
    has_false_int = has_false.to(tl.int32)
    # If has_false (1), result is 0; if no false (0), result is 1
    all_val = 1 - has_false_int
    tl.store(out, all_val)


def all(inp):
    logging.debug("GEMS ALL")

    # If input is already scalar, handle directly
    if inp.ndim == 0:
        return inp.bool()

    # 1. Flatten to 1D
    flat_tensor = inp.flatten()
    n_elements = flat_tensor.numel()

    if n_elements == 0:
        return torch.tensor(True, dtype=torch.bool, device=inp.device)

    # 2. Calculate intermediate buffer size needed
    BLOCK_SIZE = 1024
    mid_size = math.ceil(n_elements / BLOCK_SIZE)

    # 3. Allocate intermediate buffer and output
    mid = torch.empty(mid_size, dtype=torch.int32, device=inp.device)
    out = torch.empty((), dtype=torch.bool, device=inp.device)

    # 4. Step 1: compute each block's result in parallel
    grid = (mid_size,)
    with torch_device_fn.device(inp.device):
        all_kernel_1[grid](
            flat_tensor, mid, n_elements, mid_size, BLOCK_SIZE=BLOCK_SIZE
        )

    # 5. Step 2: aggregate all block results
    BLOCK_MID = 1024
    grid = (1,)
    with torch_device_fn.device(inp.device):
        all_kernel_2[grid](mid, out, mid_size, BLOCK_MID=BLOCK_MID)

    # 6. Convert to bool and return
    return out.bool()


def all_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS ALL DIM")
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
    logging.debug("GEMS ALL DIMS")

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
