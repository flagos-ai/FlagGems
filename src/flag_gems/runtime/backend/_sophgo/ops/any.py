import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

# torch.any: Tests if any elements in input evaluate to True. If the dtype of input
#            is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok.


@triton.jit
def reduce_any(a, b):
    return a or b


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("any"), key=["M", "N"])
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


def any(inp):
    """
    Implement global any by reusing any_kernel_dim.
    Strategy: Reshape input to 2D, use any_kernel_dim for two reductions.
    """
    logging.debug("GEMS ANY (via any_kernel_dim)")

    n_elements = inp.numel()

    # Handle edge cases
    if n_elements == 0:
        return torch.tensor(False, dtype=torch.bool, device=inp.device)
    if n_elements == 1:
        return inp.flatten()[0] != 0

    # Reshape input to a near-square 2D tensor
    # Calculate appropriate row and column counts
    rows = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    cols = triton.cdiv(n_elements, rows)

    # Padding if needed
    if rows * cols > n_elements:
        # Create padded 1D tensor
        inp_flat = inp.flatten()
        padding = torch.zeros(
            rows * cols - n_elements, dtype=inp.dtype, device=inp.device
        )
        inp_padded = torch.cat([inp_flat, padding])
    else:
        inp_padded = inp.flatten()[: rows * cols]

    # Reshape to 2D
    inp_2d = inp_padded.reshape(rows, cols)

    # First reduction: column-wise reduce (dim=1), get a 1D tensor of shape (rows,)
    temp_shape = [rows, 1]
    temp = torch.empty(temp_shape, dtype=torch.bool, device=inp.device)

    M1 = rows
    N1 = cols
    grid1 = lambda meta: (triton.cdiv(M1, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        any_kernel_dim[grid1](inp_2d, temp, M1, N1)

    temp = temp.squeeze(dim=1)  # (rows,)

    # Second reduction: reduce over 1D tensor
    # Treat 1D tensor as 2D tensor of shape (1, rows)
    temp_2d = temp.reshape(1, rows)
    out_shape = [1, 1]
    out = torch.empty(out_shape, dtype=torch.bool, device=inp.device)

    M2 = 1
    N2 = rows
    grid2 = lambda meta: (triton.cdiv(M2, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        any_kernel_dim[grid2](temp_2d, out, M2, N2)

    # Return scalar
    return out.squeeze()


def any_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS ANY DIM")
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

        out = torch.empty(shape, dtype=torch.bool, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            any_kernel_dim[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logging.debug("GEMS ANY DIMS")

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

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        any_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
