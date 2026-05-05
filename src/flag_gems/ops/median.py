import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def median_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    n_offset = tl.arange(0, BLOCK_N)
    n_mask = n_offset < N

    base_inp_offset = pid_m * N
    base_out_offset = pid_m * BLOCK_N

    inp_ptr = inp + base_inp_offset
    out_val_ptr = out_values + base_out_offset
    out_idx_ptr = out_indices + base_out_offset

    vals = tl.load(inp_ptr + n_offset, mask=n_mask, other=0.0)

    indices = tl.arange(0, BLOCK_N)
    sorted_vals, sorted_indices = tl.sort(vals, indices, dim=0)

    median_idx = N // 2 if N % 2 == 1 else N // 2 - 1
    median_val = sorted_vals[median_idx]
    median_idx_original = sorted_indices[median_idx]

    tl.store(out_val_ptr, median_val, mask=n_mask)
    tl.store(out_idx_ptr, median_idx_original, mask=n_mask)


def median(inp, dim=None, keepdim=False):
    logger.debug("GEMS MEDIAN")

    if inp.numel() == 0:
        raise ValueError("median cannot operate on an empty tensor")

    if dim is None:
        sorted_inp = torch.sort(inp.flatten())
        n = sorted_inp.values.numel()
        median_idx = n // 2 if n % 2 == 1 else n // 2 - 1
        return sorted_inp.values[median_idx]

    dim = dim if dim >= 0 else inp.ndim + dim
    shape = list(inp.shape)
    M = 1
    for i in range(inp.ndim):
        if i != dim:
            M *= shape[i]
    N = inp.shape[dim]

    inp = dim_compress(inp, dim)
    inp = inp.contiguous()

    out_values = torch.empty([M], dtype=inp.dtype, device=inp.device)
    out_indices = torch.empty([M], dtype=torch.long, device=inp.device)

    BLOCK_N = triton.next_power_of_2(N)

    grid = lambda meta: (M,)

    with torch_device_fn.device(inp.device):
        median_kernel[grid](inp, out_values, out_indices, M, N, BLOCK_N=BLOCK_N)

    out_values = out_values.reshape(shape[:dim] + shape[dim + 1 :])
    out_indices = out_indices.reshape(shape[:dim] + shape[dim + 1 :])

    if keepdim:
        out_values = out_values.unsqueeze(dim)
        out_indices = out_indices.unsqueeze(dim)

    return torch.return_types.median((out_values, out_indices))
