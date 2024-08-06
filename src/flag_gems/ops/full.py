import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import volume


@triton.jit(do_not_specialize=["fill_value"])
def full_kernel(
    output_ptr,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, fill_value, mask=mask)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("GEMS FULL")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cuda")

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(device):
        full_kernel[grid_fn](out, N, fill_value, BLOCK_SIZE=1024)
    return out
