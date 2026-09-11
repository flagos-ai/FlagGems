import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

device_ = device
logger = logging.getLogger(__name__)

BLOCK_SIZE = 1024
NUM_WARPS = 1
GRID_SIZE = 24


@triton.jit(do_not_specialize=["n_elements"])
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (min(triton.cdiv(n_elements, BLOCK_SIZE), GRID_SIZE),)
    zeros_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
    return out
