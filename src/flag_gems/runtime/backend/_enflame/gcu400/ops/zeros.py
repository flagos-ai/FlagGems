import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if dtype == torch.int64:
        dtype = torch.int32

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), 24),)
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024 * 16, num_warps=1)
    return out
