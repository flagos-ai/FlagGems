import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .ones import ones_kernel


def ones_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logging.debug("GEMS ONES_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = (12, 1, 1)
    block_size = triton.next_power_of_2(triton.cdiv(N, 12))
    with torch_device_fn.device(x.device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=block_size)
    return out
