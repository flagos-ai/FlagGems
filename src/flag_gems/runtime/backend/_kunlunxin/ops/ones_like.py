import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .ones import ones_kernel

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def ones_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS ONES_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = (12, 1, 1)
    block_size = triton.next_power_of_2(triton.cdiv(N, 12))
    with torch_device_fn.device(x.device):
        ones_kernel[grid_fn](
            out,
            N,
            1.0,
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
            isCloseDtypeConvert=True,
        )
    return out
