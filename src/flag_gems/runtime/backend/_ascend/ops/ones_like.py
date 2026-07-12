import logging
import math

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .ones import ones_kernel

logger = logging.getLogger(__name__)


def ones_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ASCEND ONES_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(N)))
    grid_fn = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out
