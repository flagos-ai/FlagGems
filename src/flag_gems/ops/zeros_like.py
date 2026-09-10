import logging

import torch
import triton

from flag_gems.ops.zeros import zeros_kernel
from flag_gems.runtime import device_guard

logger = logging.getLogger(__name__)


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with device_guard(x):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out

def zeros_like_paddle(x: 'Tensor', dtype = None, place = None) -> 'Tensor':
    return zeros_like(x, dtype=dtype, device=place)