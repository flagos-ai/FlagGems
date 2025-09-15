import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .ones import ones_kernel

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


def ones_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_MTHREADS ONES_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out
