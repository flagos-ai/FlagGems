import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .zeros import zeros_kernel

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ASCEND ZEROS_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    if x.numel() == 0:
        return torch.empty_like(x, device=device, dtype=dtype)
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    # grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    grid_fn = lambda meta: (triton.cdiv(N, 32768),)
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=32768, BLOCK_SIZE_SUB=1024)
    return out
