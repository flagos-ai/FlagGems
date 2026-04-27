import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .zeros import zeros_kernel

TOTAL_CORE_NUM = torch_device_fn.get_device_properties().multi_processor_count


logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_TSINGMICRO ZEROS_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](out, N)
    return out
