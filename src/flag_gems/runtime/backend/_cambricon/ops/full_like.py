import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

from ..utils import TOTAL_CORE_NUM
from .full import check_dtype, full_scalar_kernel, full_tensor_kernel

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logger.debug("GEMS_CAMBRICON FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(x.device):
        if isinstance(fill_value, torch.Tensor):
            full_tensor_kernel[grid_fn](
                out,
                N,
                fill_value,
            )
        else:
            full_scalar_kernel[grid_fn](
                out,
                N,
                fill_value,
            )
    return out
