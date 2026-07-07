import logging

import torch

from flag_gems.ops.full import check_dtype, full_func, full_func_scalar

logger = logging.getLogger(__name__)


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
    logger.debug("GEMS FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    size = x.size()
    out = torch.empty(size, device=device, dtype=dtype)
    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value)
    else:
        return full_func_scalar(out, fill_value)
