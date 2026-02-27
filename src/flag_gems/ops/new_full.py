import logging

import torch

from flag_gems.ops.full import check_dtype, full_func, full_func_scalar

logger = logging.getLogger(__name__)


def new_full(
    x,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    logger.debug("GEMS NEW_FULL")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty(size, device=device, dtype=dtype)
    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value, out0=out)
    else:
        return full_func_scalar(out, fill_value, out0=out)
