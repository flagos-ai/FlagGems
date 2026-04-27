import logging

import torch

from flag_gems.ops.full import check_dtype, full_func, full_func_scalar

logger = logging.getLogger(__name__)


def new_full(
    self,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS NEW_FULL")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty(size, device=device, dtype=dtype)
    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value)
    else:
        return full_func_scalar(out, fill_value)
