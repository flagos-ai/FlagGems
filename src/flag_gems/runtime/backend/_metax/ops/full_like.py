import logging

import torch

from .full import check_dtype, full_

logger = logging.getLogger("flag_gems." + __name__)


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
    logger.debug("METAX GEMS FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()

    return full_(out, N, dtype, device, fill_value)
