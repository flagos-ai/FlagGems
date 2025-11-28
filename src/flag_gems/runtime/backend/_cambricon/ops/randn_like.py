import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from ..utils import TOTAL_CORE_NUM
from .randn import randn_kernel

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
UNROLL = 4


def randn_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_CAMBRICON RANDN_LIKE")
    if device is None:
        device = x.device.index
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    philox_seed, philox_offset = philox_backend_seed_offset(N)
    with torch_device_fn.device(x.device):
        randn_kernel[grid_fn](
            out, N, philox_seed, philox_offset, num_stages=3, num_warps=1
        )
    return out
