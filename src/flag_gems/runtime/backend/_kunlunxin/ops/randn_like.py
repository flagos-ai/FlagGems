import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .randn import randn_kernel

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
UNROLL = 4


def randn_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS RANDN_LIKE")
    if device is None:
        device = x.device.index
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    cluster_num = 12
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)), 1024)
    grid_fn = triton.cdiv(N, BLOCK_SIZE * UNROLL)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(x.device):
        randn_kernel[(grid_fn,)](out, N, philox_seed, philox_offset, BLOCK_SIZE)
    return out
