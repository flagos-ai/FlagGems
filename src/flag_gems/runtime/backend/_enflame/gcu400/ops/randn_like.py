import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

from .randn import UNROLL, randn_kernel

logger = logging.getLogger(__name__)


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
    if N <= 65536:
        BLOCK = 1024
    elif N <= 1048576:
        BLOCK = 4096
    else:
        BLOCK = 8192
    grid = (triton.cdiv(N, BLOCK * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(x.device):
        reduced = dtype != torch.float32
        randn_kernel[grid](
            out,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            REDUCED=reduced,
            num_warps=4,
        )
    return out
