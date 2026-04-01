import logging

import torch
import triton

from flag_gems.runtime.backend._ascend.ops.randn import (
    BLOCK_SIZE_SUB,
    UNROLL,
    _compute_block_size,
    randn_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


def randn_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ASCEND RANDN_LIKE")
    if device is None:
        device = x.device.index
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    block_size = _compute_block_size(N)
    grid = (triton.cdiv(N, block_size * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(x.device):
        randn_kernel[grid](out, N, philox_seed, philox_offset, block_size, BLOCK_SIZE_SUB)
    return out
