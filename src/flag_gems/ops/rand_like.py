import logging

import torch
import triton

from flag_gems.ops.rand import rand_kernel
from flag_gems.utils.random_utils import philox_cuda_seed_offset


def rand_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logging.debug("GEMS RAND_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    philox_seed, philox_offset = philox_cuda_seed_offset(N)
    with torch.cuda.device(x.device):
        rand_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out
