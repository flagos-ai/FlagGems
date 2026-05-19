import logging

import torch
import triton

from flag_gems.ops.log_softmax import log_softmax_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(inp.device):
        log_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            K,
            BLOCK_N=128,
            num_warps=8,
        )
    return out
