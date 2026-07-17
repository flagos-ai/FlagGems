import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def scalar_tensor_kernel(out, value_scalar):
    tl.store(out, value_scalar)


def scalar_tensor(s, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS SCALAR_TENSOR")
    out = torch.empty(
        (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    if dtype == torch.bool:
        s = bool(s)
    with torch_device_fn.device(out.device):
        scalar_tensor_kernel[(1,)](out, s)
    return out
