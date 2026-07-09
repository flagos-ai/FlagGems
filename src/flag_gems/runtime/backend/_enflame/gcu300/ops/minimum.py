import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


def _is_float64_scalar(*args):
    return any(isinstance(a, torch.Tensor) and a.dtype == torch.float64 and a.ndim == 0 for a in args)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 0, "DEFAULT")])
@triton.jit
def minimum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.minimum(X, Y)


def minimum(X, Y):
    logger.debug("GEMS_ENFLAME MINIMUM")
    if _is_float64_scalar(X, Y):
        dev = X.device
        return torch.minimum(X.cpu(), Y.cpu()).to(dev)
    assert X.device.type == device and Y.device.type == device
    return minimum_kernel(X, Y)
