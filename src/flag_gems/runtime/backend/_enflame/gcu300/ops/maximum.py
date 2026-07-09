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


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


def maximum(X, Y):
    logger.debug("GEMS_ENFLAME MAXIMUM")
    if _is_float64_scalar(X, Y):
        dev = X.device
        return torch.maximum(X.cpu(), Y.cpu()).to(dev)
    assert X.device.type == device and Y.device.type == device
    return maximum_kernel(X, Y)
