import logging
from typing import Optional

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "INT_TO_FLOAT")]
)
@triton.jit
def logit_forward_kernel(x, minval, maxval):
    z = tl.minimum(tl.maximum(x.to(tl.float32), minval), maxval)
    return tl.log(z / (1.0 - z))


@pointwise_dynamic(
    is_tensor=[True, True, False, False], promotion_methods=[(0, "INT_TO_FLOAT")]
)
@triton.jit
def logit_backward_kernel(x, dy, minval, maxval):
    x_f32 = x.to(tl.float32)
    is_clamped = False
    is_clamped = (x_f32 < minval) or (x_f32 > maxval)
    x_f32 = tl.minimum(tl.maximum(x_f32, minval), maxval)
    dx = dy.to(tl.float32) / (x_f32 * (1.0 - x_f32))
    return tl.where(is_clamped, 0.0, dx)


def logit(self, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT FORWARD")
    if eps is None:
        eps = 0
    minval, maxval = eps, 1.0 - eps
    output = logit_forward_kernel(self, minval, maxval)
    return output


def logit_backward(grad_output, self, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT BACKWARD")
    if eps is None:
        eps = 0
    minval, maxval = eps, 1.0 - eps
    grad_input = logit_backward_kernel(self, grad_output, minval, maxval)
    return grad_input


def logit_(A, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT_ FORWARD")
    if eps is None:
        eps = 0
    minval, maxval = eps, 1.0 - eps
    out = logit_forward_kernel(A, minval, maxval, out0=A)
    return out


def logit_out(A, out, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT_OUT FORWARD")
    if eps is None:
        eps = 0
    minval, maxval = eps, 1.0 - eps
    out = logit_forward_kernel(A, minval, maxval, out0=out)
    return out
