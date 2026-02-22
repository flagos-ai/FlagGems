import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    m = tl.maximum(x_fp32, y_fp32)
    return m + tl.log(1.0 + tl.exp(-tl.abs(x_fp32 - y_fp32)))


def logaddexp(self, other):
    logger.debug("GEMS LOGADDEXP")
    if not self.is_floating_point() or not other.is_floating_point():
        raise RuntimeError("logaddexp is only supported for floating point tensors.")
    return logaddexp_kernel(self, other)


def logaddexp_out(self, other, out):
    logger.debug("GEMS LOGADDEXP OUT")
    if not self.is_floating_point() or not other.is_floating_point():
        raise RuntimeError("logaddexp is only supported for floating point tensors.")
    return logaddexp_kernel(self, other, out0=out)
