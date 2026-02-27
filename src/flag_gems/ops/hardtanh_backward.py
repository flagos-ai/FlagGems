import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def hardtanh_backward_kernel(grad_output, self, min_val, max_val):
    self_fp32 = self.to(tl.float32)
    in_range = (self_fp32 > min_val) & (self_fp32 < max_val)
    return tl.where(in_range, grad_output, 0)


def hardtanh_backward(grad_output, self, min_val, max_val):
    logger.debug("GEMS HARDTANH BACKWARD")
    grad_input = hardtanh_backward_kernel(grad_output, self, min_val, max_val)
    return grad_input
