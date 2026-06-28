import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def hardtanh_kernel(x, min_val, max_val):
    x_f32 = x.to(tl.float32)
    return tl.minimum(tl.maximum(x_f32, min_val), max_val).to(x.dtype)


def hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    logger.debug("GEMS HARDTANH")
    return hardtanh_kernel(x, float(min_val), float(max_val))


def hardtanh_out(
    x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0, out=None
):
    if out is None:
        return hardtanh(x, min_val, max_val)
    result = hardtanh(x, min_val, max_val)
    out.copy_(result)
    return out


@pointwise_dynamic(
    is_tensor=[True, True, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def hardtanh_backward_kernel(dy, x, min_val, max_val):
    x_f32 = x.to(tl.float32)
    in_range = (x_f32 > min_val) & (x_f32 < max_val)
    return tl.where(in_range, dy.to(tl.float32), 0.0).to(dy.dtype)


def hardtanh_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
):
    logger.debug("GEMS HARDTANH BACKWARD")
    return hardtanh_backward_kernel(
        grad_output, self, float(min_val), float(max_val)
    )
