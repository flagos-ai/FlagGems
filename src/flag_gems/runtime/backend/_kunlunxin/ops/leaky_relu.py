import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=False,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    x_fp32 = x.to(tl.float32)
    return tl.maximum(x_fp32, 0.0) + negative_slope * tl.minimum(x_fp32, 0.0)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU")
    return leaky_relu_kernel(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_")
    return leaky_relu_kernel(A, negative_slope, out0=A)


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu_kernel(A, negative_slope)
    return leaky_relu_kernel(A, negative_slope, out0=out)


_LEAKY_BACKWARD_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_kernel(g, x, negative_slope):
    step = tl.minimum(tl.maximum(x * 1.0e30, 0.0), 1.0)
    return g * (negative_slope + (1.0 - negative_slope) * step)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def leaky_relu_backward_general_kernel(g, x, negative_slope):
    x_fp32 = x.to(tl.float32)
    g_fp32 = g.to(tl.float32)
    return tl.where(x_fp32 > 0.0, g_fp32, g_fp32 * negative_slope)


def leaky_relu_backward(grad_output, self, negative_slope=0.01, self_is_result=False):
    logger.debug("GEMS_KUNLUNXIN LEAKY_RELU_BACKWARD")
    if grad_output.numel() == 0:
        return torch.empty_like(self)
    if grad_output.dtype in _LEAKY_BACKWARD_DTYPES and (
        grad_output.is_contiguous() and self.is_contiguous()
    ):
        return leaky_relu_backward_kernel(grad_output, self, negative_slope)
    return leaky_relu_backward_general_kernel(grad_output, self, negative_slope)
