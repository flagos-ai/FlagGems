import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from torch.autograd.function import once_differentiable

from flag_gems.runtime import torch_device_fn
from flag_gems.utils._atanh_autograd import (
    UnaryAutograd,
    check_inplace,
    increment_version,
)

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def atanh_func(x):
    x32 = x.to(tl.float32)
    return 0.5 * tl.log((1.0 + x32) / (1.0 - x32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def _atanh_backward(x, grad):
    value = x.to(tl.float32)
    square = (value * value).to(x.dtype).to(tl.float32)
    denominator = (1.0 - square).to(x.dtype).to(tl.float32)
    return grad.to(tl.float32) / denominator


class _AtanhAutograd(UnaryAutograd):
    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]
        with torch_device_fn.device(x.device):
            result = _atanh_backward(x, grad)
        return result, None, None, None, None


def unary_autograd(x, op, forward, alpha=1.0, inplace=False):
    if inplace:
        check_inplace(x)
    if torch.is_grad_enabled() and x.requires_grad:
        return _AtanhAutograd.apply(x, op, forward, alpha, inplace)
    result = forward(x)
    if inplace:
        increment_version(x)
    return result


def atanh(A):
    logger.debug("GEMS_KUNLUNXIN ATANH")
    return unary_autograd(A, "atanh", atanh_func)


def atanh_(A):
    logger.debug("GEMS_KUNLUNXIN ATANH_")
    if not A.dtype.is_floating_point:
        raise RuntimeError(
            f"result type Float can't be cast to the desired output type {A.dtype}"
        )
    return unary_autograd(A, "atanh", lambda x: atanh_func(x, out0=x), inplace=True)
