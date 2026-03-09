import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    output = 0.5 * x * (1 + erf(x * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        0.5
        * x
        * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2.0))))
    )
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * tl.exp(-pow(scale1 * x_fp32, 2.0))
        + 0.5 * erf(scale1 * x_fp32)
        + 0.5
    )
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * pow(x_fp32, 2.0)))
    dydx = 0.5 * x * (
        (1 - pow(tanh_out, 2.0)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2.0))
    ) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx


def gelu(self, *, approximate="none"):
    logger.debug("GEMS GELU FORWARD")
    if approximate == "tanh":
        out = gelu_tanh(self)
    else:
        out = gelu_none(self)
    return out


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS GELU BACKWARD")
    if approximate == "tanh":
        in_grad = gelu_backward_tanh(self, grad_output)
    else:
        in_grad = gelu_backward_none(self, grad_output)
    return in_grad


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS GELU_ FORWARD")
    if approximate == "tanh":
        out = gelu_tanh(A, out0=A)
    else:
        out = gelu_none(A, out0=A)
    return out
