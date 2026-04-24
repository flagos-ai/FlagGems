import logging

# import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

# Use built-in triton.language functions instead of tl_extra_shim
# to avoid None return values that cause compilation errors
erf = tl.erf
exp = tl.exp


# pow is not available, use ** operator directly
# tanh is not available, implement using exp: tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
@triton.jit
def tanh(x):
    exp2x = exp(2.0 * x)
    return (exp2x - 1.0) / (exp2x + 1.0)


logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    output = 0.5 * x * (1 + erf(x.to(tl.float32) * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    output = 0.5 * x * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * (x_fp32**2))))
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * exp(-((scale1 * x_fp32) ** 2))
        # scale2 * x_fp32 * torch.exp(-torch.pow(scale1 * x_fp32, 2))
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
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * (x_fp32**2)))
    dydx = 0.5 * x_fp32 * (
        (1 - (tanh_out**2)) * (0.79788456 + 0.1070322243 * (x_fp32**2))
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
