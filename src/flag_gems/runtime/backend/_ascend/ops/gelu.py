import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils import tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
try:
    import torch_npu # noqa: F401
except: # noqa: E722
    pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    x_f32 = x.to(tl.float32)
    scale: tl.constexpr = 0.7071067811 # 1 / math.sqrt(2)
    output = 0.5 * x * (1 + erf(x_f32 * scale))
    return output

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    x_f32 = x.to(tl.float32)
    output = 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * x_f32 * x_f32)))
    return output

@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811 # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803 # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    x_sqrt = scale1 * x_fp32
    dydx = scale2 * x_fp32 * exp(-x_sqrt * x_sqrt) + 0.5 * erf(x_sqrt) + 0.5
    dx = dydx * dy
    return dx

@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    c1 = 0.79788456 # math.sqrt(2 / math.pi)
    c2 = 0.044715
    # z = c1 * (x + c2 * x**3)
    tanh_out = tanh(c1 * x_fp32 * (1 + c2 * x_fp32 * x_fp32))
    # dz_dx = c1 * (1 + 3 * c2 * x * x)
    # 0.1070322243 = c1 * 3 *c2
    dydx = 0.5 * (
        x * ((1 - tanh_out * tanh_out) * (c1 + 0.1070322243 * x_fp32 * x_fp32))
        + (1 + tanh_out)
    )
    dx = dydx * dy
    return dx

def gelu(self, *, approximate="none"):
    logging.debug("ASCEND GEMS GELU FORWARD")
    if approximate == "tanh":
        out = gelu_tanh(self)
    else:
        out = gelu_none(self)
    return out

def gelu_backward(grad_output, self, *, approximate="none"):
    logging.debug("ASCEND GEMS GELU BACKWARD")
    if approximate == "tanh":
        in_grad = gelu_backward_tanh(self, grad_output)
    else:
        in_grad = gelu_backward_none(self, grad_output)
    return in_grad

def gelu_(A, *, approximate="none"):
    logging.debug("ASCEND GEMS GELU_ FORWARD")
    if approximate == "tanh":
        out = gelu_tanh(A, out0=A)
    else:
        out = gelu_none(A, out0=A)
    return out