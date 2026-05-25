import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

# Use built-in triton.language functions instead of tl_extra_shim
# to avoid None return values that cause compilation errors
exp = tl.exp


# Polynomial approximation of erf that does not rely on tl.math.erf
# (which PPL compiler cannot handle). Abramowitz & Stegun 7.1.26,
# max error ~1.5e-7.
@triton.jit
def erf_poly(x):
    abs_x = tl.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * abs_x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = (
        0.254829592 * t
        + (-0.284496736) * t2
        + 1.421413741 * t3
        + (-1.453152027) * t4
        + 1.061405429 * t5
    )
    result = 1.0 - poly * tl.exp(-abs_x * abs_x)
    return tl.where(x >= 0.0, result, -result)


# pow is not available, use ** operator directly
# tanh is not available, implement using exp: tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
@triton.jit
def tanh(x):
    exp2x = exp(2.0 * x)
    return (exp2x - 1.0) / (exp2x + 1.0)


logger = logging.getLogger(__name__)

_base = get_codegen_config()
_gelu_config = CodeGenConfig(
    max_tile_size=2048,
    max_grid_size=(65536, 1, 1),
    max_num_warps_per_cta=_base.max_num_warps_per_cta,
    prefer_block_pointer=_base.prefer_block_pointer,
    prefer_1d_tile=_base.prefer_1d_tile,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_gelu_config)
@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    output = 0.5 * x * (1 + erf_poly(x.to(tl.float32) * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_gelu_config)
@triton.jit
def gelu_tanh(x):
    x_fp32 = x.to(tl.float32)
    output = 0.5 * x * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * (x_fp32 * x_fp32))))
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=_gelu_config)
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    sx = scale1 * x_fp32
    dydx = (
        scale2 * x_fp32 * exp(-(sx * sx))
        + 0.5 * erf_poly(scale1 * x_fp32)
        + 0.5
    )
    dx = dydx * dy
    return dx


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=_gelu_config)
@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    # 0.79788456 = math.sqrt(2 / math.pi)
    x2 = x_fp32 * x_fp32
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * x2))
    tanh2 = tanh_out * tanh_out
    dydx = 0.5 * x_fp32 * (
        (1 - tanh2) * (0.79788456 + 0.1070322243 * x2)
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