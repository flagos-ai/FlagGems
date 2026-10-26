from .abs import abs, abs_
from .add import add, add_
from .addmm import addmm
from .all import all, all_dim, all_dims
from .amax import amax
from .bitwise_and import (
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_not import bitwise_not, bitwise_not_
from .bitwise_or import (
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_
from .cos import cos, cos_
from .div import true_divide, true_divide_
from .eq import eq, eq_scalar
from .erf import erf, erf_
from .exp import exp, exp_
# flash_attention (and its attn_vfwmacc dependency) and mm require the
# smt_rvisa eDSL, which only exists on newer spine-triton builds; keep this
# backend importable on the older stack, where those ops fall back to the
# general implementations instead of breaking the whole vendor registration
try:
    from .flash_attention import (
        flash_attention,
        scaled_dot_product_attention,
        scaled_dot_product_flash_attention_for_cpu,
    )
except ImportError:
    pass
from .gelu import gelu
from .ge import ge, ge_scalar
from .gt import gt, gt_scalar
from .isinf import isinf
from .isnan import isnan
from .le import le, le_scalar
from .log_softmax import log_softmax
from .lt import lt, lt_scalar
from .max import max, max_dim
from .mean import mean, mean_dim
from .min import min, min_dim
try:
    from .mm import mm, mm_out
except ImportError:
    pass
from .mul import mul, mul_
from .mv import mv
from .relu import relu, relu_
from .rsqrt import rsqrt
from .sigmoid import sigmoid
from .silu import silu
from .sin import sin, sin_
from .softmax import softmax, softmax_out
from .sub import sub, sub_
from .sum import sum as sum
from .sum import sum_dim, sum_dim_out
from .tanh import tanh, tanh_
from .triu import triu

__all__ = [
    "abs",
    "abs_",
    "add",
    "add_",
    "addmm",
    "all",
    "all_dim",
    "all_dims",
    "amax",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_not",
    "bitwise_not_",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "clamp",
    "clamp_",
    "clamp_tensor",
    "clamp_tensor_",
    "cos",
    "cos_",
    "eq",
    "eq_scalar",
    "erf",
    "erf_",
    "exp",
    "exp_",
    "flash_attention",
    "gelu",
    "ge",
    "ge_scalar",
    "gt",
    "gt_scalar",
    "isinf",
    "isnan",
    "le",
    "le_scalar",
    "log_softmax",
    "lt",
    "lt_scalar",
    "max",
    "max_dim",
    "mean",
    "mean_dim",
    "min",
    "min_dim",
    "mm",
    "mm_out",
    "mul",
    "mul_",
    "mv",
    "relu",
    "relu_",
    "rsqrt",
    "sigmoid",
    "silu",
    "sin",
    "sin_",
    "softmax",
    "softmax_out",
    "sub",
    "sub_",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "tanh",
    "tanh_",
    "triu",
    "scaled_dot_product_attention",
    "scaled_dot_product_flash_attention_for_cpu",
]

# drop entries whose module was unavailable on this stack (see the
# try/except guards above) so __all__ never names an undefined symbol
__all__ = [name for name in __all__ if name in globals()]
