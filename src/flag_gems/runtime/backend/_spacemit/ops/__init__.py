# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .abs import abs, abs_  # noqa: F401
from .add import add, add_  # noqa: F401
from .addmm import addmm  # noqa: F401
from .all import all, all_dim, all_dims  # noqa: F401
from .amax import amax  # noqa: F401
from .bitwise_and import (  # noqa: F401
    bitwise_and_scalar,
    bitwise_and_scalar_,
    bitwise_and_scalar_tensor,
    bitwise_and_tensor,
    bitwise_and_tensor_,
)
from .bitwise_not import bitwise_not, bitwise_not_  # noqa: F401
from .bitwise_or import (  # noqa: F401
    bitwise_or_scalar,
    bitwise_or_scalar_,
    bitwise_or_scalar_tensor,
    bitwise_or_tensor,
    bitwise_or_tensor_,
)
from .clamp import clamp, clamp_, clamp_tensor, clamp_tensor_  # noqa: F401
from .cos import cos, cos_  # noqa: F401
from .div import true_divide, true_divide_  # noqa: F401
from .eq import eq, eq_scalar  # noqa: F401
from .erf import erf, erf_  # noqa: F401
from .exp import exp, exp_  # noqa: F401

# flash_attention (and its attn_vfwmacc dependency) and mm require the
# smt_rvisa eDSL, which only exists on newer spine-triton builds; keep this
# backend importable on the older stack, where those ops fall back to the
# general implementations instead of breaking the whole vendor registration
try:
    from .flash_attention import (  # noqa: F401
        flash_attention,
        scaled_dot_product_attention,
        scaled_dot_product_flash_attention_for_cpu,
    )
except ImportError:
    pass
from .ge import ge, ge_scalar  # noqa: F401
from .gelu import gelu  # noqa: F401
from .gt import gt, gt_scalar  # noqa: F401
from .isinf import isinf  # noqa: F401
from .isnan import isnan  # noqa: F401
from .le import le, le_scalar  # noqa: F401
from .log_softmax import log_softmax  # noqa: F401
from .lt import lt, lt_scalar  # noqa: F401
from .max import max, max_dim  # noqa: F401
from .mean import mean, mean_dim  # noqa: F401
from .min import min, min_dim  # noqa: F401

try:
    from .mm import mm, mm_out  # noqa: F401
except ImportError:
    pass
from .mul import mul, mul_  # noqa: F401
from .mv import mv  # noqa: F401
from .relu import relu, relu_  # noqa: F401
from .rsqrt import rsqrt  # noqa: F401
from .sigmoid import sigmoid  # noqa: F401
from .silu import silu  # noqa: F401
from .sin import sin, sin_  # noqa: F401
from .softmax import softmax, softmax_out  # noqa: F401
from .sub import sub, sub_  # noqa: F401
from .sum import sum as sum  # noqa: F401
from .sum import sum_dim, sum_dim_out  # noqa: F401
from .tanh import tanh, tanh_  # noqa: F401
from .triu import triu  # noqa: F401

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
    "ge",
    "ge_scalar",
    "gelu",
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
    "scaled_dot_product_attention",
    "scaled_dot_product_flash_attention_for_cpu",
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
]

# drop entries whose module was unavailable on this stack (see the
# try/except guards above) so __all__ never names an undefined symbol
__all__ = [name for name in __all__ if name in globals()]
