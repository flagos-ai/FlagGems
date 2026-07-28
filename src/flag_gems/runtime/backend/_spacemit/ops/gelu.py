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

import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh
geluTanh = tl_extra_shim.gelu_tanh
geluNone = tl_extra_shim.gelu_none


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    # scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    # output = 0.5 * x * (1 + erf(x * scale))
    output = geluNone(x.to(tl.float32))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        # 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2))))
        geluTanh(x.to(tl.float32))
    )
    return output


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2))
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
    tanh_out = tanh(0.79788456 * x_fp32 * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x_fp32 * (
        (1 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))
    ) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx


def gelu(A, *, approximate="none"):
    logger.debug("GEMS_SPACEMIT GELU")
    if approximate == "tanh":
        out = gelu_tanh(A)
    else:
        out = gelu_none(A)
    return out


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_SPACEMIT GELU_BACKWARD")
    if approximate == "tanh":
        in_grad = gelu_backward_tanh(self, grad_output)
    else:
        in_grad = gelu_backward_none(self, grad_output)
    return in_grad


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_SPACEMIT GELU_")
    if approximate == "tanh":
        out = gelu_tanh(A, out0=A)
    else:
        out = gelu_none(A, out0=A)
    return out
