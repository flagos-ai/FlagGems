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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def silu_backward_kernel(x, dy):
    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    # `div_rn` from flag_gems.utils.triton_lang_extension lowers to an extern
    # libdevice call (tt.extern_elementwise). The GCU300 compiler does not
    # legalize it here, and the resulting kernel returns values that do not
    # match aten at all (fp16 input [-2, -0.5, 0, 0.5, 2] with grad 1 gives
    # [-0.0, 0.0, 1.0, 1.0, 1.0] instead of
    # [-0.0908, 0.2600, 0.5, 0.7397, 1.0908]). The forward kernel in this same
    # file uses tl.fdiv for the equivalent 1/(1+exp(-x)) and is accurate, so the
    # backward uses the same primitive.
    sigma = tl.fdiv(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx


def silu(self):
    logger.debug("GEMS_ENFLAME SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS_ENFLAME SILU_BACKWARD")
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input


def silu_(A):
    logger.debug("GEMS_ENFLAME SILU_")
    out = silu_forward(A, out0=A)
    return out
