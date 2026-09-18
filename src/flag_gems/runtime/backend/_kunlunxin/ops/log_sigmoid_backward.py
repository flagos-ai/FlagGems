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


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    # Recompute the derivative from `self` instead of consuming the buffer.
    # ATen's forward contract defines buffer == exp(-|self|), so both formulas
    # are mathematically identical, but the buffer produced by the vendor
    # log_sigmoid_forward on XPU cannot be trusted (the autograd test exercises
    # a full-size garbage buffer returned by the native forward).
    #
    # The two-branch form `where(x < 0, 1 / (1 + z), z / (1 + z))` evaluates
    # BOTH divisions (SIMT), and XPU division is expensive (~150us per 16M
    # fp32 division over the mul floor). Hoisting the shared denominator into
    # a single reciprocal is mathematically identical:
    #   sigmoid(-x) = where(x < 0, 1, z) * (1 / (1 + z)),  z = exp(-|x|)
    self_fp32 = self.to(tl.float32)
    z = tl.exp(-tl.abs(self_fp32))
    r = 1.0 / (1.0 + z)
    derivative = tl.where(self_fp32 < 0.0, 1.0, z) * r
    return grad_output * derivative


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID BACKWARD")

    del buffer
    return log_sigmoid_backward_kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID BACKWARD OUT")

    # Always go through the tuned pointwise kernel (out0=grad_input writes in
    # place). The previous dedicated contiguous kernel launched one 1024-element
    # program per tile (16M elements -> 16384 tiny programs, ~94 GB/s) and was
    # measured far slower than the pointwise path for every benchmark shape.
    del buffer
    return log_sigmoid_backward_kernel(grad_output, self, out0=grad_input)
