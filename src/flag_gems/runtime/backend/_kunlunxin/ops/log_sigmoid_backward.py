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
import os

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# The flagtree compiler shipped in the CI image (flagtree 0.7.0+xpu3.6, an
# older triton fork pinned in backends.yaml) cannot legalize the fused
# triton_xpu.vsigmoidf op that tl.sigmoid(0.0 - x) lowers to
# (ConvertTritonXPUToLLVM marks it illegal), so it fails to compile there. Use
# the explicit 1/(1+exp(x)) = sigmoid(-x) form on flagtree; the newer triton
# build legalizes vsigmoidf and keeps the fused single-op sigmoid.
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel(grad_output, self):
    # Recompute the derivative from `self` instead of consuming the buffer.
    # ATen's forward contract defines buffer == exp(-|self|), so both formulas
    # are mathematically identical, but the buffer produced by the vendor
    # log_sigmoid_forward on XPU cannot be trusted (the autograd test exercises
    # a full-size garbage buffer returned by the native forward).
    #
    # The previous form `where(x < 0, 1, z) * (1 / (1 + z))` with z = exp(-|x|)
    # lowered to a full-precision `arith.divf` + `select` (XPU division is
    # ~1.9x the cost of a fused sigmoid; profiled kernel = 92% of op time).
    # The XPU backend's Vectorize pass pattern-matches
    # `1 / (1 + exp(-y))` (with exp's input an `0 - y` subtraction) into a
    # single native `vsigmoidf` vector op, so writing sigmoid(-x) explicitly
    # as `tl.sigmoid(0.0 - x)` removes the division and the select entirely.
    self_fp32 = self.to(tl.float32)
    derivative = tl.sigmoid(0.0 - self_fp32)
    return grad_output * derivative


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_kernel_flagtree(grad_output, self):
    # flagtree-safe sigmoid(-x) = 1 / (1 + e^x). Mathematically identical to
    # tl.sigmoid(0.0 - x) above, but never fuses into vsigmoidf (flagtree's
    # ConvertTritonXPUToLLVM cannot legalize that op), so it compiles there.
    self_fp32 = self.to(tl.float32)
    derivative = 1.0 / (1.0 + tl.exp(self_fp32))
    return grad_output * derivative


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD")

    del buffer
    kernel = (
        log_sigmoid_backward_kernel_flagtree
        if _IS_FLAGTREE
        else log_sigmoid_backward_kernel
    )
    return kernel(grad_output, self)


def log_sigmoid_backward_out(grad_output, self, buffer, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN LOG_SIGMOID_BACKWARD_OUT")

    # Always go through the tuned pointwise kernel (out0=grad_input writes in
    # place). The previous dedicated contiguous kernel launched one 1024-element
    # program per tile (16M elements -> 16384 tiny programs, ~94 GB/s) and was
    # measured far slower than the pointwise path for every benchmark shape.
    del buffer
    kernel = (
        log_sigmoid_backward_kernel_flagtree
        if _IS_FLAGTREE
        else log_sigmoid_backward_kernel
    )
    return kernel(grad_output, self, out0=grad_input)
