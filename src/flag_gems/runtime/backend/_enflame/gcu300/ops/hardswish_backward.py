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
def hardswish_backward_func(grad_output, self):
    # hardswish: y = x * clamp(x + 3, 0, 6) / 6
    # gradient:
    #   x <= -3:     0
    #   -3 < x < 3:  (2x + 3) / 6
    #   x >= 3:      1
    #
    # The two inner ranges are selected with nested `tl.where` instead of
    # `(x > -3) & (x < 3)`: on this backend combining two float comparisons
    # with `&` (or `|`, or `~`) makes the codegen emit a vector `fabs` symbol
    # that ld.lld cannot relocate
    # ("relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol
    # fabs(float vector[128])"), so the kernel fails to link. Two selects are
    # also fewer operations than mask materialise + and + select.
    grad_output_fp32 = grad_output.to(tl.float32)
    self_fp32 = self.to(tl.float32)
    grad_at_self = (2.0 * self_fp32 + 3.0) / 6.0
    grad = tl.where(
        self_fp32 <= -3.0,
        0.0,
        tl.where(self_fp32 < 3.0, grad_at_self, 1.0),
    )
    result = grad_output_fp32 * grad
    return result.to(grad_output.dtype)


def hardswish_backward(grad_output, self):
    logger.debug("GEMS_ENFLAME HARDSWISH_BACKWARD")
    return hardswish_backward_func(grad_output, self)
