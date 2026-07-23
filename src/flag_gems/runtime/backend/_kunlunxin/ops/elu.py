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


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    x_fp32 = x.to(tl.float32)
    return tl.where(
        x_fp32 > 0,
        scale * input_scale * x_fp32,
        scale * alpha * (tl.exp(x_fp32 * input_scale) - 1),
    )


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def elu_backward_kernel(grad_output, x, alpha, scale, input_scale, is_result):
    x_fp32 = x.to(tl.float32)
    grad_pos = grad_output * scale * input_scale
    if is_result:
        grad_neg = grad_output * input_scale * (x_fp32 + scale * alpha)
    else:
        grad_neg = (
            grad_output * scale * alpha * input_scale * tl.exp(x_fp32 * input_scale)
        )

    return tl.where(x_fp32 > 0, grad_pos, grad_neg)


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS_KUNLUNXIN ELU")
    return elu_forward_kernel(A, alpha, scale, input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS_KUNLUNXIN ELU_")
    return elu_forward_kernel(A, alpha, scale, input_scale, out0=A)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS_KUNLUNXIN ELU_BACKWARD")
    grad_input = elu_backward_kernel(
        grad_output, self_or_result, alpha, scale, input_scale, is_result
    )
    return grad_input
