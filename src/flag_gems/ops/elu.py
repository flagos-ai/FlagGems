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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def elu_forward_kernel(x, alpha, scale, input_scale):
    return tl.where(
        x > 0,
        scale * x,
        scale * alpha * (tl.exp(x.to(tl.float32) * input_scale) - 1),
    )


@pointwise_dynamic(
    is_tensor=[True, False, False, False, True], promotion_methods=[(0, 4, "DEFAULT")]
)
@triton.jit
def elu_backward_kernel_with_self(grad_output, alpha, scale, input_scale, x):
    x_fp32 = x.to(tl.float32)
    grad_input = tl.where(
        x > 0,
        grad_output * scale,
        grad_output * (scale * alpha * tl.exp(x_fp32 * input_scale) * input_scale),
    )
    return grad_input


@pointwise_dynamic(
    is_tensor=[True, False, False, False, True], promotion_methods=[(0, 4, "DEFAULT")]
)
@triton.jit
def elu_backward_kernel_with_result(grad_output, alpha, scale, input_scale, y):
    grad_input = tl.where(
        y > 0,
        grad_output * scale,
        grad_output * ((y + scale * alpha) * input_scale),
    )
    return grad_input


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU")
    return elu_forward_kernel(A, alpha, scale, input_scale)


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU_")
    return elu_forward_kernel(A, alpha, scale, input_scale, out0=A)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS ELU_BACKWARD")
    if is_result:
        return elu_backward_kernel_with_result(
            grad_output, alpha, scale, input_scale, self_or_result
        )
    else:
        return elu_backward_kernel_with_self(
            grad_output, alpha, scale, input_scale, self_or_result
        )


@libentry()
@triton.jit
def elu_backward_grad_input_kernel(
    grad_output_ptr,
    self_or_result_ptr,
    grad_input_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    IS_RESULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    src = tl.load(self_or_result_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if IS_RESULT:
        # `src` holds the forward output y = elu(x); recover the slack slope.
        grad_input = tl.where(
            src <= 0,
            grad_output * input_scale * (src + alpha * scale),
            grad_output * scale,
        )
    else:
        # `src` holds the input x.
        grad_input = tl.where(
            src <= 0,
            grad_output * input_scale * alpha * scale * tl.exp(src * input_scale),
            grad_output * scale,
        )

    tl.store(
        grad_input_ptr + offsets,
        grad_input.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def elu_backward_out(
    grad_output, alpha, scale, input_scale, is_result, self_or_result, *, grad_input
):
    logger.debug("GEMS ELU_BACKWARD_OUT")
    n_elements = grad_output.numel()
    if n_elements == 0:
        return grad_input

    if (
        grad_output.is_contiguous()
        and self_or_result.is_contiguous()
        and grad_input.is_contiguous()
    ):
        block_size = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(grad_output.device):
            elu_backward_grad_input_kernel[grid](
                grad_output,
                self_or_result,
                grad_input,
                n_elements,
                float(alpha),
                float(scale),
                float(input_scale),
                IS_RESULT=bool(is_result),
                BLOCK_SIZE=block_size,
            )
        return grad_input

    # Strided fallback via the pointwise_dynamic wrapper.
    if is_result:
        return elu_backward_grad_input_kernel_with_result(
            grad_output,
            alpha,
            scale,
            input_scale,
            self_or_result,
            out0=grad_input,
        )
    else:
        return elu_backward_grad_input_kernel_with_self(
            grad_output,
            alpha,
            scale,
            input_scale,
            self_or_result,
            out0=grad_input,
        )


@pointwise_dynamic(
    is_tensor=[True, False, False, False, True],
    promotion_methods=[(0, 4, "DEFAULT")],
)
@triton.jit
def elu_backward_grad_input_kernel_with_self(grad_output, alpha, scale, input_scale, x):
    x_fp32 = x.to(tl.float32)
    grad_input = tl.where(
        x > 0,
        grad_output * scale * input_scale,
        grad_output * (scale * alpha * tl.exp(x_fp32 * input_scale) * input_scale),
    )
    return grad_input


@pointwise_dynamic(
    is_tensor=[True, False, False, False, True],
    promotion_methods=[(0, 4, "DEFAULT")],
)
@triton.jit
def elu_backward_grad_input_kernel_with_result(
    grad_output, alpha, scale, input_scale, y
):
    grad_input = tl.where(
        y > 0,
        grad_output * scale * input_scale,
        grad_output * ((y + scale * alpha) * input_scale),
    )
    return grad_input
