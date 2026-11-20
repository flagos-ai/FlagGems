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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def expm1_func(x):
    return tl.exp(x.to(tl.float32)) - 1.0


@triton.jit
def _expm1_grad(grad_output, output):
    # ATen rounds output + 1 to the output dtype before multiplying.
    scale = (output.to(tl.float32) + 1.0).to(output.dtype)
    return grad_output * scale


_expm1_backward_strided = pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])(
    _expm1_grad
)


@triton.jit
def _expm1_backward_kernel(
    grad_output, output, grad_input, n, BLOCK: tl.constexpr, LARGE: tl.constexpr
):
    pid = tl.program_id(0)
    if LARGE:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    grad = tl.load(grad_output + offsets, mask, other=0)
    value = tl.load(output + offsets, mask, other=0)
    tl.store(grad_input + offsets, _expm1_grad(grad, value), mask)


class Expm1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = expm1_func(x)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        if not (output.is_contiguous() and grad_output.is_contiguous()):
            return _expm1_backward_strided(grad_output, output)
        grad_input = torch.empty_like(output)
        n = output.numel()
        if n:
            block_size = 512
            with torch_device_fn.device(output.device):
                _expm1_backward_kernel[(triton.cdiv(n, block_size),)](
                    grad_output,
                    output,
                    grad_input,
                    n,
                    BLOCK=block_size,
                    LARGE=n > 2147483647,
                )
        return grad_input


def expm1(A):
    logger.debug("GEMS EXPM1")
    if torch.is_grad_enabled() and A.requires_grad:
        return Expm1Function.apply(A)
    return expm1_func(A)


def expm1_(A):
    logger.debug("GEMS EXPM1_")
    return expm1_func(A, out0=A)


# expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def expm1_out(A, out):
    logger.debug("GEMS EXPM1_OUT")
    return expm1_func(A, out0=out)
