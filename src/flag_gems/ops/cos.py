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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


class CosFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return cos_func(A)

    @staticmethod
    def backward(ctx, grad_output):
        # ``aten::cos`` has no standalone backward operator, so the gradient
        # runs a Triton kernel from ``flag_gems.ops.cos_backward``.
        (A,) = ctx.saved_tensors
        from flag_gems.ops.cos_backward import cos_backward

        return cos_backward(grad_output, A)


def cos(A):
    logger.debug("GEMS COS")
    if torch.is_grad_enabled() and A.requires_grad:
        return CosFunction.apply(A)
    return cos_func(A)


def cos_(A):
    logger.debug("GEMS COS_")
    cos_func(A, out0=A)
    return A
