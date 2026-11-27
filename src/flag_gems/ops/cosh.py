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
def cosh_func(x):
    x = x.to(tl.float32)
    return 0.5 * (tl.exp(x) + tl.exp(-x))


class CoshFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return cosh_func(A)

    @staticmethod
    def backward(ctx, grad_output):
        # ``aten::cosh`` has no standalone backward operator, so the gradient
        # runs a Triton kernel from ``flag_gems.ops.cosh_backward``.
        (A,) = ctx.saved_tensors
        from flag_gems.ops.cosh_backward import cosh_backward

        return cosh_backward(grad_output, A)


def cosh(A):
    logger.debug("GEMS COSH")
    if torch.is_grad_enabled() and A.requires_grad:
        return CoshFunction.apply(A)
    return cosh_func(A)


def cosh_(A):
    logger.debug("GEMS COSH_")
    return cosh_func(A, out0=A)


def cosh_out(A, out):
    logger.debug("GEMS COSH_OUT")
    return cosh_func(A, out0=out)
