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

from flag_gems.ops.logaddexp2_backward import (
    _can_use_backward_fast_path,
    _launch_backward_fast_path,
    _unbroadcast,
)
from flag_gems.ops.logaddexp2_backward import logaddexp2_backward as logaddexp2_backward
from flag_gems.ops.logaddexp2_backward import (
    logaddexp2_backward_func,
    logaddexp2_backward_other_func,
    logaddexp2_backward_self_func,
)
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp2_func(x, y):
    # log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
    # `delta` is NaN when x and y are equal infinities (inf - inf); the result
    # is then m, e.g. logaddexp2(inf, inf) = inf, logaddexp2(-inf, -inf) = -inf.
    res = tl.where(delta != delta, m, res)
    # Genuine NaN inputs must still propagate NaN.
    is_nan = (x_f32 != x_f32) | (y_f32 != y_f32)
    return tl.where(is_nan, float("nan"), res)


# ATen derives the two input gradients of ``logaddexp2`` from the base-2
# exponentials of the forward definition:
#
#   d/dx log2(2**x + 2**y) = 2**x / (2**x + 2**y) = 1 / (1 + 2**(y - x))
#   d/dy log2(2**x + 2**y) = 2**y / (2**x + 2**y) = 1 / (1 + 2**(x - y))
#
# The weights are base 2 (not the natural-base weights of ``logaddexp``). ATen
# evaluates ``grad / (1 + pow(2, other - self))`` as a chain of elementwise ATen
# ops, so the difference, the ``pow(2, .)`` result, the ``1 + .`` denominator
# and the quotient are each rounded to the element dtype. Keeping that same
# rounding order is what makes the kernels match native autograd exactly at
# FP16 (rounding only the final half-precision value is off by up to 4 ULP);
# fusing the whole pair into one launch costs no extra memory traffic.
#
# Equal infinities make the exponent ``inf - inf``, i.e. NaN, exactly like the
# native formula; finite extremes saturate to the 0/1 weights that ``2**``
# underflow/overflow produces natively.


class Logaddexp2Function(torch.autograd.Function):
    """``logaddexp2`` with a fused Triton backward for both inputs."""

    @staticmethod
    def forward(ctx, self, other):
        ctx.save_for_backward(self, other)
        return logaddexp2_func(self, other)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS LOGADDEXP2_BACKWARD")
        self_, other_ = ctx.saved_tensors
        need_self, need_other = ctx.needs_input_grad
        grad_self = None
        grad_other = None
        if _can_use_backward_fast_path(grad_output, self_, other_):
            grad_self, grad_other = _launch_backward_fast_path(
                grad_output, self_, other_, need_self, need_other
            )
        elif need_self and need_other:
            grad_self, grad_other = logaddexp2_backward_func(grad_output, self_, other_)
        elif need_self:
            grad_self = logaddexp2_backward_self_func(grad_output, self_, other_)
        elif need_other:
            grad_other = logaddexp2_backward_other_func(grad_output, self_, other_)
        if grad_self is not None:
            grad_self = _unbroadcast(grad_self, self_.shape).to(self_.dtype)
        if grad_other is not None:
            grad_other = _unbroadcast(grad_other, other_.shape).to(other_.dtype)
        return grad_self, grad_other


def logaddexp2(self, other):
    logger.debug("GEMS LOGADDEXP2")
    if torch.is_grad_enabled() and (self.requires_grad or other.requires_grad):
        return Logaddexp2Function.apply(self, other)
    return logaddexp2_func(self, other)


def logaddexp2_out(self, other, out):
    logger.debug("GEMS LOGADDEXP2_OUT")
    logaddexp2_func(self, other, out0=out)
    return out
