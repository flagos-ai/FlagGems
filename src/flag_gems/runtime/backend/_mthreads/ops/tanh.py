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

# This custom op requires musa device capability >= 31.
# We determine whether to enable this op by distinguish the op registration for different arch.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

pow = tl_extra_shim.pow
fast_tanh = tl_extra_shim.fast_tanh

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward(x):
    return fast_tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward(y, dy):
    return dy * (1.0 - pow(y.to(tl.float32), 2))


class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("GEMS_MTHREADS TANH_FORWARD")
        if A.requires_grad is True:
            out = tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = tanh_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_MTHREADS TANH_BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = tanh_backward(out, out_grad)
        return in_grad


def tanh(A):
    return Tanh.apply(A)
