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

from flag_gems.ops.log1p_backward import log1p_backward
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def log1p_func(x):
    return tl.log(1.0 + x.to(tl.float32)).to(x.dtype)


class Log1pFunction(torch.autograd.Function):
    """Autograd wrapper that gives log1p a fused FlagGems backward.

    ATen declares ``aten::log1p`` CompositeImplicitAutograd with a generated
    ``Log1PBackward0`` formula, so serving its CUDA implementation from
    FlagGems alone leaves the backward to ATen's composite ``1 + x`` add
    followed by a division.  Saving ``x`` and launching the fused Triton kernel
    keeps the whole autograd path inside FlagGems.  The registration pairs this
    wrapper with an Autograd fallthrough key, matching the ``clone`` and
    ``adaptive_avg_pool3d`` precedent.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return log1p_func(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return log1p_backward(grad_output, x)


def log1p(A):
    logger.debug("GEMS LOG1P")
    if torch.is_grad_enabled() and A.requires_grad:
        return Log1pFunction.apply(A)
    return log1p_func(A)


def log1p_out(A, out):
    logger.debug("GEMS LOG1P_OUT")
    return log1p_func(A, out0=out)


log1p._flag_gems_autograd_impl = torch.library.fallthrough_kernel
