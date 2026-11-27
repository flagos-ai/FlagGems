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

"""GCU400 specialisation of the fused ``weight_norm`` operator.

The common implementation lives in ``flag_gems.fused.weight_norm`` and binds
``weight_norm_interface`` from the generic ``flag_gems.ops.weightnorm`` module
at import time, so the GCU400 kernels in ``.weightnorm`` never reach the fused
entry point.  This module re-creates the autograd wrapper on top of the GCU400
interface; the rarely used middle-dimension path is reused from the common
module instead of being duplicated.
"""

import logging

import torch

from flag_gems.fused.weight_norm import (
    weight_norm_except_dim,
    weight_norm_except_dim_backward,
)

from .weightnorm import weight_norm_interface, weight_norm_interface_backward

logger = logging.getLogger(__name__)


class WeightNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim=0):
        logger.debug("GEMS_ENFLAME WEIGHT_NORM")
        dim = dim % v.ndim
        can_use_fused = dim == 0 or dim == v.ndim - 1
        if can_use_fused:
            output, norm = weight_norm_interface(v, g, dim)
        else:
            output, norm = weight_norm_except_dim(v, g, dim)
        ctx.save_for_backward(v, g, norm)
        ctx.dim = dim
        ctx.can_use_fused = can_use_fused
        return output

    @staticmethod
    def backward(ctx, grad):
        logger.debug("GEMS_ENFLAME WEIGHT_NORM_BACKWARD")
        v, g, norm = ctx.saved_tensors
        dim = ctx.dim
        if ctx.can_use_fused:
            v_grad, g_grad = weight_norm_interface_backward(grad, v, g, norm, dim)
        else:
            v_grad, g_grad = weight_norm_except_dim_backward(grad, v, g, norm, dim)
        return v_grad, g_grad, None


def weight_norm(v, g, dim=0):
    return WeightNorm.apply(v, g, dim)
