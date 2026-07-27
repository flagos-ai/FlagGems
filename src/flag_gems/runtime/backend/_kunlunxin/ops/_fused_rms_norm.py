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

import math

import torch

from .rms_norm import rms_norm_backward, rms_norm_forward


class _FusedRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y, inv_rms

    @staticmethod
    def backward(ctx, dy, d_inv_rms):
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, inv_rms, ctx.normalized_shape, weight, ctx.eps
        )
        return dx, None, dw, None


def _fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    if weight is not None:
        return _FusedRmsNorm.apply(x, normalized_shape, weight, eps)

    n = math.prod(normalized_shape)
    unit_weight = torch.ones(n, dtype=x.dtype, device=x.device)
    return rms_norm_forward(x, normalized_shape, unit_weight, eps)
