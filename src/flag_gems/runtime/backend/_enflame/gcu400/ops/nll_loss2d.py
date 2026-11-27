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

from .nllloss import _contiguous_if_needed, nll_loss2d_backward, nll_loss2d_forward

logger = logging.getLogger(__name__)


class _NLLLoss2dFunction(torch.autograd.Function):
    # Triton kernels only produce numbers, without autograd metadata, so wrap them
    # in an autograd.Function to keep gradients flowing downstream.
    @staticmethod
    def forward(ctx, self, target, weight, reduction, ignore_index):
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        output, total_weight = nll_loss2d_forward(
            self, target, weight, reduction, ignore_index
        )
        ctx.save_for_backward(self, target, weight, total_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        self, target, weight, total_weight = ctx.saved_tensors
        grad_input = nll_loss2d_backward(
            grad_output,
            self,
            target,
            weight,
            ctx.reduction,
            ctx.ignore_index,
            total_weight,
        )
        return grad_input, None, None, None, None


def nll_loss2d(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS_ENFLAME NLL_LOSS2D")

    assert self.ndim == 4, "nll_loss2d: expected 4D input (N, C, H, W)"
    assert self.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"nll_loss2d: unsupported dtype {self.dtype}"

    reduction_val = (
        int(reduction.item()) if isinstance(reduction, torch.Tensor) else int(reduction)
    )
    ignore_index_val = (
        int(ignore_index.item())
        if isinstance(ignore_index, torch.Tensor)
        else int(ignore_index)
    )
    self = _contiguous_if_needed(self)
    target = _contiguous_if_needed(target)
    if weight is not None:
        weight = _contiguous_if_needed(weight)

    return _NLLLoss2dFunction.apply(
        self, target, weight, reduction_val, ignore_index_val
    )
