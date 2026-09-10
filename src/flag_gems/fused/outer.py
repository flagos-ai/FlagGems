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

from flag_gems.ops import mul, mv

logger = logging.getLogger(__name__)


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("GEMS OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        inp1 = inp[:, None]
        weight1 = weight[None, :]
        inp1 = inp1.contiguous()
        weight1 = weight1.contiguous()
        out = mul(inp1, weight1)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS OUTER VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_grad = mv(out_grad, weight)
        weight_grad = mv(out_grad.t(), inp)

        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)
