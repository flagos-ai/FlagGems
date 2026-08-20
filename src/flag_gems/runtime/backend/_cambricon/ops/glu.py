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

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b

    return result


@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ]
)
@triton.jit
def glu_backward_kernel(grad_output, a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    db = grad_output.to(tl.float32) * a * sigmoid_b * (1.0 - sigmoid_b)

    return da, db


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS_CAMBRICON GLU")
    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)

    return out


def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS_CAMBRICON GLU_BACKWARD")
    # Recreate a and b
    a, b = torch.chunk(self, 2, dim=dim)
    grad_input = torch.empty_like(self, memory_format=torch.contiguous_format)
    grad_a, grad_b = torch.chunk(grad_input, 2, dim=dim)
    glu_backward_kernel(grad_output, a, b, out0=grad_a, out1=grad_b)

    return grad_input
