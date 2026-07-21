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


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS_ENFLAME GLU")
    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)

    return out
