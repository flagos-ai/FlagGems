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

from ..utils.pointwise_dynamic import pointwise_dynamic
from .scalar_tensor import scalar_tensor

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hypot_inplace_kernel(x, y):
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    return tl.sqrt(x * x + y * y)


def hypot_(self: torch.Tensor, other) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN HYPOT_")
    if not isinstance(other, torch.Tensor):
        other = scalar_tensor(other, dtype=self.dtype, device=self.device)
    hypot_inplace_kernel(self, other, out0=self)
    return self
