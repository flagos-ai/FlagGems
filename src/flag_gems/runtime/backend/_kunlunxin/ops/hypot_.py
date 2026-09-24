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
from .scalar_tensor import scalar_tensor  # [call-fix 2026-09-24] in-tree direct call

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def hypot_inplace_kernel(x, y):
    # Compute in fp32 for stability (matches generic hypot_); the launcher
    # stores the result back in self's dtype.
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    return tl.sqrt(x * x + y * y)


def hypot_(self: torch.Tensor, other) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN HYPOT_")
    # The generic hypot_ materializes the broadcast other via
    # torch.broadcast_to(...).contiguous(), which hits the XPU copy_
    # "invalid device function" failure on strided/broadcast sources.
    # pointwise_dynamic reads the broadcast operand in-kernel instead.
    # [call-fix 2026-09-24] build the scalar operand with the backend's own
    # scalar_tensor kernel instead of the torch.tensor constructor.
    if not isinstance(other, torch.Tensor):
        other = scalar_tensor(other, dtype=self.dtype, device=self.device)
    hypot_inplace_kernel(self, other, out0=self)
    return self
