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

# Ascend-specific override for aten::isfinite.
#
# CANN 9.1.1 aclnnIsFinite crashes (error 507035, vector core exception) when
# given a bfloat16 tensor.  This Triton implementation avoids that path entirely
# by using the pure-arithmetic IEEE identity  (x - x) == 0.0, which is exactly
# 0 for every finite value and NaN for ±inf and NaN.  bfloat16/float16 inputs
# are upcast to float32 before the subtraction so the vector pipe always sees a
# type it can handle, and the bool result is returned directly.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _isfinite_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # IEEE identity: x - x == 0 iff x is finite; NaN/±inf give NaN != 0.
    result = (x - x) == 0.0
    tl.store(out_ptr + offs, result, mask=mask)


def isfinite(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ASCEND ISFINITE")

    if not A.is_floating_point():
        # Integer and bool tensors never contain inf/nan.
        return torch.ones(A.shape, dtype=torch.bool, device=A.device)

    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    n = A.numel()
    if n == 0:
        return out

    a_flat = A.contiguous().reshape(-1)
    out_flat = out.reshape(-1)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    with torch_device_fn.device(A.device):
        _isfinite_kernel[grid](a_flat, out_flat, n, BLOCK=BLOCK)

    return out
