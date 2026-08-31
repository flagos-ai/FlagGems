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

from flag_gems.ops.add import add as generic_add
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def _can_use_fast_path(x, y, alpha):
    return (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and isinstance(alpha, (int, float))
        and not isinstance(alpha, bool)
        and alpha == 1
        and x.device == y.device
        and x.dtype == y.dtype
        and x.dtype in _FAST_DTYPES
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
    )


def add(x, y, *, alpha=1):
    if not _can_use_fast_path(x, y, alpha):
        return generic_add(x, y, alpha=alpha)

    logger.debug("GEMS ADD NVIDIA FAST PATH")
    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(x.device):
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
