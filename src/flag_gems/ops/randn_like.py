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
import math

import torch
import triton

from flag_gems.ops.randn import randn_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
UNROLL = 4


def randn_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS RANDN_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype, memory_format=memory_format)
    if out.numel() == 0:
        return out

    is_complex = out.is_complex()
    components = torch.view_as_real(out) if is_complex else out
    buffer = components
    if is_complex and not components.is_contiguous():
        buffer = torch.empty_like(components, memory_format=torch.contiguous_format)

    # Each component of a standard complex normal has variance 1/2.
    scale = math.sqrt(0.5) if is_complex else 1.0
    N = buffer.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(out.device):
        randn_kernel[grid_fn](buffer, N, philox_seed, philox_offset, SCALE=scale)
    if buffer is not components:
        components.copy_(buffer)
    return out
