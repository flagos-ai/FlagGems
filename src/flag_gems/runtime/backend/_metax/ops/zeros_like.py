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

from flag_gems.runtime import torch_device_fn

from .zeros import zeros_kernel

logger = logging.getLogger(__name__)


def zeros_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_METAX ZEROS_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out
