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

from flag_gems.runtime import torch_device_fn

from .full import check_dtype, full_kernel

logger = logging.getLogger(__name__)


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logger.debug("GEMS_ASCEND FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    fill_value = check_dtype(fill_value, dtype, device)
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), 2048)
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        full_kernel[grid_fn](
            out,
            N,
            fill_value,
            BLOCK_SIZE=BLOCK_SIZE,
            SUBBLOCK_SIZE=SUBBLOCK_SIZE,
        )
    return out
