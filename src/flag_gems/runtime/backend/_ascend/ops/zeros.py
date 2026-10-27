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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(__name__)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = ext.program_id(axis=0)

    for sub_block_start_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        sub_offset = (
            pid * BLOCK_SIZE + sub_block_start_idx + tl.arange(0, BLOCK_SIZE_SUB)
        )
        mask = sub_offset < n_elements
        tl.store(output_ptr + sub_offset, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (max(triton.cdiv(N, meta["BLOCK_SIZE"]), 1),)
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=20480, BLOCK_SIZE_SUB=1024)
    return out
