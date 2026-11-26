# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.runtime import torch_device_fn

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@triton.jit
def _resize_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_id in range(pid, total_blocks, num_jobs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, vals, mask=mask)


def resize(inp: torch.Tensor, size, memory_format=None):
    logger.debug("GEMS_CAMBRICON RESIZE")

    if not isinstance(size, tuple):
        size = tuple(size)

    # Calculate total number of elements
    total_elements = 1
    for dim in size:
        total_elements *= dim

    out = torch.empty(size, device=inp.device, dtype=inp.dtype)
    if inp.numel() == 0 or out.numel() == 0:
        return out

    n_elements = min(inp.numel(), total_elements)
    grid = lambda meta: (
        min(triton.cdiv(n_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(inp.device):
        _resize_kernel[grid](inp, out, n_elements, BLOCK_SIZE=1024)

    return out


def resize_(inp: torch.Tensor, size, memory_format=None):
    logger.debug("GEMS_CAMBRICON RESIZE_")

    if not isinstance(size, tuple):
        size = tuple(size)

    inp.set_(inp.untyped_storage(), 0, size)
    return inp
