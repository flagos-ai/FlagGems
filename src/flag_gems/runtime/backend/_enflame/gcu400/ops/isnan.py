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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_ISNAN = tl_extra_shim.isnan
_GRID_SIZE = 96
_BLOCK_SIZE = 8192
_NUM_WARPS = 4
_NUM_STAGES = 3


@libentry()
@triton.jit
def isnan_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tile = (n_elements + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tile, GRID, num_stages=STAGES):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, _ISNAN(x.to(tl.float32)), mask=mask)


def isnan(A):
    logger.debug("GEMS_ENFLAME ISNAN")
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    n_elements = A.numel()
    if n_elements == 0:
        return out
    x_contig = A.contiguous()
    grid = (min(_GRID_SIZE, triton.cdiv(n_elements, _BLOCK_SIZE)), 1, 1)
    with torch_device_fn.device(x_contig.device):
        isnan_kernel[grid](
            x_contig.view(-1),
            out.view(-1),
            n_elements,
            BLOCK=_BLOCK_SIZE,
            GRID=min(_GRID_SIZE, triton.cdiv(n_elements, _BLOCK_SIZE)),
            STAGES=_NUM_STAGES,
            num_warps=_NUM_WARPS,
        )
    if not A.is_contiguous():
        out = out.contiguous()
    return out
