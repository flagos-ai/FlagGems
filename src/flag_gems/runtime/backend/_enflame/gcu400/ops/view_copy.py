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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs); see ops/deg2rad.py for the grid.x limit
# rationale.
_GRID_SIZE = 48


@libentry()
@triton.jit
def view_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, vals, mask=mask)


def view_copy(x: torch.Tensor, size) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME VIEW_COPY")
    """Wrapper for aten::view_copy: copy of `x` with the specified shape."""
    # Handle SymInt[] - convert to tuple of ints
    if isinstance(size, torch.SymInt):
        size = (int(size),)
    elif isinstance(size, (list, tuple)):
        size = tuple(int(s) if isinstance(s, torch.SymInt) else s for s in size)

    n_elements = x.numel()

    # Handle -1 (infer this dimension)
    if -1 in size:
        if size.count(-1) > 1:
            raise RuntimeError(f"view_copy: only one dimension can be -1, got {size}")
        target_numel_except_minus1 = 1
        for s in size:
            if s != -1:
                target_numel_except_minus1 *= s
        inferred_dim = n_elements // target_numel_except_minus1
        size = tuple(inferred_dim if s == -1 else s for s in size)

    # Validate total number of elements matches
    target_numel = 1
    for s in size:
        target_numel *= s
    if n_elements != target_numel:
        raise RuntimeError(
            f"view_copy: cannot reshape tensor of size {n_elements} into shape {size}"
        )

    if n_elements == 0:
        return torch.empty(size, dtype=x.dtype, device=x.device)

    out = torch.empty(size, dtype=x.dtype, device=x.device)
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()

    block_size = 32768
    num_tile = triton.cdiv(n_elements, block_size)
    grid = (min(_GRID_SIZE, num_tile), 1, 1)
    with torch_device_fn.device(x.device.index):
        view_copy_kernel[grid](
            src,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=4,
            num_warps=4,
        )
    return out
