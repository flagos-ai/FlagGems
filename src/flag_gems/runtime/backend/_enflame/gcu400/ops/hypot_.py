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
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400's grid.x limit is far below the block count needed for large shapes. Cap CTA count at
# 2x SIP and cover the remaining blocks with a grid-stride loop to avoid out of resource.
NUM_SIPS = 24
GRID_CAP = NUM_SIPS * 2
SMALL_TILE_THRESHOLD = 262144
LARGE_BLOCK_SIZE = 32768
SMALL_WARPS = 4
LARGE_WARPS = 2


def _torch_dtype_to_triton(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dtype for hypot_: {dtype}")


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _hypot_inplace_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """In-place hypot: each CTA strides over several contiguous blocks and writes results back in the input dtype."""
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    element_offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK_SIZE + element_offsets
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        x32 = x.to(COMPUTE_DTYPE)
        y32 = y.to(COMPUTE_DTYPE)
        result = tl.sqrt(x32 * x32 + y32 * y32)
        tl.store(out_ptr + offsets, result.to(OUT_DTYPE), mask=mask)


def _launch_config(n_elements):
    """Small shapes use a single-wave small tile; large shapes use a large tile to cut CTA scheduling overhead."""
    if n_elements <= SMALL_TILE_THRESHOLD:
        return max(triton.next_power_of_2(32 * math.isqrt(n_elements)), 32), SMALL_WARPS
    return LARGE_BLOCK_SIZE, LARGE_WARPS


def hypot_(self: torch.Tensor, other):
    logger.debug("GEMS_ENFLAME HYPOT_")

    # In-place operators keep self's dtype; scalar or tensor other is dtype-aligned first, then broadcast to a
    # contiguous view.
    if isinstance(other, torch.Tensor):
        other_t = other.to(device=self.device, dtype=self.dtype)
    else:
        other_t = torch.tensor(other, device=self.device, dtype=self.dtype)
    other_b = torch.broadcast_to(other_t, self.shape).contiguous()

    n_elements = self.numel()
    if n_elements == 0:
        return self

    x_contig = self.contiguous()
    out_buf = x_contig if self.is_contiguous() else torch.empty_like(x_contig)

    block_size, num_warps = _launch_config(n_elements)
    grid = (min(triton.cdiv(n_elements, block_size), GRID_CAP),)
    with torch_device_fn.device(self.device):
        _hypot_inplace_kernel[grid](
            x_contig.view(-1),
            other_b.view(-1),
            out_buf.view(-1),
            n_elements,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=_torch_dtype_to_triton(self.dtype),
            COMPUTE_DTYPE=tl.float32,
            num_warps=num_warps,
        )

    if not self.is_contiguous():
        self.copy_(out_buf)

    return self
