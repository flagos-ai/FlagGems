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
def digamma_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Math identical to the generic implementation (reflection + recurrence
    # shift + asymptotic expansion); only the launch strategy differs.
    pid = tl.program_id(axis=0)
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)

        pi = 3.1415926535897932384626433832795028841971

        # Reflection for x < 0.5: psi(x) = psi(1 - x) - pi * cot(pi * x)
        reflect_mask = x_f32 < 0.5
        xr = tl.where(reflect_mask, 1.0 - x_f32, x_f32)

        # Use recurrence to shift xr to >= 8 for better asymptotic precision
        s = tl.zeros_like(x_f32)
        y = xr
        for _ in range(8):
            m = y < 8.0
            s = s - tl.where(m, 1.0 / y, 0.0)
            y = tl.where(m, y + 1.0, y)

        # Asymptotic expansion for digamma at large y
        r = 1.0 / y
        r2 = r * r
        t2 = r2
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        series = (
            (-0.5 * r)
            + (-1.0 / 12.0) * t2
            + (1.0 / 120.0) * t4
            + (-1.0 / 252.0) * t6
            + (1.0 / 240.0) * t8
        )
        psi_y = tl.log(y) + s + series

        # Apply reflection if needed
        cot_term = tl.cos(pi * x_f32) / tl.sin(pi * x_f32)
        result = tl.where(reflect_mask, psi_y - pi * cot_term, psi_y)

        result = result.to(x.dtype)
        tl.store(output_ptr + offsets, result, mask=mask)


def _launch(x_contig: torch.Tensor, out_contig: torch.Tensor):
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    block_size = 32768
    num_tile = triton.cdiv(n_elements, block_size)
    grid = (min(_GRID_SIZE, num_tile), 1, 1)
    with torch_device_fn.device(out_contig.device.index):
        digamma_kernel[grid](
            x_contig,
            out_contig,
            n_elements,
            BLOCK_SIZE=block_size,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=4,
            num_warps=4,
        )


def digamma(A):
    logger.debug("GEMS_ENFLAME DIGAMMA")
    x_contig = A.contiguous()
    out = torch.empty_like(x_contig)
    _launch(x_contig.view(-1), out.view(-1))
    return out.view_as(A)


def digamma_out(A, out):
    logger.debug("GEMS_ENFLAME DIGAMMA_OUT")
    x_contig = A.to(out.dtype).contiguous()
    if out.is_contiguous():
        _launch(x_contig.view(-1), out.view(-1))
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch(x_contig.view(-1), out_contig.view(-1))
        out.copy_(out_contig)
    return out
