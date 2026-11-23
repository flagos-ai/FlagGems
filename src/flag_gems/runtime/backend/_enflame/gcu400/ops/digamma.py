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
# Tile size is picked per element-count tier: the transcendental-heavy body is
# latency bound for small inputs (a single oversized tile idles most lanes) and
# bandwidth bound for large ones (a small tile adds loop overhead). The four
# tiers below are the measured optima of a full BLOCK sweep on GCU400.
_TIER_TINY_MAX = 16 * 1024
_TIER_SMALL_MAX = 64 * 1024
_TIER_LARGE_MAX = 16 * 1024 * 1024
_BLOCK_TINY = 512
_BLOCK_SMALL = 4096
_BLOCK_MEDIUM = 2048
_BLOCK_LARGE = 16384


def _pick_block_size(num_elements: int) -> int:
    if num_elements <= _TIER_TINY_MAX:
        return _BLOCK_TINY
    if num_elements <= _TIER_SMALL_MAX:
        return _BLOCK_SMALL
    if num_elements <= _TIER_LARGE_MAX:
        return _BLOCK_MEDIUM
    return _BLOCK_LARGE


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

        # Recurrence shifts xr up to 6 before the asymptotic series. At y = 6
        # the first omitted term is 1/(132*y^10) ~ 8e-10, far below fp32
        # resolution, so this keeps full precision while removing two of the
        # eight costly divisions per element.
        s = tl.zeros_like(x_f32)
        y = xr
        for _ in range(6):
            m = y < 6.0
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
        # log2 is a native GCU transcendental; multiplying by ln(2) is cheaper
        # than the generic natural-log lowering and stays within fp32 accuracy.
        psi_y = tl.math.log2(y) * 0.6931471805599453 + s + series

        # Apply reflection if needed
        cot_term = tl.cos(pi * x_f32) / tl.sin(pi * x_f32)
        result = tl.where(reflect_mask, psi_y - pi * cot_term, psi_y)

        result = result.to(x.dtype)
        tl.store(output_ptr + offsets, result, mask=mask)


def _launch(x_contig: torch.Tensor, out_contig: torch.Tensor):
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    block_size = _pick_block_size(n_elements)
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
    x_contig = A if A.is_contiguous() else A.contiguous()
    out = torch.empty_like(x_contig)
    # The kernel only uses the flat element count and the data pointers, so the
    # tensors are passed as-is: `view(-1)`/`view_as` would add three dispatches
    # per call, which is the whole launch cost for tiny inputs.
    _launch(x_contig, out)
    return out


def digamma_out(A, out):
    logger.debug("GEMS_ENFLAME DIGAMMA_OUT")
    x_contig = A.to(out.dtype).contiguous()
    if out.is_contiguous():
        _launch(x_contig, out)
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch(x_contig, out_contig)
        out.copy_(out_contig)
    return out
