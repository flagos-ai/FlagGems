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

# Fixed Grid on GCU400 (48 CTAs). The generic kernel launches
# cdiv(n_elements, BLOCK_SIZE) CTAs, which exceeds the 65535 grid.x hardware
# limit for large inputs (e.g. 2**28 elements -> 262144 CTAs ->
# OutOfResources). Striding from a fixed grid keeps every launch within the
# limit regardless of input size.
_GRID_SIZE = 48


@libentry()
@triton.jit
def log1p_kernel(
    x_ptr,
    y_ptr,
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
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.log(1.0 + x.to(tl.float32)).to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)


def _launch(x_contig: torch.Tensor, out_contig: torch.Tensor):
    n_elements = out_contig.numel()
    if n_elements == 0:
        return
    # Tuned on GCU400: BLOCK=32768 + num_warps=4 + num_stages=4.
    block_size = 32768
    num_tile = triton.cdiv(n_elements, block_size)
    grid = (min(_GRID_SIZE, num_tile), 1, 1)
    with torch_device_fn.device(out_contig.device.index):
        log1p_kernel[grid](
            x_contig,
            out_contig,
            n_elements,
            BLOCK_SIZE=block_size,
            GRID_SIZE=min(_GRID_SIZE, num_tile),
            NUM_STAGES=4,
            num_warps=4,
        )


def log1p(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME LOG1P")
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    _launch(x_contig.view(-1), out.view(-1))
    return out.view_as(x)


def log1p_(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME LOG1P_")
    if not x.is_floating_point():
        # Integer in-place is not representable; mirror the functional path.
        x.copy_(log1p(x))
        return x

    if x.is_contiguous():
        _launch(x.view(-1), x.view(-1))
    else:
        x_contig = x.contiguous()
        _launch(x_contig.view(-1), x_contig.view(-1))
        x.copy_(x_contig)
    return x


def log1p_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME LOG1P_OUT")
    x_contig = x.to(out.dtype).contiguous()

    if out.is_contiguous():
        _launch(x_contig.view(-1), out.view(-1))
    else:
        out_contig = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch(x_contig.view(-1), out_contig.view(-1))
        out.copy_(out_contig)
    return out
