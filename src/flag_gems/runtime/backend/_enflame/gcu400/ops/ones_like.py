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

from .ones import _fill_ones_flat, _ones_fill_params

logger = logging.getLogger(__name__)

# Pure-write stream calibration on GCU400 (new_full RSI, 2026-11-16): the 24
# schedulable SIPs saturate the DRAM write wall (~3.25 TB/s), so the grid is
# capped at 24 (32 is pathological: +20% on mid shapes); 32K-element tiles are
# the mid-band sweet spot and tie the wall at stream sizes, so smaller tiles
# also keep DSM headroom per program.  Tiny sizes sit on the launch floor and
# use one exact (mask-free) CTA covering the whole tensor.
_MAX_GRID = 24
_MID_BLOCK = 32768
_MIN_BLOCK = 2048
_TINY_MAX = 8192


@libentry()
@triton.jit
def ones_like_kernel(
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # Same store semantics as the flat ones fill, specialized with EXACT
    # to drop the mask store predicate when BLOCK divides the flat size.
    # int64/uint64 never reach this kernel: the wrapper routes them through
    # the layout-probing flat fill instead (see ones.py).
    pid = tl.program_id(axis=0)
    num_ctas = tl.num_programs(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        if EXACT:
            tl.store(output_ptr + offsets, 1.0)
        else:
            tl.store(output_ptr + offsets, 1.0, mask=offsets < n_elements)


def _ones_like_block(n_elements):
    if n_elements <= _TINY_MAX:
        # One exact CTA covering the whole tensor; n == 0 never launches.
        return max(2, triton.next_power_of_2(n_elements))
    return min(
        _MID_BLOCK,
        max(_MIN_BLOCK, triton.next_power_of_2(triton.cdiv(n_elements, 48))),
    )


def ones_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS_ENFLAME ONES_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    n = x.numel()
    if out.dtype in (torch.int64, torch.uint64):
        # int64/uint64 int32-word storage layout is runtime-dependent (P17):
        # under the full-width model the legacy single-word fill splats into
        # both halves of every element (0x100000001 vs native 1), so route
        # through the layout-probing flat fill shared with ones() (ones.py).
        n_units, high_word = _ones_fill_params(n, out.dtype, device)
        return _fill_ones_flat(out, n_units, high_word)
    if n == 0:
        return out
    block = _ones_like_block(n)
    grid = (min(_MAX_GRID, triton.cdiv(n, block)),)
    with torch_device_fn.device(x.device):
        ones_like_kernel[grid](
            out,
            n,
            BLOCK=block,
            EXACT=(n % block == 0),
            num_warps=2,
        )
    return out
