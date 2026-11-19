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

NUM_SIPS = 24


@triton.jit
def _reduce_all(a, b):
    return a and b


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _is_all_true_stage1(inp, mid, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    offs = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], True, tl.int1)
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + offs
        mask = offsets < n_elements
        v = tl.load(inp + offsets, mask=mask, other=1).to(tl.int1)
        acc = acc & v
    all_val = tl.reduce(acc, axis=0, combine_fn=_reduce_all)
    tl.store(mid + pid, all_val)


@libentry()
@triton.jit
def _is_all_true_stage2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_val = tl.load(mid + offset, mask=offset < mid_size, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=_reduce_all)
    tl.store(out, all_val)


def _choose_block(n_elements):
    if n_elements <= 4096:
        return triton.next_power_of_2(n_elements)
    return 131072


def _is_all_true(inp):
    logger.debug("GEMS_ENFLAME _IS_ALL_TRUE")
    assert inp.dtype == torch.bool, "Input tensor must be of type bool"

    n_elements = inp.numel()
    if n_elements == 0:
        return torch.tensor(True, dtype=torch.bool, device=inp.device)

    inp = inp if inp.is_contiguous() else inp.contiguous()
    BLOCK = _choose_block(n_elements)
    grid = min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)
    mid = torch.empty((grid,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)
    BLOCK_MID = max(triton.next_power_of_2(grid), 1)

    with torch_device_fn.device(inp.device):
        _is_all_true_stage1[(grid,)](inp, mid, n_elements, BLOCK=BLOCK, num_warps=4)
        if grid == 1:
            return mid.reshape([])
        _is_all_true_stage2[(1,)](mid, out, grid, BLOCK_MID=BLOCK_MID, num_warps=4)
    return out
