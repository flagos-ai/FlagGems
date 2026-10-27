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


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def abs_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        out = tl.abs(x)
        tl.store(out_ptr + off, out, mask=mask)


def _run_abs(inp, out, N_total):
    if N_total <= 65536:
        BLOCK = 8192
    elif N_total <= 524288:
        BLOCK = 32768
    else:
        BLOCK = 65536

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        abs_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=8,
        )


def abs(A):
    logger.debug("GEMS_ENFLAME ABS")
    inp = A.contiguous()
    out = torch.empty_like(inp)
    _run_abs(inp, out, inp.numel())
    return out


def abs_(A):
    logger.debug("GEMS_ENFLAME ABS_")
    inp = A.contiguous()
    _run_abs(inp, A, inp.numel())
    return A
