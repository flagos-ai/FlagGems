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
MAX_GRID = NUM_SIPS * 2
MAX_BLOCK = 65536
MIN_BLOCK = 2048
SMALL_N = 32768


@libentry()
@triton.jit(do_not_specialize=["N"])
def isnan_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + off)
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask)
        if tl.constexpr(x.dtype.is_fp32()):
            xi = x.to(tl.int32, bitcast=True)
            out = (xi & 0x7FFFFFFF) > 0x7F800000
        elif tl.constexpr(x.dtype.is_fp64()):
            xi = x.to(tl.int64, bitcast=True)
            out = (xi & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
        else:
            out = x != x
        if EVEN:
            tl.store(out_ptr + off, out)
        else:
            tl.store(out_ptr + off, out, mask=mask)


def _choose_block(N):
    if N <= 1024:
        return 1024
    if N <= SMALL_N:
        return triton.next_power_of_2(N)
    BLOCK = MAX_BLOCK
    while BLOCK > MIN_BLOCK and triton.cdiv(N, BLOCK) < MAX_GRID:
        BLOCK //= 2
    return BLOCK


def _launch(inp, out):
    N = inp.numel()
    if N == 0:
        return
    BLOCK = _choose_block(N)
    num_warps = 4 if N <= SMALL_N else 2
    grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
    with torch_device_fn.device(inp.device):
        isnan_kernel[(grid_size,)](
            inp,
            out,
            N,
            BLOCK=BLOCK,
            EVEN=(N % BLOCK) == 0,
            num_warps=num_warps,
        )


def isnan(A):
    logger.debug("GEMS_ENFLAME ISNAN")
    inp = A if A.is_contiguous() else A.contiguous()
    out = torch.empty(inp.shape, dtype=torch.bool, device=inp.device)
    _launch(inp, out)
    return out
