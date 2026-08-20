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

MAX_GRID = 48


@libentry()
@triton.jit(do_not_specialize=["N"])
def exp2_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.exp2(x)
        tl.store(out_ptr + off, out, mask=mask)


def _launch_exp2(inp, out, N):
    is_fp32 = inp.dtype == torch.float32
    if N <= 1024:
        BLOCK = 1024
    elif N <= 32768:
        BLOCK = triton.next_power_of_2(N)
    else:
        BLOCK = 65536 if is_fp32 else 131072

    grid_size = min(triton.cdiv(N, BLOCK), MAX_GRID)
    with torch_device_fn.device(inp.device):
        exp2_kernel[(grid_size,)](inp, out, N, BLOCK=BLOCK, num_warps=4)


def exp2(A):
    logger.debug("GEMS_ENFLAME EXP2")
    inp = A if A.is_contiguous() else A.contiguous()
    out = torch.empty_like(inp)
    _launch_exp2(inp, out, inp.numel())
    return out


def exp2_(A):
    logger.debug("GEMS_ENFLAME EXP2_")
    inp = A if A.is_contiguous() else A.contiguous()
    _launch_exp2(inp, inp, inp.numel())
    if not A.is_contiguous():
        A.copy_(inp)
    return A
