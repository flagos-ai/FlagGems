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
@triton.jit(do_not_specialize=["N"])
def logical_or_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0)
        y = tl.load(Y + off, mask=mask, other=0)
        res = (x != 0) | (y != 0)
        tl.store(Out + off, res, mask=mask)


def _choose_block(N, is_wide):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_wide:
        return 65536
    return 8192


def _launch(inp, other, out):
    N = inp.numel()
    is_wide = inp.element_size() >= 4
    BLOCK = _choose_block(N, is_wide)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        logical_or_kernel[(grid_size,)](
            inp,
            other,
            out,
            N,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def _is_i64(A, B):
    return A.dtype == torch.int64 or B.dtype == torch.int64


def logical_or(A, B):
    logger.debug("GEMS_ENFLAME LOGICAL_OR")
    if _is_i64(A, B):
        A, B = A.to(torch.int32), B.to(torch.int32)
    inp = A if A.is_contiguous() else A.contiguous()
    y = B if B.is_contiguous() else B.contiguous()
    out = torch.empty(inp.shape, dtype=torch.bool, device=inp.device)
    return _launch(inp, y, out)


def logical_or_(A, B):
    logger.debug("GEMS_ENFLAME LOGICAL_OR_")
    if _is_i64(A, B):
        tmp = logical_or(A, B)
        A.copy_(tmp)
        return A
    y = B if B.is_contiguous() else B.contiguous()
    if not A.is_contiguous():
        A.copy_(logical_or(A, y))
        return A
    return _launch(A, y, A)
