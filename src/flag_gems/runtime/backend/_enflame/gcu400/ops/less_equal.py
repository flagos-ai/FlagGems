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
@triton.jit(do_not_specialize=["N", "Y_SCALAR"])
def less_equal_kernel(X, Y, Out, N, Y_SCALAR, HAS_Y: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if HAS_Y:
            y = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
        else:
            y = Y_SCALAR
        res = x <= y
        tl.store(Out + off, res, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return min(triton.next_power_of_2(N), 8192)
    if is_fp32:
        return 65536
    return 8192


def _launch(inp, other, out, has_y):
    N = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    y_ptr = other if has_y else inp
    y_scalar = 0.0 if has_y else float(other)
    with torch_device_fn.device(inp.device):
        less_equal_kernel[(grid_size,)](
            inp,
            y_ptr,
            out,
            N,
            y_scalar,
            HAS_Y=has_y,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def _bool_out(inp):
    return torch.empty(inp.shape, dtype=torch.bool, device=inp.device)


def less_equal(A, B):
    logger.debug("GEMS_ENFLAME LESS_EQUAL")
    inp = A if A.is_contiguous() else A.contiguous()
    y = B if B.is_contiguous() else B.contiguous()
    return _launch(inp, y, _bool_out(inp), True)


def less_equal_scalar(A, B):
    logger.debug("GEMS_ENFLAME LESS_EQUAL_SCALAR")
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, B, _bool_out(inp), False)
