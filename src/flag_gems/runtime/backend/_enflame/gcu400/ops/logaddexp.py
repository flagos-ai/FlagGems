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
def logaddexp_kernel(X, Y, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask).to(tl.float32)
        y = tl.load(Y + off, mask=mask).to(tl.float32)
        m = tl.maximum(x, y)
        delta = x - y
        out = m + tl.log(1.0 + tl.exp(-tl.abs(delta)))
        tl.store(Out + off, out, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_fp32:
        return 65536
    return 16384


def _launch(x, y, out):
    N = x.numel()
    is_fp32 = x.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        logaddexp_kernel[(grid_size,)](
            x,
            y,
            out,
            N,
            BLOCK=BLOCK,
            num_warps=2,
        )


def logaddexp(self, other):
    logger.debug("GEMS_ENFLAME LOGADDEXP")
    x = self if self.is_contiguous() else self.contiguous()
    y = other if other.is_contiguous() else other.contiguous()
    out = torch.empty_like(x)
    _launch(x, y, out)
    return out


def logaddexp_out(self, other, out):
    logger.debug("GEMS_ENFLAME LOGADDEXP_OUT")
    x = self if self.is_contiguous() else self.contiguous()
    y = other if other.is_contiguous() else other.contiguous()
    _launch(x, y, out)
    return out
