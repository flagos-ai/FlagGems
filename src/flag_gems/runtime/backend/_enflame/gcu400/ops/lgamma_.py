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
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N"])
def _lgamma_kernel(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        y = tl_extra_shim.lgamma(x)
        tl.store(Out + off, y, mask=mask)


def _choose_block(N):
    if N <= 1024:
        return triton.next_power_of_2(max(N, 1))
    return 512


def _launch_lgamma(X, out):
    N = X.numel()
    if N == 0:
        return out
    BLOCK = _choose_block(N)
    grid = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(X.device):
        _lgamma_kernel[(grid,)](X, out, N, BLOCK=BLOCK, num_warps=4)
    return out


def lgamma(A):
    logger.debug("GEMS_ENFLAME LGAMMA")
    x = A if A.is_contiguous() else A.contiguous()
    out_dtype = torch.float32 if not x.dtype.is_floating_point else x.dtype
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    return _launch_lgamma(x, out)


def lgamma_(A):
    logger.debug("GEMS_ENFLAME LGAMMA_")
    x = A if A.is_contiguous() else A.contiguous()
    out = _launch_lgamma(x, x)
    if out is not A:
        A.copy_(out)
    return A
