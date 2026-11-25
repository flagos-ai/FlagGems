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

_tan = tl_extra_shim.tan

logger = logging.getLogger(__name__)

NUM_SIPS = 48


@libentry()
@triton.jit(do_not_specialize=["N"])
def tan_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = _tan(x)
        tl.store(out_ptr + off, out, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_fp32:
        return 65536
    return 8192


def _launch(inp, out):
    N = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        tan_kernel[(grid_size,)](
            inp,
            out,
            N,
            BLOCK=BLOCK,
            num_warps=4,
        )


def tan(A):
    logger.debug("GEMS_ENFLAME TAN")
    inp = A if A.is_contiguous() else A.contiguous()
    out = torch.empty_like(inp)
    _launch(inp, out)
    return out


def tan_(A):
    logger.debug("GEMS_ENFLAME TAN_")
    if not A.is_contiguous():
        A.copy_(tan(A))
        return A
    _launch(A, A)
    return A
