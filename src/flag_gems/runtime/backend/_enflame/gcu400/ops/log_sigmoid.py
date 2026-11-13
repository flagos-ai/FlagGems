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

from flag_gems.ops.log_sigmoid import log_sigmoid as _log_sigmoid_template
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SPC = 48
INT32_MAX = torch.iinfo(torch.int32).max


@libentry()
@triton.jit(do_not_specialize=["N"])
def log_sigmoid_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        y = -tl.log(1.0 + tl.exp(-x))
        tl.store(out_ptr + off, y, mask=mask)


def _choose_block(n, itemsize):
    if n <= 2048:
        return max(256, triton.next_power_of_2(n))
    if n <= 65536:
        return 2048
    if n <= 4194304:
        return 8192
    if itemsize == 2:
        return 65536 if n <= 67108864 else 131072
    return 65536


def log_sigmoid(x):
    logger.debug("GEMS_ENFLAME LOG_SIGMOID")
    if (
        (not x.is_contiguous())
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or x.numel() == 0
    ):
        return _log_sigmoid_template(x)

    inp = x
    n = inp.numel()
    block = _choose_block(n, inp.element_size())
    if triton.cdiv(n, block) * block > INT32_MAX:
        return _log_sigmoid_template(x)
    out = torch.empty_like(inp)
    grid = (min(triton.cdiv(n, block), NUM_SPC),)
    nw = 2 if inp.element_size() == 2 else 4
    with torch_device_fn.device(inp.device):
        log_sigmoid_kernel[grid](
            inp,
            out,
            n,
            BLOCK=block,
            num_warps=nw,
        )
    return out
