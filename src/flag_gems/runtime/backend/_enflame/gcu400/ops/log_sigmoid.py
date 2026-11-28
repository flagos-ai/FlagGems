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
def log_sigmoid_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
            y = -tl.log(1.0 + tl.exp(-x))
            tl.store(out_ptr + off, y)
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
            y = -tl.log(1.0 + tl.exp(-x))
            tl.store(out_ptr + off, y, mask=mask)


def _choose_block(n, itemsize):
    # (BLOCK, num_warps) ladder, locked by two isolated do_bench A/B sweeps on
    # GCU400/L600 (probe w500/r100, official bench w1000/r100 resample); same
    # bands as the relu.py / sigmoid.py ladders. Each (dtype, BLOCK) pair maps
    # to exactly one num_warps: LibEntry's dispatch key omits num_warps, so a
    # shared BLOCK must never launch with two different warp counts. Grid stays
    # capped at NUM_SPC=48: 96/192 measured worse on every band.
    if n <= 2048:
        return max(256, triton.next_power_of_2(n)), 2
    if n <= 4096:
        # 2-byte tiny shapes prefer two 2048 CTAs (3.33us vs 3.56us single);
        # fp32 prefers one full 4096 CTA (3.33us vs 3.59us @2048x2).
        return (2048, 2) if itemsize == 2 else (4096, 4)
    if n <= 65536:
        return 2048, 2
    if n <= 4194304:
        return 8192, 2
    if itemsize == 2:
        return 131072, 8
    return 65536, 4


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
    block, nw = _choose_block(n, inp.element_size())
    if triton.cdiv(n, block) * block > INT32_MAX:
        return _log_sigmoid_template(x)
    out = torch.empty_like(inp)
    grid = (min(triton.cdiv(n, block), NUM_SPC),)
    with torch_device_fn.device(inp.device):
        log_sigmoid_kernel[grid](
            inp,
            out,
            n,
            BLOCK=block,
            EVEN=(n % block == 0),
            num_warps=nw,
        )
    return out
