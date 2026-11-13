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

# fixed grid 匹配 GCU400 的 SIP 拓扑，核内跨步循环覆盖大 shape
NUM_SIPS = 24
GRID_MULT = 2
# 按元素字节数反比选 tile：16-bit 与 32-bit 分别取块大小
BLOCK_16 = 16384
BLOCK_32 = 65536
NUM_WARPS = 4


@libentry()
@triton.jit(do_not_specialize=["N"])
def exp_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    USE_EXP2: tl.constexpr,
    EVEN_PATH: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN_PATH:
            # 满块路径：去掉边界掩码与 other 填充，减少谓词计算
            x = tl.load(x_ptr + off).to(tl.float32)
            if USE_EXP2:
                y = tl.exp2(x * 1.4426950408889634)
            else:
                y = tl.exp(x)
            tl.store(out_ptr + off, y)
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
            if USE_EXP2:
                y = tl.exp2(x * 1.4426950408889634)
            else:
                y = tl.exp(x)
            tl.store(out_ptr + off, y, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_fp32:
        return BLOCK_32
    return BLOCK_16


def _launch(inp, out):
    N = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * GRID_MULT)
    # 仅当 N 恰好被整数个 BLOCK 覆盖时才能安全去掉掩码
    even_path = False and (N % BLOCK == 0)
    with torch_device_fn.device(inp.device):
        exp_kernel[(grid_size,)](
            inp,
            out,
            N,
            BLOCK=BLOCK,
            USE_EXP2=False,
            EVEN_PATH=even_path,
            num_warps=NUM_WARPS,
        )
    return out


def exp(A):
    logger.debug("GEMS_ENFLAME EXP")
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


def exp_(A):
    logger.debug("GEMS_ENFLAME EXP_")
    if not A.is_contiguous():
        A.copy_(exp(A))
        return A
    return _launch(A, A)


def exp_out(A, out):
    logger.debug("GEMS_ENFLAME EXP_OUT")
    inp = A if A.is_contiguous() else A.contiguous()
    if not out.is_contiguous():
        out.copy_(_launch(inp, torch.empty_like(inp)))
        return out
    return _launch(inp, out)
