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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
device_ = device


@libentry()
@triton.jit
def logcumsumexp_chunk_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK: tl.constexpr,
    GRID_DIM: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    # 固定网格 + 行跨步；每行按 BLOCK 分块做 log-sum-exp 前缀扫描，
    # 用标量 running_max / running_sum 在块间传递，避免 next_pow2(N) 超大 tile 打爆 DSM。
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK)
    total_rows = M * K
    for row_id in tl.range(pid, total_rows, num_programs):
        row_m = row_id // K
        row_k = row_id - row_m * K
        row_base = row_m * N * K + row_k
        running_max = float("-inf")
        running_sum = 0.0
        for chunk_id in range(tl.cdiv(N, BLOCK)):
            cols = chunk_id * BLOCK + offsets
            mask = cols < N
            x = tl.load(inp + row_base + cols * K, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            chunk_max = tl.max(x, 0)
            new_max = tl.maximum(running_max, chunk_max)
            if USE_EXP2:
                shifted = tl.math.exp2((x - new_max) * 1.4426950408889634)
                prev_scaled = running_sum * tl.math.exp2(
                    (running_max - new_max) * 1.4426950408889634
                )
            else:
                shifted = tl.exp(x - new_max)
                prev_scaled = running_sum * tl.exp(running_max - new_max)
            local_prefix = tl.cumsum(shifted, 0)
            if USE_EXP2:
                result = (
                    new_max
                    + tl.math.log2(prev_scaled + local_prefix) * 0.6931471805599453
                )
            else:
                result = new_max + tl.log(prev_scaled + local_prefix)
            tl.store(out + row_base + cols * K, result, mask=mask)
            running_sum = prev_scaled + tl.sum(shifted, 0)
            running_max = new_max


def logcumsumexp_out(inp, dim=1, *, dtype=None, out):
    """GCU400 专属实现：按维度把输入规整为 (M, N, K)，沿 N 做分块前缀 log-sum-exp。"""
    logger.debug("GEMS_ENFLAME LOGCUMSUMEXP_OUT")
    dim = dim % inp.ndim
    shape = inp.shape
    M = 1
    for i in range(dim):
        M *= shape[i]
    N = shape[dim]
    K = inp.numel() // M // N
    inp = inp.contiguous()
    if out is None:
        out = torch.empty_like(inp, dtype=dtype or inp.dtype)
    BLOCK = 4096
    GRID_DIM = 48
    grid = (min(GRID_DIM, M * K),)
    with torch_device_fn.device(inp.device):
        logcumsumexp_chunk_kernel[grid](
            inp, out, M, N, K, BLOCK, GRID_DIM, False, num_warps=4
        )
    return out
