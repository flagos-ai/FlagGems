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

"""GCU400 专属 mv：固定 tile 与固定 grid，避免 libtuner autotune 冷编译。"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 固定 grid 上限取 GCU400 的计算单元数量级，剩余行块由核内跨步循环覆盖。
NUM_SIPS = 48
BLOCK_N = 16
BLOCK_M = 1024


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    num_programs = tl.num_programs(0)
    num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
    for row_block in tl.range(tl.program_id(0), num_row_blocks, num_programs):
        row_index = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offset_n = row_index[:, None]
        offset_m = tl.arange(0, BLOCK_M)[None, :]
        n_mask = offset_n < N
        A_ptrs = A + offset_n * stride_an + offset_m * stride_am
        B_ptrs = B + offset_m * stride_bm
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_mask = m + offset_m < M
            a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc += a * b
            A_ptrs += BLOCK_M * stride_am
            B_ptrs += BLOCK_M * stride_bm

        acc = tl.sum(acc, axis=1)
        tl.store(C + row_index * stride_cn, acc, mask=row_index < N)


def mv(inp, vec):
    logger.debug("GEMS_ENFLAME MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = (triton.cdiv(N, BLOCK_N) if N <= 48 * BLOCK_N else NUM_SIPS,)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_M=BLOCK_M,
            num_warps=4,
        )
    return out
