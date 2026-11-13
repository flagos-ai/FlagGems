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

"""GCU400 专属 mv：行核（大 N）+ 1D 行流核（小 N）双臂分派，不走 autotune。

GCU400 实测定档（L600/48 slot，DSM 917504B、每 block ≤448KB 才 2 block/SPC 共驻）：
- num_warps=1（tile 内 axis=1 归约的 nw1 分水岭，nw≥2 回归 1.7~3.3×）；
- NUM_M_ITERS constexpr 静态展开（动态 range 循环 -19%）；
- b 以直下标寻址（可被 LICM 提出行块循环，-19%），a/b 不做 fp32 预转型
  （原生 FMA 进 fp32 acc，DSM 流量减半，-11%）；
- BN*BM ≤ 32768 elem（nocast dsm ≈196KB）：超限并发降级（BN64*BM1024 实测 3.6× 回归）；
- 小 N（≤SMALL_N_MAX）行数不足以铺满 grid，改 1D 行流核（N=1 实测 1.37×）。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 48
BLOCK_N = 32
BLOCK_M = 1024
SMALL_N_MAX = 128


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
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    NUM_M_ITERS: tl.constexpr,
):
    num_programs = tl.num_programs(0)
    if EVEN_N:
        num_row_blocks = N // BLOCK_N
    else:
        num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    for row_block in tl.range(tl.program_id(0), num_row_blocks, num_programs):
        row_index = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if EVEN_N and EVEN_M:
            A_ptrs = A + row_index[:, None] * stride_an + offs_m[None, :] * stride_am
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for it in tl.static_range(NUM_M_ITERS):
                a = tl.load(A_ptrs)
                b = tl.load(B + it * BLOCK_M * stride_bm + offs_m * stride_bm)
                acc += a * b
                A_ptrs += BLOCK_M * stride_am
            out = tl.sum(acc, axis=1)
            tl.store(C + row_index * stride_cn, out)
        else:
            offset_n = row_index[:, None]
            m_off = offs_m[None, :]
            n_mask = offset_n < N
            A_ptrs = A + offset_n * stride_an + m_off * stride_am
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for it in tl.static_range(NUM_M_ITERS):
                m_mask = it * BLOCK_M + m_off < M
                a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0)
                b = tl.load(
                    B + it * BLOCK_M * stride_bm + offs_m * stride_bm,
                    mask=it * BLOCK_M + offs_m < M,
                    other=0.0,
                )
                acc += a * b
                A_ptrs += BLOCK_M * stride_am
            out = tl.sum(acc, axis=1)
            tl.store(C + row_index * stride_cn, out, mask=row_index < N)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def mv_kernel_small(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    NUM_M_ITERS: tl.constexpr,
):
    num_programs = tl.num_programs(0)
    offs_m = tl.arange(0, BLOCK_M)
    for row in tl.range(tl.program_id(0), N, num_programs):
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for it in tl.static_range(NUM_M_ITERS):
            m_mask = it * BLOCK_M + offs_m < M
            a = tl.load(
                A + row * stride_an + it * BLOCK_M * stride_am + offs_m * stride_am,
                mask=m_mask,
                other=0.0,
            )
            b = tl.load(
                B + it * BLOCK_M * stride_bm + offs_m * stride_bm,
                mask=m_mask,
                other=0.0,
            )
            acc += a * b
        out = tl.sum(acc, axis=0)
        tl.store(C + row * stride_cn, out)


def _next_pow2(value):
    return int(triton.next_power_of_2(max(int(value), 1)))


def mv(inp, vec):
    logger.debug("GEMS_ENFLAME MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    if N == 0 or M == 0:
        return out
    with torch_device_fn.device(inp.device):
        if N <= SMALL_N_MAX:
            block_m = min(2048, _next_pow2(M))
            grid = (min(NUM_SIPS, N),)
            mv_kernel_small[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
                BLOCK_M=block_m,
                NUM_M_ITERS=triton.cdiv(M, block_m),
                num_warps=1,
            )
        else:
            block_m = min(BLOCK_M, _next_pow2(M))
            num_m_iters = triton.cdiv(M, block_m)
            block_n = BLOCK_N if num_m_iters <= 4 else 16
            grid = (min(NUM_SIPS, triton.cdiv(N, block_n)),)
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
                BLOCK_N=block_n,
                BLOCK_M=block_m,
                EVEN_N=(N % block_n == 0),
                EVEN_M=(M % block_m == 0),
                NUM_M_ITERS=num_m_iters,
                num_warps=1,
            )
    return out
