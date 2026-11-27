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

"""GCU400-specific mv: two-arm dispatch of a row kernel (large N) + 1D row-stream kernel (small N), no autotune.

GCU400 measured tiers (L600/48 slot, DSM 917504B, each block <=448KB for 2 blocks/SPC residency):
- num_warps=1 (the nw1 watershed for in-tile axis=1 reduction, nw>=2 regresses 1.7~3.3x);
- NUM_M_ITERS constexpr static unrolling (dynamic range loop -19%);
- b addressed by direct index (LICM can hoist it out of the row-block loop, -19%), no fp32 pre-cast of a/b
  (native FMA into fp32 acc halves DSM traffic, -11%);
- BN*BM <= 32768 elem (nocast dsm ~196KB): exceeding it degrades concurrency (BN64*BM1024 measured 3.6x regression);
- small N (<=SMALL_N_MAX) has too few rows to fill the grid, so use the 1D row-stream kernel (N=1 measured 1.37x).
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
# The row kernel unrolls every m-chunk statically, so its DSM demand grows with
# the chunk count (one [BLOCK_N, BLOCK_M] tile per chunk): 64 chunks (=K 65536)
# need 917632B and fail to compile against the 917504B limit, and a runtime loop
# instead measures 7979us vs 227us for aten. Above this chunk count the 1D
# row-stream kernel takes over with a 4096-wide chunk (measured 183us).
_ROW_STREAM_CHUNK_LIMIT = 8
_ROW_STREAM_BLOCK_M = 4096


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


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def mv_kernel_trap(
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
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for it in tl.static_range(NUM_M_ITERS):
                a = tl.load(A_ptrs)
                b = tl.load(B + it * BLOCK_M * stride_bm + offs_m * stride_bm)
                acc += tl.sum(a * b[None, :], axis=1)
                A_ptrs += BLOCK_M * stride_am
            tl.store(C + row_index * stride_cn, acc)
        else:
            offset_n = row_index[:, None]
            m_off = offs_m[None, :]
            n_mask = offset_n < N
            A_ptrs = A + offset_n * stride_an + m_off * stride_am
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for it in tl.static_range(NUM_M_ITERS):
                m_mask = it * BLOCK_M + m_off < M
                a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0)
                b = tl.load(
                    B + it * BLOCK_M * stride_bm + offs_m * stride_bm,
                    mask=it * BLOCK_M + offs_m < M,
                    other=0.0,
                )
                acc += tl.sum(a * b[None, :], axis=1)
                A_ptrs += BLOCK_M * stride_am
            tl.store(C + row_index * stride_cn, acc, mask=row_index < N)


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

        def _launch_row_stream(block_m_cap):
            """1D row-stream kernel: each program walks the rows, DSM keeps a single [BLOCK_M] tile."""
            block_m = min(block_m_cap, _next_pow2(M))
            mv_kernel_small[(min(NUM_SIPS, N),)](
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

        if N <= SMALL_N_MAX:
            _launch_row_stream(2048)
        elif triton.cdiv(M, BLOCK_M) > _ROW_STREAM_CHUNK_LIMIT:
            # Row-stream kernel fetches along M: row-major is contiguous (183us), while column-major with stride=M is
            # fully non-contiguous
            # (8831us). Materializing once into row-major (128MB @ ~700GB/s, about 366us) and then using
            # the row-stream kernel is an order of magnitude faster than running directly on column-major.
            if inp.stride(1) != 1:
                inp = inp.contiguous()
            _launch_row_stream(_ROW_STREAM_BLOCK_M)
        else:
            block_m = min(BLOCK_M, _next_pow2(M))
            num_m_iters = triton.cdiv(M, block_m)
            # fp32 trap arm (per-chunk on-the-fly tree + nw2): only for the levels that measurably win;
            # the 16-bit tree tax grows with the chunk count (measured 0.80× on (16384,4096)), so keep the original row
            # kernel.
            use_trap = inp.dtype == torch.float32 and (
                3 <= num_m_iters <= 4 or (num_m_iters == 2 and N <= 2048)
            )
            if use_trap:
                grid = (min(NUM_SIPS, triton.cdiv(N, BLOCK_N)),)
                mv_kernel_trap[grid](
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
                    BLOCK_M=block_m,
                    EVEN_N=(N % BLOCK_N == 0),
                    EVEN_M=(M % block_m == 0),
                    NUM_M_ITERS=num_m_iters,
                    num_warps=2,
                )
            else:
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
