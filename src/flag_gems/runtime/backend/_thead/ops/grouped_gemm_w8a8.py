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
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

SMALL_ROW_LIMIT = 64
SHORT_REDUCTION_LIMIT = 1024
PREFETCH_REDUCTION_LIMIT = 128
ROW_PREFETCH_LIMIT = 32
GEMV_ROW_LIMIT = 640
RESIDENT_ROWS_PER_EXPERT = 64
STREAMING_GEMV_ROW_LIMIT = 128
GROUPED_VECTOR_MIN_ROWS = 512
LARGE_REDUCTION_MIN = 4096
PIPELINE_BLOCK_M = 16
PIPELINE_DENSE_BLOCK_M = 128
PIPELINE_DENSE_ROWS_PER_EXPERT = 64
PIPELINE_PROGRAMS = 80
PIPELINE_MAX_ROWS_PER_EXPERT = 16


@libentry()
@triton.jit
def grouped_expert_offsets(Counts, Starts, E: tl.constexpr, BLOCK_E: tl.constexpr):
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(Counts + experts, experts < E, other=0)
    tl.store(Starts + experts, tl.cumsum(counts) - counts, experts < E)


@triton.jit
def grouped_aiu_copy(shared, source, dim_x, dim_k, start_x, BLOCK_N: tl.constexpr):
    return tl.inline_asm_elementwise(
        "{ .reg .u32 tid; .reg .pred copy; ppu.mov.u32 tid,%tid.x; "
        "ppu.setp.lt.u32 copy,tid,32; "
        "@copy ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill.2d.b8 "
        "[$1],[$2],{1,$3,$4,0,$5,0},{1,$6,128,1}; "
        "ppu.mov.u32 $0,0; }",
        "=r,r,l,r,r,r,r",
        [shared, source, dim_x, dim_k, start_x, BLOCK_N],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_aiu_commit(dependency):
    return tl.inline_asm_elementwise(
        "ppu.cp.async.commit_group; ppu.mov.u32 $0,0;",
        "=r,r",
        [dependency],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_aiu_prefetch(shared, source, dim_x, dim_k, start_x, BLOCK_N: tl.constexpr):
    return grouped_aiu_commit(
        grouped_aiu_copy(shared, source, dim_x, dim_k, start_x, BLOCK_N)
    )


@triton.jit
def grouped_aiu_wait(dependency):
    return tl.inline_asm_elementwise(
        "ppu.cp.async.wait_group 0; bar.sync 0; ppu.mov.u32 $0,0;",
        "=r,r",
        [dependency],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_aiu_fragment(shared, x, channel, dependency, BLOCK_N: tl.constexpr):
    return tl.inline_asm_elementwise(
        "ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 "
        "{$0,$1,$2,$3},[$4],{0,$5,1,$6,1,$7};",
        "=r,=r,=r,=r,r,r,r,r,r",
        [shared, x, BLOCK_N, channel, dependency],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_mma(a, b, c):
    return tl.inline_asm_elementwise(
        "ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32 "
        "{$0,$1,$2,$3,$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15}, "
        "{$16,$17,$18,$19,$20,$21,$22,$23};",
        "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
        [
            a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[1],
            b[2],
            b[3],
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            c[6],
            c[7],
        ],
        dtype=(
            tl.int32,
            tl.int32,
            tl.int32,
            tl.int32,
            tl.int32,
            tl.int32,
            tl.int32,
            tl.int32,
        ),
        is_pure=True,
        pack=1,
    )


@libentry()
@triton.jit
def grouped_pipeline_metadata(
    Counts, Tiles, E: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr
):
    expert = tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(Counts + experts, experts < E, other=0)
    tiles = tl.cdiv(counts, BLOCK_M)
    first_row = tl.sum(tl.where(experts < expert, counts, 0), 0)
    first_tile = tl.sum(tl.where(experts < expert, tiles, 0), 0)
    count = tl.load(Counts + expert)
    if expert == 0:
        tl.store(Tiles, tl.sum(tiles, 0))
    for block in range(tl.cdiv(count, BLOCK_M)):
        row = first_row + block * BLOCK_M
        tile = Tiles + 1 + 3 * (first_tile + block)
        tl.store(tile, expert)
        tl.store(tile + 1, row)
        tl.store(tile + 2, tl.minimum(first_row + count, row + BLOCK_M))


@triton.jit
def grouped_pipeline_copy(
    shared, source, dim_x, dim_k, start_x, start_k, X: tl.constexpr, dependency
):
    return tl.inline_asm_elementwise(
        "{ .reg .u32 tid; .reg .pred copy; ppu.mov.u32 tid,%tid.x; "
        "ppu.setp.lt.u32 copy,tid,32; "
        "@copy ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill.2d.b8 "
        "[$1],[$2],{1,$3,$4,0,$5,$6},{1,$7,128,1}; ppu.mov.u32 $0,0; }",
        "=r,r,l,r,r,r,r,r,r,~{memory}",
        [shared, source, dim_x, dim_k, start_x, start_k, X, dependency],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_pipeline_stage(
    shared,
    A,
    B,
    first,
    end,
    nfirst,
    kfirst,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    dependency,
):
    for part in tl.static_range(BK // 128):
        token = grouped_pipeline_copy(
            shared + part * BM * 128,
            A,
            end,
            K,
            first,
            kfirst + part * 128,
            BM,
            dependency,
        )
        token = grouped_pipeline_copy(
            shared + BM * BK + part * BN * 128,
            B,
            N,
            K,
            nfirst,
            kfirst + part * 128,
            BN,
            token,
        )
    return tl.inline_asm_elementwise(
        "ppu.cp.async.commit_group; ppu.mov.u32 $0,0;",
        "=r,r,~{memory}",
        [token],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_pipeline_wait(dependency, REMAIN: tl.constexpr):
    return tl.inline_asm_elementwise(
        "ppu.cp.async.wait_group $2; bar.sync 0; ppu.mov.u32 $0,0;",
        "=r,r,n,~{memory}",
        [dependency, REMAIN],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@libentry()
@triton.jit
def grouped_pipeline_int8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    Tiles,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    tl.static_assert(BM % 16 == 0 and BN % 16 == 0 and BK % 128 == 0)
    tl.static_assert(N % 4 == 0 and K % 128 == 0 and (STAGES == 2 or STAGES == 3))
    tl.static_assert(triton.cdiv(K, BK) % STAGES == 0)
    STAGE_BYTES: tl.constexpr = (BM + BN) * BK
    CHUNKS: tl.constexpr = triton.cdiv(K, BK)
    NT: tl.constexpr = triton.cdiv(N, BN)
    thread = tl.arange(0, 32 * (BN // 16))
    lane = thread % 32
    warp = thread // 32
    smem = tle.gpu.alloc(
        [triton.next_power_of_2(STAGES * STAGE_BYTES)],
        tl.int8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared = tle.gpu.local_ptr(smem, (0,)).to(tl.int64).to(tl.int32)
    total = tl.load(Tiles) * NT
    for task in range(tl.program_id(0), total, PROGRAMS):
        tl.debug_barrier()
        mt, nt = task // NT, task % NT
        expert = tl.load(Tiles + 1 + 3 * mt).to(tl.int64)
        first = tl.load(Tiles + 2 + 3 * mt).to(tl.int64)
        end = tl.load(Tiles + 3 + 3 * mt).to(tl.int64)
        bptr = B + expert * N * K
        nfirst = nt * BN
        token = tl.full((), 0, tl.int32)
        for pre in tl.static_range(min(STAGES - 1, CHUNKS)):
            token = grouped_pipeline_stage(
                shared + pre * STAGE_BYTES,
                A,
                bptr,
                first,
                end,
                nfirst,
                pre * BK,
                N,
                K,
                BM,
                BN,
                BK,
                token,
            )
        zero = tl.full((32 * (BN // 16),), 0, tl.int32)
        acc = (zero,) * (8 * (BM // 16))
        # Constant shared slots avoid dynamic ring pointers in the PPU backend.
        for batch in range(CHUNKS // STAGES):
            for slot in tl.static_range(STAGES):
                chunk = batch * STAGES + slot
                if slot == STAGES - 1:
                    if batch + 1 < CHUNKS // STAGES:
                        ready = grouped_pipeline_wait(token, STAGES - 2)
                    else:
                        ready = grouped_pipeline_wait(token, 0)
                else:
                    ready = grouped_pipeline_wait(token, STAGES - 2)
                current = shared + slot * STAGE_BYTES
                if slot == 0 or batch + 1 < CHUNKS // STAGES:
                    token = grouped_pipeline_stage(
                        shared + ((slot + STAGES - 1) % STAGES) * STAGE_BYTES,
                        A,
                        bptr,
                        first,
                        end,
                        nfirst,
                        (chunk + STAGES - 1) * BK,
                        N,
                        K,
                        BM,
                        BN,
                        BK,
                        ready,
                    )
                else:
                    token = ready
                for sub in tl.static_range(BK // 32):
                    cube = sub // 4
                    channel = (sub % 4) * 32
                    b = grouped_aiu_fragment(
                        current + BM * BK + cube * BN * 128,
                        warp * 16,
                        channel,
                        ready,
                        BN,
                    )
                    updated = ()
                    for mi in tl.static_range(BM // 16):
                        a = grouped_aiu_fragment(
                            current + cube * BM * 128, mi * 16, channel, ready, BM
                        )
                        updated += grouped_mma(a, b, acc[mi * 8 : mi * 8 + 8])
                    acc = updated
        nbase = nfirst + warp * 16
        sb = grouped_b_scales(Bs, expert, nbase + lane % 4, N)
        for mi in tl.static_range(BM // 16):
            row0 = first + mi * 16 + lane // 4
            row1 = row0 + 8
            sa = (tl.load(As + row0, row0 < end, 0), tl.load(As + row1, row1 < end, 0))
            grouped_mma_epilogue(
                C, sb, row0, row1, end, nbase, lane, sa, acc[mi * 8 : mi * 8 + 8], N
            )


@triton.jit
def grouped_a_fragment(A, row0, row1, end, kw, K_OFFSET: tl.constexpr):
    words = A.to(tl.pointer_type(tl.int32))
    a0 = tl.load(words + row0 * 32 + kw + K_OFFSET, row0 < end, other=0)
    a1 = tl.load(words + row0 * 32 + kw + K_OFFSET + 4, row0 < end, other=0)
    a2 = tl.load(words + row1 * 32 + kw + K_OFFSET, row1 < end, other=0)
    a3 = tl.load(words + row1 * 32 + kw + K_OFFSET + 4, row1 < end, other=0)
    return a0, a1, a2, a3


@triton.jit
def grouped_shuffle_xor(value, DISTANCE: tl.constexpr):
    return tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0,$1,$2,31,-1;",
        "=r,r,r",
        [value, DISTANCE],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_pack_bfloat16(low, high):
    return tl.inline_asm_elementwise(
        "ppu.cvt.rn.bf16x2.f32 $0,$2,$1;",
        "=r,f,f",
        [low, high],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_transpose_bfloat16(x0, x1, lane):
    neighbor0 = grouped_shuffle_xor(x0, 1)
    neighbor1 = grouped_shuffle_xor(x1, 1)
    control = tl.where((lane & 1) == 0, 0x5410, 0x3276)
    y0 = tl.inline_asm_elementwise(
        "prmt.b32 $0,$1,$2,$3;",
        "=r,r,r,r",
        [x0, neighbor0, control],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    y1 = tl.inline_asm_elementwise(
        "prmt.b32 $0,$1,$2,$3;",
        "=r,r,r,r",
        [x1, neighbor1, control],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    low = (lane & 2) == 0
    exchanged = grouped_shuffle_xor(tl.where(low, y1, y0), 2)
    return tl.where(low, y0, exchanged), tl.where(low, exchanged, y1)


@triton.jit
def grouped_store_bfloat16(C, row, col, low, high, mask, N: tl.constexpr):
    address = C.to(tl.pointer_type(tl.int32)) + row.to(tl.int64) * (N // 2) + col // 2
    tl.inline_asm_elementwise(
        "@$4 ppu.st.global.v2.b32 [$1],{$2,$3}; ppu.mov.u32 $0,0;",
        "=r,l,r,r,b",
        [address, low, high, mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_b_scales(Bs, expert, col, N: tl.constexpr):
    return (
        tl.load(Bs + expert * N + col, col < N, other=0),
        tl.load(Bs + expert * N + col + 4, col + 4 < N, other=0),
        tl.load(Bs + expert * N + col + 8, col + 8 < N, other=0),
        tl.load(Bs + expert * N + col + 12, col + 12 < N, other=0),
    )


@triton.jit
def grouped_mma_epilogue(C, sb, row0, row1, end, nbase, lane, sa, c, N: tl.constexpr):
    p00 = grouped_pack_bfloat16(
        c[0].to(tl.float32) * sa[0] * sb[0], c[1].to(tl.float32) * sa[0] * sb[1]
    )
    p01 = grouped_pack_bfloat16(
        c[2].to(tl.float32) * sa[0] * sb[2], c[3].to(tl.float32) * sa[0] * sb[3]
    )
    p10 = grouped_pack_bfloat16(
        c[4].to(tl.float32) * sa[1] * sb[0], c[5].to(tl.float32) * sa[1] * sb[1]
    )
    p11 = grouped_pack_bfloat16(
        c[6].to(tl.float32) * sa[1] * sb[2], c[7].to(tl.float32) * sa[1] * sb[3]
    )
    lo0, hi0 = grouped_transpose_bfloat16(p00, p01, lane)
    lo1, hi1 = grouped_transpose_bfloat16(p10, p11, lane)
    packed_col = nbase + (lane % 4) * 4
    grouped_store_bfloat16(
        C, row0, packed_col, lo0, hi0, (row0 < end) & (packed_col < N), N
    )
    grouped_store_bfloat16(
        C, row1, packed_col, lo1, hi1, (row1 < end) & (packed_col < N), N
    )


@triton.jit
def grouped_persistent_int8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    Counts,
    Starts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    ROW_MODE: tl.constexpr,
    WARPS: tl.constexpr,
    N_REPS: tl.constexpr,
    N_GROUPS: tl.constexpr,
):
    tl.static_assert(K == 128 and N % 4 == 0)
    BN: tl.constexpr = WARPS * N_REPS * 16
    thread = tl.arange(0, 32 * WARPS * N_REPS)
    lane = thread % 32
    warp = thread // 32
    if ROW_MODE:
        first = tl.program_id(1).to(tl.int64)
        expert = tl.load(RowExperts + first).to(tl.int64)
        count = 1
    else:
        expert = tl.program_id(1).to(tl.int64)
        first = tl.load(Starts + expert).to(tl.int64)
        count = tl.load(Counts + expert).to(tl.int64)
    first_n = tl.program_id(0)
    tiles_n: tl.constexpr = triton.cdiv(N, BN)
    if count == 0 or first_n >= tiles_n:
        return
    smem = tle.gpu.alloc(
        [2 * BN * 128],
        tl.int8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared = tle.gpu.local_ptr(smem, (0,)).to(tl.int64).to(tl.int32)
    bptr = B + expert * N * K
    iteration = 0
    for mb in range(tl.cdiv(count, 16)):
        row0 = first + mb * 16 + lane // 4
        row1 = row0 + 8
        group = grouped_aiu_prefetch(
            shared + (iteration % 2) * BN * 128, bptr, N, K, first_n * BN, BN
        )
        a0 = grouped_a_fragment(A, row0, row1, first + count, lane % 4, 0)
        a1 = grouped_a_fragment(A, row0, row1, first + count, lane % 4, 8)
        a2 = grouped_a_fragment(A, row0, row1, first + count, lane % 4, 16)
        a3 = grouped_a_fragment(A, row0, row1, first + count, lane % 4, 24)
        sa = (
            tl.load(As + row0, row0 < first + count, other=0),
            tl.load(As + row1, row1 < first + count, other=0),
        )
        for nt in range(first_n, tiles_n, N_GROUPS):
            ready = grouped_aiu_wait(group)
            slot = iteration % 2
            current = shared + slot * BN * 128
            if nt + N_GROUPS < tiles_n:
                group = grouped_aiu_prefetch(
                    shared + (1 - slot) * BN * 128, bptr, N, K, (nt + N_GROUPS) * BN, BN
                )
            else:
                group = ready
            zero = tl.full((32 * WARPS * N_REPS,), 0, tl.int32)
            c = (zero, zero, zero, zero, zero, zero, zero, zero)
            c = grouped_mma(
                a0, grouped_aiu_fragment(current, warp * 16, 0, ready, BN), c
            )
            c = grouped_mma(
                a1, grouped_aiu_fragment(current, warp * 16, 32, ready, BN), c
            )
            c = grouped_mma(
                a2, grouped_aiu_fragment(current, warp * 16, 64, ready, BN), c
            )
            c = grouped_mma(
                a3, grouped_aiu_fragment(current, warp * 16, 96, ready, BN), c
            )
            grouped_mma_epilogue(
                C,
                grouped_b_scales(Bs, expert, nt * BN + warp * 16 + lane % 4, N),
                row0,
                row1,
                first + count,
                nt * BN + warp * 16,
                lane,
                sa,
                c,
                N,
            )
            iteration += 1


@triton.jit
def grouped_resident_int8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    Counts,
    Starts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
):
    tl.static_assert(K == 128 and N % 4 == 0)
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 64
    WARPS: tl.constexpr = 2
    N_REPS: tl.constexpr = 2
    thread = tl.arange(0, 32 * WARPS * N_REPS)
    lane = thread % 32
    rep = thread // (32 * WARPS)
    warp = tl.inline_asm_elementwise(
        "{ .reg .u32 tid,w; ppu.mov.u32 tid,%tid.x; ppu.shr.u32 w,tid,5; ppu.to.uniform.b32 $0,w; }",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    expert = tl.program_id(1).to(tl.int64)
    first = tl.load(Starts + expert).to(tl.int64)
    count = tl.load(Counts + expert).to(tl.int64)
    end = first + count
    if count == 0:
        return
    smem = tle.gpu.alloc(
        [(BLOCK_N + 2 * BLOCK_M) * 128],
        tl.int8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared = tle.gpu.local_ptr(smem, (0,)).to(tl.int64).to(tl.int32)
    shared_a = shared + BLOCK_N * 128
    nfirst = tl.program_id(0) * BLOCK_N
    nx = warp * N_REPS * 16 + rep * 16
    nbase = nfirst + nx
    token = grouped_aiu_copy(shared, B + expert * N * K, N, K, nfirst, BLOCK_N)
    token = grouped_aiu_copy(shared_a, A, end, K, first, BLOCK_M)
    group = grouped_aiu_commit(token)
    initial = grouped_aiu_wait(group)
    b = (
        grouped_aiu_fragment(shared, nx, 0, initial, BLOCK_N),
        grouped_aiu_fragment(shared, nx, 32, initial, BLOCK_N),
        grouped_aiu_fragment(shared, nx, 64, initial, BLOCK_N),
        grouped_aiu_fragment(shared, nx, 96, initial, BLOCK_N),
    )
    sb = grouped_b_scales(Bs, expert, nbase + lane % 4, N)
    for mb in range(tl.cdiv(count, BLOCK_M)):
        if mb == 0:
            ready = initial
        else:
            ready = grouped_aiu_wait(group)
        row = first + mb * BLOCK_M
        current = shared_a + (mb % 2) * BLOCK_M * 128
        if (mb + 1) * BLOCK_M < count:
            group = grouped_aiu_prefetch(
                shared_a + ((mb + 1) % 2) * BLOCK_M * 128,
                A,
                end,
                K,
                row + BLOCK_M,
                BLOCK_M,
            )
        else:
            group = ready
        zero = tl.full((32 * WARPS * N_REPS,), 0, tl.int32)
        c0 = (zero, zero, zero, zero, zero, zero, zero, zero)
        c1 = (zero, zero, zero, zero, zero, zero, zero, zero)
        for sub in tl.static_range(4):
            a0 = grouped_aiu_fragment(current, 0, sub * 32, ready, BLOCK_M)
            c0 = grouped_mma(a0, b[sub], c0)
            a1 = grouped_aiu_fragment(current, 16, sub * 32, ready, BLOCK_M)
            c1 = grouped_mma(a1, b[sub], c1)
        r0 = row + lane // 4
        r1 = r0 + 8
        sa = (tl.load(As + r0, r0 < end, other=0), tl.load(As + r1, r1 < end, other=0))
        grouped_mma_epilogue(C, sb, r0, r1, end, nbase, lane, sa, c0, N)
        r0 = row + 16 + lane // 4
        r1 = r0 + 8
        sa = (tl.load(As + r0, r0 < end, other=0), tl.load(As + r1, r1 < end, other=0))
        grouped_mma_epilogue(C, sb, r0, r1, end, nbase, lane, sa, c1, N)


@triton.jit
def load_int8_vector(pointer, mask):
    return tl.inline_asm_elementwise(
        "{ ppu.mov.u32 $0, 0; ppu.mov.u32 $1, 0; ppu.mov.u32 $2, 0; "
        "ppu.mov.u32 $3, 0; @$5 ppu.ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4]; }",
        "=r,=r,=r,=r,l,b",
        [pointer, mask],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_dot_vector(a, b, acc):
    return tl.inline_asm_elementwise(
        "{ .reg .s32 partial; dp4a.s32.s32 partial,$1,$5,$9; "
        "dp4a.s32.s32 partial,$2,$6,partial; "
        "dp4a.s32.s32 partial,$3,$7,partial; "
        "dp4a.s32.s32 $0,$4,$8,partial; }",
        "=r,r,r,r,r,r,r,r,r,r",
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], acc],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_stream_copy(shared, source, dependency):
    return tl.inline_asm_elementwise(
        "ppu.cp.async.cg.shared.global [$1],[$2],16; "
        "ppu.cp.async.commit_group; ppu.mov.u32 $0,0;",
        "=r,r,l,r,~{memory}",
        [shared, source, dependency],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_stream_wait(dependency):
    return tl.inline_asm_elementwise(
        "ppu.cp.async.wait_group 0; ppu.mov.u32 $0,0;",
        "=r,r,~{memory}",
        [dependency],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_stream_fragment(shared, dependency):
    return tl.inline_asm_elementwise(
        "ppu.ld.shared.v4.b32 {$0,$1,$2,$3},[$4];",
        "=r,=r,=r,=r,r,r,~{memory}",
        [shared, dependency],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def grouped_stream_dp4a(a, b, acc):
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0,$1,$2,$3;",
        "=r,r,r,r",
        [a, b, acc],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_streaming_gemv_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
):
    tl.static_assert(K % 512 == 0 and K >= 1024)
    WARPS: tl.constexpr = 1
    thread = tl.arange(0, 32 * WARPS)
    row = tl.program_id(1).to(tl.int64)
    expert = tl.load(RowExperts + row).to(tl.int64)
    col = tl.program_id(0) * WARPS + thread // 32
    kw = (thread % 32) * 16
    smem = tle.gpu.alloc(
        [2 * 32 * WARPS * 16],
        tl.int8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    bptr = B + expert * N * K + col * K + kw
    aptr = A + row * K + kw
    shared = tle.gpu.local_ptr(smem, (thread * 16,)).to(tl.int64).to(tl.int32)
    group = grouped_stream_copy(shared, bptr, 0)
    acc = tl.full((32 * WARPS,), 0, tl.int32)
    for tile in range(K // 512):
        ready = grouped_stream_wait(group)
        if tile + 1 < K // 512:
            following = (
                tle.gpu.local_ptr(
                    smem, ((((tile + 1) % 2) * 32 * WARPS + thread) * 16,)
                )
                .to(tl.int64)
                .to(tl.int32)
            )
            group = grouped_stream_copy(following, bptr + (tile + 1) * 512, ready)
        else:
            group = ready
        current = (
            tle.gpu.local_ptr(smem, (((tile % 2) * 32 * WARPS + thread) * 16,))
            .to(tl.int64)
            .to(tl.int32)
        )
        b = grouped_stream_fragment(current, group)
        a = load_int8_vector(aptr + tile * 512, tl.full((32 * WARPS,), True, tl.int1))
        for word in tl.static_range(4):
            acc = grouped_stream_dp4a(a[word], b[word], acc)
    value = tl.sum(tl.reshape(acc, (WARPS, 32)), 1).to(tl.float32)
    cols = tl.program_id(0) * WARPS + tl.arange(0, WARPS)
    scale_a = tl.load(As + row)
    scale_b = tl.load(Bs + expert * N + cols)
    tl.store(C + row * N + cols, value * scale_a * scale_b)


@triton.jit
def grouped_vector_expert_kernel(
    A,
    As,
    B,
    Bs,
    C,
    Counts,
    Starts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K16: tl.constexpr,
    UNROLL: tl.constexpr,
):
    expert = tl.program_id(1)
    first = tl.load(Starts + expert)
    count = tl.load(Counts + expert)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    vectors = tl.arange(0, BLOCK_K16)
    for block in range(tl.cdiv(count, BLOCK_M)):
        rows = first + block * BLOCK_M + tl.arange(0, BLOCK_M)
        aptr = A + rows[:, None] * K + vectors[None, :] * 16
        bptr = (
            B + expert.to(tl.int64) * N * K + cols[:, None] * K + vectors[None, :] * 16
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K16), tl.int32)
        for tile in tl.range(tl.cdiv(K, 16 * BLOCK_K16), loop_unroll_factor=UNROLL):
            valid_k = vectors + tile * BLOCK_K16 < K // 16
            a = load_int8_vector(
                aptr + tile * BLOCK_K16 * 16,
                (rows[:, None] < first + count) & valid_k[None, :],
            )
            b = load_int8_vector(
                bptr + tile * BLOCK_K16 * 16, (cols[:, None] < N) & valid_k[None, :]
            )
            acc = grouped_dot_vector(
                (
                    a[0][:, None, :],
                    a[1][:, None, :],
                    a[2][:, None, :],
                    a[3][:, None, :],
                ),
                (
                    b[0][None, :, :],
                    b[1][None, :, :],
                    b[2][None, :, :],
                    b[3][None, :, :],
                ),
                acc,
            )
        value = tl.sum(acc, 2).to(tl.float32)
        scale_a = tl.load(As + rows, rows < first + count, other=0)
        scale_b = tl.load(Bs + expert * N + cols, cols < N, other=0)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            value * scale_a[:, None] * scale_b[None, :],
            (rows[:, None] < first + count) & (cols[None, :] < N),
        )


@triton.jit
def grouped_gemv_vector_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K16: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_n = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * tiles_n)
    group_rows = tl.minimum(M - group * GROUP_M, GROUP_M)
    local_pid = pid % (GROUP_M * tiles_n)
    row = (group * GROUP_M + local_pid % group_rows).to(tl.int64)
    cols = local_pid // group_rows * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.load(RowExperts + row).to(tl.int64)
    vectors = tl.arange(0, BLOCK_K16)
    a_ptr = A + row * K + vectors * 16
    b_ptr = B + expert * N * K + cols[:, None] * K + vectors[None, :] * 16
    acc = tl.zeros((BLOCK_N, BLOCK_K16), tl.int32)
    for tile in range(tl.cdiv(K, 16 * BLOCK_K16)):
        valid_k = vectors + tile * BLOCK_K16 < K // 16
        a0, a1, a2, a3 = load_int8_vector(a_ptr + tile * BLOCK_K16 * 16, valid_k)
        b0, b1, b2, b3 = load_int8_vector(
            b_ptr + tile * BLOCK_K16 * 16, (cols[:, None] < N) & valid_k[None, :]
        )
        acc = grouped_dot_vector(
            (a0[None, :], a1[None, :], a2[None, :], a3[None, :]),
            (b0, b1, b2, b3),
            acc,
        )
    dot = tl.sum(acc, 1).to(tl.float32)
    scale_a = tl.load(As + row)
    scale_b = tl.load(Bs + expert * N + cols, cols < N, other=0)
    tl.store(C + row * N + cols, dot * scale_a * scale_b, cols < N)


@triton.jit
def packed_gemv_step(
    a_ptrs,
    b_ptrs,
    prefetch_ptrs,
    acc,
    mask_k,
    prefetch_mask_k,
    mask_n,
    CACHE_B: tl.constexpr,
    EVICT_B: tl.constexpr,
    PREFETCH_B: tl.constexpr,
):
    if PREFETCH_B:
        _ = tl.inline_asm_elementwise(
            "@$2 ppu.prefetch.global.L2 [ $1 ]; ppu.mov.u32 $0, 0;",
            "=r,l,b",
            [prefetch_ptrs, mask_n[:, None] & prefetch_mask_k[None, :]],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    a = tl.load(a_ptrs, mask_k, other=0)
    b = tl.load(
        b_ptrs,
        mask_n[:, None] & mask_k[None, :],
        other=0,
        cache_modifier=CACHE_B,
        eviction_policy=EVICT_B,
    )
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[a[None, :], b, acc],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_gemv_int8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K4: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
    TLE_PIPELINE: tl.constexpr,
    CACHE_B: tl.constexpr = "",
    EVICT_B: tl.constexpr = "",
    PREFETCH_B: tl.constexpr = False,
    PREFETCH_DISTANCE: tl.constexpr = 1,
):
    tl.static_assert(K % 4 == 0 and WARPS > 0)
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.load(RowExperts + row).to(tl.int64)
    words = tl.arange(0, BLOCK_K4)
    a_words = A.to(tl.pointer_type(tl.int32)) + row * (K // 4) + words
    b_words = B.to(tl.pointer_type(tl.int32)) + expert * N * (K // 4)
    b_words += cols[:, None] * (K // 4) + words[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_K4), tl.int32)
    if TLE_PIPELINE:
        for tile in tle.gpu.pipeline(0, tl.cdiv(K // 4, BLOCK_K4), num_stages=STAGES):
            offset = tile * BLOCK_K4
            acc = packed_gemv_step(
                a_words + offset,
                b_words + offset,
                b_words + offset + PREFETCH_DISTANCE * BLOCK_K4,
                acc,
                words + offset < K // 4,
                words + offset + PREFETCH_DISTANCE * BLOCK_K4 < K // 4,
                cols < N,
                CACHE_B,
                EVICT_B,
                PREFETCH_B,
            )
    else:
        for tile in range(tl.cdiv(K // 4, BLOCK_K4)):
            offset = tile * BLOCK_K4
            acc = packed_gemv_step(
                a_words + offset,
                b_words + offset,
                b_words + offset + PREFETCH_DISTANCE * BLOCK_K4,
                acc,
                words + offset < K // 4,
                words + offset + PREFETCH_DISTANCE * BLOCK_K4 < K // 4,
                cols < N,
                CACHE_B,
                EVICT_B,
                PREFETCH_B,
            )
    dot = tl.sum(acc, 1).to(tl.float32)
    scale_a = tl.load(As + row)
    scale_b = tl.load(Bs + expert * N + cols, cols < N, other=0)
    tl.store(C + row * N + cols, dot * scale_a * scale_b, cols < N)


@triton.jit
def grouped_dot_step(
    a_ptr,
    b_ptr,
    acc,
    BLOCK_K: tl.constexpr,
    SWAP: tl.constexpr,
    ASYNC_A: tl.constexpr,
    ASYNC_B: tl.constexpr,
):
    a = tle.load(a_ptr, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC_A)
    b = tle.load(b_ptr, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC_B)
    if SWAP:
        acc = tl.dot(b, tl.trans(a), acc, out_dtype=tl.int32)
    else:
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.int32)
    return tl.advance(a_ptr, (0, BLOCK_K)), tl.advance(b_ptr, (0, BLOCK_K)), acc


@triton.jit
def grouped_gemm_compute_tile(
    A,
    As,
    B,
    Bs,
    C,
    SortedRows,
    row_offset,
    first_row,
    expert,
    tile_n,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
    SWAP: tl.constexpr,
    TLE_PIPELINE: tl.constexpr,
    NAIVE: tl.constexpr,
    ASYNC_A: tl.constexpr,
    ASYNC_B: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_M)
    if NAIVE:
        rows = tl.where(lanes == 0, first_row, M)
    else:
        rows = tl.load(SortedRows + row_offset + lanes)
    # Extra rows in the physical block are ignored by the output mask. The
    # descriptor's full allocation extent keeps the final group in bounds.
    a_ptr = tl.make_block_ptr(
        A,
        shape=(M, K),
        strides=(K, 1),
        offsets=(first_row, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_ptr = tl.make_block_ptr(
        B + expert * N * K,
        shape=(N, K),
        strides=(K, 1),
        offsets=(tile_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    if SWAP:
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    if TLE_PIPELINE:
        for _ in tle.gpu.pipeline(0, tl.cdiv(K, BLOCK_K), num_stages=STAGES):
            a_ptr, b_ptr, acc = grouped_dot_step(
                a_ptr, b_ptr, acc, BLOCK_K, SWAP, ASYNC_A, ASYNC_B
            )
    else:
        for _ in range(tl.cdiv(K, BLOCK_K)):
            a_ptr, b_ptr, acc = grouped_dot_step(
                a_ptr, b_ptr, acc, BLOCK_K, SWAP, ASYNC_A, ASYNC_B
            )
    if SWAP:
        acc = tl.trans(acc)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scale = tl.load(As + rows, rows < M, other=0)
    b_scale = tl.load(Bs + expert * N + cols, cols < N, other=0)
    value = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    tl.store(
        C + rows[:, None].to(tl.int64) * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def grouped_gemm_tiled_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    Rows,
    SortedRows,
    BlockExperts,
    PaddedRows,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
    SWAP: tl.constexpr,
    TLE_PIPELINE: tl.constexpr,
    NAIVE: tl.constexpr,
    GROUP_M: tl.constexpr = 0,
    ADAPTIVE_M: tl.constexpr = False,
    ASYNC_A: tl.constexpr = False,
    ASYNC_B: tl.constexpr = False,
    SCALAR_PROBE: tl.constexpr = False,
):
    tl.static_assert(WARPS > 0)
    if GROUP_M == 0:
        tile_m = tl.program_id(0)
        tile_n = tl.program_id(1)
        if tile_m * BLOCK_M >= tl.load(PaddedRows):
            return
    else:
        pid = tl.program_id(0)
        num_n = tl.cdiv(N, BLOCK_N)
        tile_m = pid // (GROUP_M * num_n) * GROUP_M + pid % GROUP_M
        tile_n = pid // GROUP_M % num_n
        if tile_m * BLOCK_M >= tl.load(PaddedRows):
            return
    row_offset = tile_m * BLOCK_M
    if NAIVE:
        first_row = tile_m
        expert = tl.load(RowExperts + tile_m).to(tl.int64)
    else:
        first_row = tl.load(SortedRows + row_offset)
        expert = tl.load(BlockExperts + tile_m).to(tl.int64)
    if ADAPTIVE_M and not NAIVE and BLOCK_M > 32:
        if SCALAR_PROBE:
            has16 = tl.load(SortedRows + row_offset + 16) < M
            has32 = tl.load(SortedRows + row_offset + 32) < M
            if BLOCK_M > 64:
                has64 = tl.load(SortedRows + row_offset + 64) < M
            else:
                has64 = False
            valid_rows = tl.where(
                has64, 128, tl.where(has32, 64, tl.where(has16, 32, 16))
            )
        else:
            rows = tl.load(SortedRows + row_offset + tl.arange(0, BLOCK_M))
            valid_rows = tl.sum((rows < M).to(tl.int32), 0)
        if valid_rows <= 16:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                16,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
        elif valid_rows <= 32:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                32,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
        elif BLOCK_M > 64 and valid_rows <= 64:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                64,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
        else:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
    else:
        grouped_gemm_compute_tile(
            A,
            As,
            B,
            Bs,
            C,
            SortedRows,
            row_offset,
            first_row,
            expert,
            tile_n,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            STAGES,
            WARPS,
            SWAP,
            TLE_PIPELINE,
            NAIVE,
            ASYNC_A,
            ASYNC_B,
        )


@libentry()
@triton.jit
def grouped_gemm_metadata(
    RowExperts,
    Rows,
    SortedRows,
    BlockExperts,
    PaddedRows,
    M: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NAIVE: tl.constexpr = False,
):
    if NAIVE:
        tl.store(PaddedRows, M * BLOCK_M)
    else:
        expert = tl.program_id(0)
        groups = tl.arange(0, BLOCK_E)
        counts = tl.load(Rows + groups, groups < E, other=0)
        padded_counts = tl.cdiv(counts, BLOCK_M) * BLOCK_M
        start = tl.sum(tl.where(groups < expert, counts, 0), 0)
        padded_start = tl.sum(tl.where(groups < expert, padded_counts, 0), 0)
        count = tl.load(Rows + expert)
        expert_id = tl.load(RowExperts + start, count > 0, other=-1)
        if expert == 0:
            tl.store(PaddedRows, tl.sum(padded_counts, 0))
        lanes = tl.arange(0, BLOCK_M)
        for block in range(tl.cdiv(count, BLOCK_M)):
            offsets = block * BLOCK_M + lanes
            rows = tl.where(offsets < count, start + offsets, M)
            tl.store(SortedRows + padded_start + offsets, rows)
            tl.store(BlockExperts + padded_start // BLOCK_M + block, expert_id)


def prune_grouped_configs(configs, named_args, **kwargs):
    m, n, k = named_args["M"], named_args["N"], named_args["K"]
    return [
        config
        for config in configs
        if (not config.kwargs["NAIVE"] or m <= 512)
        and (
            not (config.kwargs["ASYNC_A"] or config.kwargs["ASYNC_B"])
            or (n % config.kwargs["BLOCK_N"] == 0 and k % config.kwargs["BLOCK_K"] == 0)
        )
    ]


grouped_gemm_configs = runtime.get_tuned_config("grouped_gemm_w8a8_gemm")
MAX_GROUPED_BLOCK_M = max(config.kwargs["BLOCK_M"] for config in grouped_gemm_configs)
MIN_GROUPED_BLOCK_M = min(config.kwargs["BLOCK_M"] for config in grouped_gemm_configs)

grouped_gemm_tuner = libtuner(
    configs=grouped_gemm_configs,
    key=["M", "N", "K", "E"],
    prune_configs_by={"early_config_prune": prune_grouped_configs},
    warmup=5,
    rep=20,
)(grouped_gemm_tiled_kernel)
autotuned_grouped_gemm = libentry()(grouped_gemm_tuner)
autotuned_grouped_gemv = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("grouped_gemm_w8a8_gemv"),
        key=["M", "N", "K", "E"],
        warmup=5,
        rep=20,
    )(grouped_gemv_int8_kernel)
)
autotuned_grouped_prefetch_gemv = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("grouped_gemm_w8a8_gemv_prefetch"),
        key=["M", "N", "K", "E"],
        warmup=5,
        rep=20,
    )(grouped_gemv_int8_kernel)
)
autotuned_grouped_vector_gemv = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("grouped_gemm_w8a8_gemv_vector"),
        key=["M", "N", "K", "E"],
        warmup=5,
        rep=20,
    )(grouped_gemv_vector_kernel)
)


autotuned_grouped_persistent = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("grouped_gemm_w8a8_persistent"),
        key=["M", "N", "K", "E", "ROW_MODE"],
        warmup=5,
        rep=20,
    )(grouped_persistent_int8_kernel)
)


@triton.jit
def grouped_vector_choice_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    Counts,
    Starts,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    NAIVE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr = 1,
    BLOCK_K16: tl.constexpr = 32,
    UNROLL: tl.constexpr = 1,
    BLOCK_K4: tl.constexpr = 128,
    STAGES: tl.constexpr = 4,
    WARPS: tl.constexpr = 1,
    TLE_PIPELINE: tl.constexpr = False,
    CACHE_B: tl.constexpr = "",
    EVICT_B: tl.constexpr = "",
):
    if NAIVE:
        grouped_gemv_int8_kernel(
            A,
            As,
            B,
            Bs,
            C,
            RowExperts,
            M,
            N,
            K,
            E,
            BLOCK_N,
            BLOCK_K4,
            STAGES,
            WARPS,
            TLE_PIPELINE,
            CACHE_B,
            EVICT_B,
        )
    else:
        grouped_vector_expert_kernel(
            A,
            As,
            B,
            Bs,
            C,
            Counts,
            Starts,
            M,
            N,
            K,
            E,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K16,
            UNROLL,
        )


# Benchmark tables require the same fields for both algorithms.
grouped_vector_expert_configs = [
    triton.Config(
        {
            "BLOCK_M": 1,
            "BLOCK_K16": 32,
            "UNROLL": 1,
            "BLOCK_K4": 128,
            "STAGES": 4,
            "WARPS": 1,
            "TLE_PIPELINE": False,
            "CACHE_B": "",
            "EVICT_B": "",
            **config.kwargs,
        },
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
    )
    for config in runtime.get_tuned_config("grouped_gemm_w8a8_vector_expert")
]
MAX_GROUPED_VECTOR_M = max(
    config.kwargs["BLOCK_M"]
    for config in grouped_vector_expert_configs
    if not config.kwargs["NAIVE"]
)
autotuned_grouped_vector_expert = libentry()(
    libtuner(
        configs=grouped_vector_expert_configs,
        key=["M", "N", "K", "E"],
        warmup=5,
        rep=20,
    )(grouped_vector_choice_kernel)
)


def grouped_gemm_w8a8(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    """Write per-channel INT8 grouped NT GEMM into the supplied BF16 output.

    Rows must be contiguous groups in ascending expert order. m_rows contains
    nonnegative lengths summing to M; m_indices identifies each row's expert.
    These device-value preconditions are supplied by the caller without a host
    synchronization. Only routing metadata is padded; A, B and C stay unpadded.
    """
    logger.debug("GEMS_THEAD GROUPED_GEMM_W8A8")
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("A must be [M,K] and B must be [E,N,K]")
    M, K = A.shape
    E, N, weight_k = B.shape
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    expected_dtypes = (
        torch.int8,
        torch.float32,
        torch.int8,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int32,
    )
    if any(t.dtype != dtype for t, dtype in zip(tensors, expected_dtypes)):
        raise NotImplementedError(
            "requires INT8 A/B, FP32 scales, BF16 C, INT32 indices"
        )
    if A.device.type != "cuda" or any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same PPU device")
    if not all(t.is_contiguous() for t in tensors):
        raise NotImplementedError("only contiguous tensors are supported")
    expected_shapes = ((M, 1), (E, N, 1), (M, N), (M,), (E,))
    if weight_k != K or any(
        tuple(t.shape) != shape
        for t, shape in zip((As, Bs, C, m_indices, m_rows), expected_shapes)
    ):
        raise ValueError("incompatible scales, output or expert metadata shapes")
    if K > 131071 or (M > 0 and E == 0):
        raise ValueError("requires K <= 131071 and experts for nonempty input")
    c_start = C.data_ptr()
    c_end = c_start + C.numel() * C.element_size()
    for tensor in (A, As, B, Bs, m_indices, m_rows):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if c_start < end and start < c_end:
            raise ValueError("C must not overlap an input")
    if M == 0 or N == 0:
        return
    props = torch_device_fn.get_device_properties(A.device.index)
    if (props.name, props.major, props.minor, props.warp_size) != (
        "PPU-ZW810E",
        8,
        0,
        32,
    ):
        raise NotImplementedError("grouped INT8 GEMM requires PPU-ZW810E")

    if (
        K >= LARGE_REDUCTION_MIN
        and K % 512 == 0
        and M <= STREAMING_GEMV_ROW_LIMIT
        and N % 4 == 0
        and A.data_ptr() % 16 == 0
        and B.data_ptr() % 16 == 0
    ):
        with torch_device_fn.device(A.device):
            grouped_streaming_gemv_kernel[(N, M)](
                A,
                As,
                B,
                Bs,
                C,
                m_indices,
                M,
                N,
                K,
                E,
                num_warps=1,
                num_stages=1,
            )
        return

    if (
        K >= LARGE_REDUCTION_MIN
        and K % 768 == 0
        and (
            max(GROUPED_VECTOR_MIN_ROWS, 4 * E) < M <= PIPELINE_MAX_ROWS_PER_EXPERT * E
            or M >= PIPELINE_DENSE_ROWS_PER_EXPERT * E
        )
        and 4 <= N <= 256
        and N % 4 == 0
        and A.data_ptr() % 128 == 0
        and B.data_ptr() % 128 == 0
        and C.data_ptr() % 8 == 0
    ):
        is_dense = M >= PIPELINE_DENSE_ROWS_PER_EXPERT * E
        block_m = PIPELINE_DENSE_BLOCK_M if is_dense else PIPELINE_BLOCK_M
        block_n = 128 if is_dense else 64
        block_k = 128 if is_dense else 256
        stages = 2 if is_dense else 3
        tile_capacity = triton.cdiv(M, block_m) + E
        tiles = torch.empty(1 + 3 * tile_capacity, dtype=torch.int32, device=A.device)
        with torch_device_fn.device(A.device):
            grouped_pipeline_metadata[(E,)](
                m_rows,
                tiles,
                E,
                triton.next_power_of_2(E),
                block_m,
                num_warps=4,
            )
            grouped_pipeline_int8_kernel[(PIPELINE_PROGRAMS,)](
                A,
                As,
                B,
                Bs,
                C,
                tiles,
                N,
                K,
                BM=block_m,
                BN=block_n,
                BK=block_k,
                STAGES=stages,
                PROGRAMS=PIPELINE_PROGRAMS,
                num_warps=4,
                num_stages=1,
            )
        return

    if (
        K >= LARGE_REDUCTION_MIN
        and K % 512 == 0
        and N >= 32
        and GROUPED_VECTOR_MIN_ROWS <= M
        and 2 * E <= M <= 4 * E
        and (M + MAX_GROUPED_VECTOR_M) * K < 2**31
        and E * N * K < 2**31
        and 2 * (M + MAX_GROUPED_VECTOR_M) * N < 2**31
        and A.data_ptr() % 16 == 0
        and B.data_ptr() % 16 == 0
    ):
        starts = torch.empty_like(m_rows)

        def vector_grid(meta):
            if meta["NAIVE"]:
                return triton.cdiv(N, meta["BLOCK_N"]), M
            else:
                grouped_expert_offsets[(1,)](
                    m_rows,
                    starts,
                    E,
                    triton.next_power_of_2(E),
                    num_warps=4,
                )
                return triton.cdiv(N, meta["BLOCK_N"]), E

        with torch_device_fn.device(A.device):
            autotuned_grouped_vector_expert[vector_grid](
                A, As, B, Bs, C, m_indices, m_rows, starts, M, N, K, E
            )
        return

    is_row_prefetch = M <= min(E, ROW_PREFETCH_LIMIT) or (
        M >= STREAMING_GEMV_ROW_LIMIT and E // 2 < M <= E
    )
    if (
        K == PREFETCH_REDUCTION_LIMIT
        and (is_row_prefetch or E <= M <= 16 * E)
        and N % 4 == 0
        and A.data_ptr() % 128 == 0
        and B.data_ptr() % 128 == 0
        and C.data_ptr() % 8 == 0
    ):
        grid = lambda meta: (
            min(meta["N_GROUPS"], triton.cdiv(N, 16 * meta["WARPS"] * meta["N_REPS"])),
            M if is_row_prefetch else E,
        )
        with torch_device_fn.device(A.device):
            if is_row_prefetch:
                starts = None
            else:
                starts = torch.empty_like(m_rows)
                grouped_expert_offsets[(1,)](
                    m_rows, starts, E, triton.next_power_of_2(E), num_warps=4
                )
            autotuned_grouped_persistent[grid](
                A,
                As,
                B,
                Bs,
                C,
                m_indices,
                m_rows,
                starts,
                M,
                N,
                K,
                E,
                ROW_MODE=is_row_prefetch,
            )
        return

    if (
        M <= GEMV_ROW_LIMIT
        and K > 0
        and K % 4 == 0
        and A.data_ptr() % 4 == 0
        and B.data_ptr() % 4 == 0
    ):
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)
        with torch_device_fn.device(A.device):
            if (
                K == PREFETCH_REDUCTION_LIMIT
                and A.data_ptr() % 16 == 0
                and B.data_ptr() % 16 == 0
            ):
                vector_grid = lambda meta: (M * triton.cdiv(N, meta["BLOCK_N"]),)
                autotuned_grouped_vector_gemv[vector_grid](
                    A, As, B, Bs, C, m_indices, M, N, K, E
                )
            elif K <= PREFETCH_REDUCTION_LIMIT:
                autotuned_grouped_prefetch_gemv[grid](
                    A, As, B, Bs, C, m_indices, M, N, K, E
                )
            else:
                autotuned_grouped_gemv[grid](A, As, B, Bs, C, m_indices, M, N, K, E)
        return

    if (
        K == PREFETCH_REDUCTION_LIMIT
        and M >= RESIDENT_ROWS_PER_EXPERT * E
        and N % 4 == 0
        and A.data_ptr() % 128 == 0
        and B.data_ptr() % 128 == 0
        and C.data_ptr() % 8 == 0
    ):
        starts = torch.empty_like(m_rows)
        with torch_device_fn.device(A.device):
            grouped_expert_offsets[(1,)](
                m_rows, starts, E, triton.next_power_of_2(E), num_warps=4
            )
            grouped_resident_int8_kernel[(triton.cdiv(N, 64), E)](
                A,
                As,
                B,
                Bs,
                C,
                m_rows,
                starts,
                M,
                N,
                K,
                E,
                num_warps=2,
                num_stages=1,
            )
        return

    if (
        K > 0
        and K % 128 == 0
        and N % 64 == 0
        and A.data_ptr() % 128 == 0
        and B.data_ptr() % 128 == 0
    ):
        sorted_rows = torch.empty(
            (M + E * (MAX_GROUPED_BLOCK_M - 1),), dtype=torch.int32, device=A.device
        )
        block_experts = torch.empty(
            (triton.cdiv(M, MIN_GROUPED_BLOCK_M) + E,),
            dtype=torch.int32,
            device=A.device,
        )
        padded_rows = torch.empty((1,), dtype=torch.int32, device=A.device)

        def _grid(meta):
            # Grid callbacks retain the selected tile size on cached launches and tuning trials.
            grouped_gemm_metadata[(1 if meta["NAIVE"] else E,)](
                m_indices,
                m_rows,
                sorted_rows,
                block_experts,
                padded_rows,
                M,
                E,
                BLOCK_M=meta["BLOCK_M"],
                BLOCK_E=triton.next_power_of_2(E),
                NAIVE=meta["NAIVE"],
                num_warps=4,
            )
            tiles_m = M if meta["NAIVE"] else triton.cdiv(M, meta["BLOCK_M"]) + E - 1
            tiles_n = triton.cdiv(N, meta["BLOCK_N"])
            if meta["GROUP_M"] == 0:
                return tiles_m, tiles_n
            else:
                # Pad the last group to keep the kernel's index divisions constant.
                tiles_m = triton.cdiv(tiles_m, meta["GROUP_M"]) * meta["GROUP_M"]
                return (tiles_m * tiles_n,)

        with torch_device_fn.device(A.device):
            autotuned_grouped_gemm[_grid](
                A,
                As,
                B,
                Bs,
                C,
                m_indices,
                m_rows,
                sorted_rows,
                block_experts,
                padded_rows,
                M,
                N,
                K,
                E,
                triton.next_power_of_2(E),
            )
        return

    block_m = 16 if M <= 64 * E else 64
    # Larger reductions and batches benefit from reusing weights across grouped rows.
    use_naive = M <= min(E, SMALL_ROW_LIMIT) and K <= SHORT_REDUCTION_LIMIT
    if use_naive:
        sorted_rows = None
        block_experts = m_indices
    else:
        capacity = triton.cdiv(M + E * (block_m - 1), block_m) * block_m
        sorted_rows = torch.empty((capacity,), dtype=torch.int32, device=A.device)
        block_experts = torch.empty(
            (capacity // block_m,), dtype=torch.int32, device=A.device
        )
    padded_rows = torch.empty((1,), dtype=torch.int32, device=A.device)
    with torch_device_fn.device(A.device):
        grouped_gemm_metadata[(1 if use_naive else E,)](
            m_indices,
            m_rows,
            sorted_rows,
            block_experts,
            padded_rows,
            M,
            E,
            BLOCK_M=block_m,
            BLOCK_E=triton.next_power_of_2(E),
            NAIVE=use_naive,
            num_warps=4,
        )
        tiles_m = M if use_naive else triton.cdiv(capacity, block_m)
        # Reuse the synchronous INT32 tile for tails without a vLLM MoE dependency.
        grouped_gemm_tiled_kernel[(tiles_m, triton.cdiv(N, 64))](
            A,
            As,
            B,
            Bs,
            C,
            m_indices,
            m_rows,
            sorted_rows,
            block_experts,
            padded_rows,
            M,
            N,
            K,
            E,
            triton.next_power_of_2(E),
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=128,
            STAGES=2,
            WARPS=4,
            SWAP=False,
            TLE_PIPELINE=False,
            NAIVE=use_naive,
            num_warps=4,
            num_stages=2,
        )
