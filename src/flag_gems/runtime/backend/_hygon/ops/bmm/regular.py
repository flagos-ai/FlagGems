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

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.utils import libentry


@triton.jit
def ax(m, k, BK: tl.constexpr, V: tl.constexpr):
    return m * BK + ((k // V) ^ (m % (BK // V))) * V + k % V


@triton.jit
def bx(k, n, BN: tl.constexpr, V: tl.constexpr):
    return k * BN + ((n // V) ^ (k % 4)) * V + n % V


_ASM_0_64_64_128_2_2 = tl.constexpr("""v_readfirstlane_b32 s32, $16
v_readfirstlane_b32 s33, $17
v_readfirstlane_b32 s34, $18
v_readfirstlane_b32 s36, $19
v_readfirstlane_b32 s37, $20
v_readfirstlane_b32 s38, $21
v_readfirstlane_b32 s47, $22
v_readfirstlane_b32 s48, $23
v_readfirstlane_b32 s46, $24
s_mov_b32 s35, 0x20000
s_mov_b32 s39, 0x20000
s_mov_b32 s44,0
s_mov_b32 s45,0
v_mov_b32 v0,0
v_mov_b32 v1,0
v_mov_b32 v2,0
v_mov_b32 v3,0
v_mov_b32 v4,0
v_mov_b32 v5,0
v_mov_b32 v6,0
v_mov_b32 v7,0
v_mov_b32 v8,0
v_mov_b32 v9,0
v_mov_b32 v10,0
v_mov_b32 v11,0
v_mov_b32 v12,0
v_mov_b32 v13,0
v_mov_b32 v14,0
v_mov_b32 v15,0
buffer_load_dwordx4 v[32:35], $25, s[32:35], s44 offen
buffer_load_dwordx4 v[36:39], $27, s[32:35], s44 offen
buffer_load_dwordx4 v[40:43], $29, s[32:35], s44 offen
buffer_load_dwordx4 v[44:47], $31, s[32:35], s44 offen
buffer_load_dwordx4 v[48:51], $33, s[36:39], s45 offen
buffer_load_dwordx4 v[52:55], $35, s[36:39], s45 offen
buffer_load_dwordx4 v[56:59], $37, s[36:39], s45 offen
buffer_load_dwordx4 v[60:63], $39, s[36:39], s45 offen
s_waitcnt vmcnt(0)
ds_write_b128 $26, v[32:35]
ds_write_b128 $28, v[36:39]
ds_write_b128 $30, v[40:43]
ds_write_b128 $32, v[44:47]
ds_write_b128 $34, v[48:51]
ds_write_b128 $36, v[52:55]
ds_write_b128 $38, v[56:59]
ds_write_b128 $40, v[60:63]
s_waitcnt lgkmcnt(0)
s_barrier
Lregular_loop:
s_add_u32 s44,s44,s47
s_add_u32 s45,s45,s48
buffer_load_dwordx4 v[32:35], $25, s[32:35], s44 offen
buffer_load_dwordx4 v[36:39], $27, s[32:35], s44 offen
buffer_load_dwordx4 v[40:43], $29, s[32:35], s44 offen
buffer_load_dwordx4 v[44:47], $31, s[32:35], s44 offen
buffer_load_dwordx4 v[48:51], $33, s[36:39], s45 offen
buffer_load_dwordx4 v[52:55], $35, s[36:39], s45 offen
buffer_load_dwordx4 v[56:59], $37, s[36:39], s45 offen
buffer_load_dwordx4 v[60:63], $39, s[36:39], s45 offen
ds_read_b64 v[16:17], $41
ds_read_b64 v[18:19], $42
ds_read_m32x16_b16 v[20:23], $43
ds_read_b64 v[24:25], $44
ds_read_b64 v[26:27], $45
ds_read_m32x16_b16 v[28:31], $46
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[20:21], v[16:17], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[22:23], v[16:17], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[20:21], v[18:19], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[22:23], v[18:19], v[12:15]
ds_read_b64 v[16:17], $47
ds_read_b64 v[18:19], $48
ds_read_m32x16_b16 v[20:23], $49
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[28:29], v[24:25], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[30:31], v[24:25], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[28:29], v[26:27], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[30:31], v[26:27], v[12:15]
ds_read_b64 v[24:25], $50
ds_read_b64 v[26:27], $51
ds_read_m32x16_b16 v[28:31], $52
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[20:21], v[16:17], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[22:23], v[16:17], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[20:21], v[18:19], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[22:23], v[18:19], v[12:15]
ds_read_b64 v[16:17], $53
ds_read_b64 v[18:19], $54
ds_read_m32x16_b16 v[20:23], $55
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[28:29], v[24:25], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[30:31], v[24:25], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[28:29], v[26:27], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[30:31], v[26:27], v[12:15]
ds_read_b64 v[24:25], $56
ds_read_b64 v[26:27], $57
ds_read_m32x16_b16 v[28:31], $58
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[20:21], v[16:17], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[22:23], v[16:17], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[20:21], v[18:19], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[22:23], v[18:19], v[12:15]
ds_read_b64 v[16:17], $59
ds_read_b64 v[18:19], $60
ds_read_m32x16_b16 v[20:23], $61
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[28:29], v[24:25], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[30:31], v[24:25], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[28:29], v[26:27], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[30:31], v[26:27], v[12:15]
ds_read_b64 v[24:25], $62
ds_read_b64 v[26:27], $63
ds_read_m32x16_b16 v[28:31], $64
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[0:3], v[20:21], v[16:17], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[22:23], v[16:17], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[20:21], v[18:19], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[22:23], v[18:19], v[12:15]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[0:3], v[28:29], v[24:25], v[0:3]
v_mmac_f32_16x16x16_f16 v[4:7], v[30:31], v[24:25], v[4:7]
v_mmac_f32_16x16x16_f16 v[8:11], v[28:29], v[26:27], v[8:11]
v_mmac_f32_16x16x16_f16 v[12:15], v[30:31], v[26:27], v[12:15]
s_barrier
s_sub_u32 s46,s46,1
s_cmp_eq_u32 s46,0
s_cbranch_scc1 Lregular_done
s_waitcnt vmcnt(0)
ds_write_b128 $26, v[32:35]
ds_write_b128 $28, v[36:39]
ds_write_b128 $30, v[40:43]
ds_write_b128 $32, v[44:47]
ds_write_b128 $34, v[48:51]
ds_write_b128 $36, v[52:55]
ds_write_b128 $38, v[56:59]
ds_write_b128 $40, v[60:63]
s_waitcnt lgkmcnt(0)
s_barrier
s_branch Lregular_loop
Lregular_done:
""")
_CON_0_64_64_128_2_2 = tl.constexpr(
    "=&{v0},=&{v1},=&{v2},=&{v3},=&{v4},=&{v5},=&{v6},=&{v7},=&{v8},=&{v9},=&{v10},=&{v11},=&{v12},"
    "=&{v13},=&{v14},=&{v15},v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,"
    "v,v,v,v,v,v,v,v,v,v,v,~{s32},~{s33},~{s34},~{s35},~{s36},~{s37},~{s38},~{s39},~{s44},~{s45},~{s46},"
    "~{s47},~{s48},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},"
    "~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},"
    "~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},"
    "~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},~{vcc},~{m0},~{scc},~{memory}"
)

_ASM_1_32_64_128_2_2 = tl.constexpr("""v_readfirstlane_b32 s32, $8
v_readfirstlane_b32 s33, $9
v_readfirstlane_b32 s34, $10
v_readfirstlane_b32 s36, $11
v_readfirstlane_b32 s37, $12
v_readfirstlane_b32 s38, $13
v_readfirstlane_b32 s47, $14
v_readfirstlane_b32 s48, $15
v_readfirstlane_b32 s46, $16
s_mov_b32 s35, 0x20000
s_mov_b32 s39, 0x20000
s_mov_b32 s44,0
s_mov_b32 s45,0
v_mov_b32 v0,0
v_mov_b32 v1,0
v_mov_b32 v2,0
v_mov_b32 v3,0
v_mov_b32 v4,0
v_mov_b32 v5,0
v_mov_b32 v6,0
v_mov_b32 v7,0
buffer_load_dwordx4 v[20:23], $17, s[32:35], s44 offen
buffer_load_dwordx4 v[24:27], $19, s[32:35], s44 offen
buffer_load_dwordx4 v[28:31], $21, s[36:39], s45 offen
buffer_load_dwordx4 v[32:35], $23, s[36:39], s45 offen
buffer_load_dwordx4 v[36:39], $25, s[36:39], s45 offen
buffer_load_dwordx4 v[40:43], $27, s[36:39], s45 offen
s_waitcnt vmcnt(0)
ds_write_b128 $18, v[20:23]
ds_write_b128 $20, v[24:27]
ds_write_b128 $22, v[28:31]
ds_write_b128 $24, v[32:35]
ds_write_b128 $26, v[36:39]
ds_write_b128 $28, v[40:43]
s_waitcnt lgkmcnt(0)
s_barrier
Lregular_loop:
s_add_u32 s44,s44,s47
s_add_u32 s45,s45,s48
buffer_load_dwordx4 v[20:23], $17, s[32:35], s44 offen
buffer_load_dwordx4 v[24:27], $19, s[32:35], s44 offen
buffer_load_dwordx4 v[28:31], $21, s[36:39], s45 offen
buffer_load_dwordx4 v[32:35], $23, s[36:39], s45 offen
buffer_load_dwordx4 v[36:39], $25, s[36:39], s45 offen
buffer_load_dwordx4 v[40:43], $27, s[36:39], s45 offen
ds_read_b64 v[8:9], $29
ds_read_m32x16_b16 v[10:13], $30
ds_read_b64 v[14:15], $31
ds_read_m32x16_b16 v[16:19], $32
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[10:11], v[8:9], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[12:13], v[8:9], v[4:7]
ds_read_b64 v[8:9], $33
ds_read_m32x16_b16 v[10:13], $34
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[16:17], v[14:15], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[18:19], v[14:15], v[4:7]
ds_read_b64 v[14:15], $35
ds_read_m32x16_b16 v[16:19], $36
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[10:11], v[8:9], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[12:13], v[8:9], v[4:7]
ds_read_b64 v[8:9], $37
ds_read_m32x16_b16 v[10:13], $38
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[16:17], v[14:15], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[18:19], v[14:15], v[4:7]
ds_read_b64 v[14:15], $39
ds_read_m32x16_b16 v[16:19], $40
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[10:11], v[8:9], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[12:13], v[8:9], v[4:7]
ds_read_b64 v[8:9], $41
ds_read_m32x16_b16 v[10:13], $42
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[16:17], v[14:15], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[18:19], v[14:15], v[4:7]
ds_read_b64 v[14:15], $43
ds_read_m32x16_b16 v[16:19], $44
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_bf16 v[0:3], v[10:11], v[8:9], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[12:13], v[8:9], v[4:7]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_bf16 v[0:3], v[16:17], v[14:15], v[0:3]
v_mmac_f32_16x16x16_bf16 v[4:7], v[18:19], v[14:15], v[4:7]
s_barrier
s_sub_u32 s46,s46,1
s_cmp_eq_u32 s46,0
s_cbranch_scc1 Lregular_done
s_waitcnt vmcnt(0)
ds_write_b128 $18, v[20:23]
ds_write_b128 $20, v[24:27]
ds_write_b128 $22, v[28:31]
ds_write_b128 $24, v[32:35]
ds_write_b128 $26, v[36:39]
ds_write_b128 $28, v[40:43]
s_waitcnt lgkmcnt(0)
s_barrier
s_branch Lregular_loop
Lregular_done:
""")
_CON_1_32_64_128_2_2 = tl.constexpr(
    "=&{v0},=&{v1},=&{v2},=&{v3},=&{v4},=&{v5},=&{v6},=&{v7},v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,"
    "v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,~{s32},~{s33},~{s34},~{s35},~{s36},~{s37},~{s38},~{s39},~{s44},~{s45},"
    "~{s46},~{s47},~{s48},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},"
    "~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},"
    "~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{vcc},~{m0},~{scc},"
    "~{memory}"
)


@triton.jit
def _native_dot(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
    STAGES: tl.constexpr,
):
    # Complete native body: dtype/layout scheduling is left to the compiler.
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    gm = tl.cdiv(M, TM)
    gn = tl.cdiv(N, TN)
    first = pid // (GROUP * gn) * GROUP
    gs = tl.minimum(GROUP, gm - first)
    m0 = (first + pid % (GROUP * gn) % gs) * TM
    n0 = pid % (GROUP * gn) // gs * TN
    rm = m0 + tl.arange(0, TM)
    rn = n0 + tl.arange(0, TN)
    rk = tl.arange(0, TK)
    acc = tl.zeros((TM, TN), tl.float32)
    for off in tl.range(0, K, TK, num_stages=STAGES, loop_unroll_factor=1):
        kk = off + rk
        av = tl.load(
            A + batch * M * K + rm[:, None].to(tl.int64) * K + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            B + batch * K * N + kk[:, None].to(tl.int64) * N + rn[None, :],
            (kk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        C + batch * M * N + rm[:, None].to(tl.int64) * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def regular_2_128_128_16_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 128, 128, 16, 2)


@libentry()
@triton.jit
def regular_2_64_64_128_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 128, 1)


@libentry()
@triton.jit
def regular_0_64_64_128_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    BM: tl.constexpr = 64
    BN: tl.constexpr = 64
    BK: tl.constexpr = 128
    V: tl.constexpr = 8
    ES: tl.constexpr = 2
    t = tl.arange(0, 256)
    lane = t % 64
    wave = t // 64
    wm = wave % 2
    wn = wave // 2
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    pid = tl.program_id(0)
    first = pid // (GROUP * gn) * GROUP
    gs = tl.minimum(GROUP, gm - first)
    m0 = (first + pid % (GROUP * gn) % gs) * BM
    n0 = pid % (GROUP * gn) // gs * BN
    batch = tl.program_id(1)
    shared = tle.gpu.alloc((8192,), tl.uint32, nv_mma_shared_layout=False)
    base = tle.gpu.local_ptr(shared, (0,)).to(tl.uint64).to(tl.uint32)
    aptr = (A + batch * M * K).to(tl.uint64)
    bptr = (B + batch * K * N).to(tl.uint64)
    a0 = aptr.to(tl.uint32)
    a1 = (aptr >> 32).to(tl.uint32)
    alim = tl.full((), M * K * ES, tl.uint32)
    b0 = bptr.to(tl.uint32)
    b1 = (bptr >> 32).to(tl.uint32)
    blim = tl.full((), K * N * ES, tl.uint32)
    inca = tl.full((), BK * ES, tl.uint32)
    incb = tl.full((), BK * N * ES, tl.uint32)
    tiles = tl.full((), K // BK, tl.uint32)
    x = (t + 0) * V
    ag0 = tl.where(
        m0 + x // BK < M, ((m0 + x // BK) * K + x % BK) * ES, -2147483648
    ).to(tl.uint32)
    as0 = base + ax(x // BK, x % BK, BK, V) * ES
    x = (t + 256) * V
    ag1 = tl.where(
        m0 + x // BK < M, ((m0 + x // BK) * K + x % BK) * ES, -2147483648
    ).to(tl.uint32)
    as1 = base + ax(x // BK, x % BK, BK, V) * ES
    x = (t + 512) * V
    ag2 = tl.where(
        m0 + x // BK < M, ((m0 + x // BK) * K + x % BK) * ES, -2147483648
    ).to(tl.uint32)
    as2 = base + ax(x // BK, x % BK, BK, V) * ES
    x = (t + 768) * V
    ag3 = tl.where(
        m0 + x // BK < M, ((m0 + x // BK) * K + x % BK) * ES, -2147483648
    ).to(tl.uint32)
    as3 = base + ax(x // BK, x % BK, BK, V) * ES
    x = (t + 0) * V
    bg0 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs0 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    x = (t + 256) * V
    bg1 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs1 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    x = (t + 512) * V
    bg2 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs2 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    x = (t + 768) * V
    bg3 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs3 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    ap0_0 = base + ax(wm * 32 + 0 + lane % 16, 0 + lane // 16 * 4, BK, V) * ES
    ap1_0 = base + ax(wm * 32 + 16 + lane % 16, 0 + lane // 16 * 4, BK, V) * ES
    bp0_0 = (
        base + BM * BK * ES + bx(0 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_1 = base + ax(wm * 32 + 0 + lane % 16, 16 + lane // 16 * 4, BK, V) * ES
    ap1_1 = base + ax(wm * 32 + 16 + lane % 16, 16 + lane // 16 * 4, BK, V) * ES
    bp0_1 = (
        base + BM * BK * ES + bx(16 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_2 = base + ax(wm * 32 + 0 + lane % 16, 32 + lane // 16 * 4, BK, V) * ES
    ap1_2 = base + ax(wm * 32 + 16 + lane % 16, 32 + lane // 16 * 4, BK, V) * ES
    bp0_2 = (
        base + BM * BK * ES + bx(32 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_3 = base + ax(wm * 32 + 0 + lane % 16, 48 + lane // 16 * 4, BK, V) * ES
    ap1_3 = base + ax(wm * 32 + 16 + lane % 16, 48 + lane // 16 * 4, BK, V) * ES
    bp0_3 = (
        base + BM * BK * ES + bx(48 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_4 = base + ax(wm * 32 + 0 + lane % 16, 64 + lane // 16 * 4, BK, V) * ES
    ap1_4 = base + ax(wm * 32 + 16 + lane % 16, 64 + lane // 16 * 4, BK, V) * ES
    bp0_4 = (
        base + BM * BK * ES + bx(64 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_5 = base + ax(wm * 32 + 0 + lane % 16, 80 + lane // 16 * 4, BK, V) * ES
    ap1_5 = base + ax(wm * 32 + 16 + lane % 16, 80 + lane // 16 * 4, BK, V) * ES
    bp0_5 = (
        base + BM * BK * ES + bx(80 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_6 = base + ax(wm * 32 + 0 + lane % 16, 96 + lane // 16 * 4, BK, V) * ES
    ap1_6 = base + ax(wm * 32 + 16 + lane % 16, 96 + lane // 16 * 4, BK, V) * ES
    bp0_6 = (
        base + BM * BK * ES + bx(96 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_7 = base + ax(wm * 32 + 0 + lane % 16, 112 + lane // 16 * 4, BK, V) * ES
    ap1_7 = base + ax(wm * 32 + 16 + lane % 16, 112 + lane // 16 * 4, BK, V) * ES
    bp0_7 = (
        base
        + BM * BK * ES
        + bx(112 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    acc = tl.inline_asm_elementwise(
        _ASM_0_64_64_128_2_2,
        _CON_0_64_64_128_2_2,
        [
            a0,
            a1,
            alim,
            b0,
            b1,
            blim,
            inca,
            incb,
            tiles,
            ag0,
            as0,
            ag1,
            as1,
            ag2,
            as2,
            ag3,
            as3,
            bg0,
            bs0,
            bg1,
            bs1,
            bg2,
            bs2,
            bg3,
            bs3,
            ap0_0,
            ap1_0,
            bp0_0,
            ap0_1,
            ap1_1,
            bp0_1,
            ap0_2,
            ap1_2,
            bp0_2,
            ap0_3,
            ap1_3,
            bp0_3,
            ap0_4,
            ap1_4,
            bp0_4,
            ap0_5,
            ap1_5,
            bp0_5,
            ap0_6,
            ap1_6,
            bp0_6,
            ap0_7,
            ap1_7,
            bp0_7,
        ],
        dtype=(
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
        ),
        is_pure=False,
        pack=1,
    )
    row = m0 + wm * (BM // 2) + lane // 16
    col = n0 + wn * (BN // 2) + lane % 16
    for i in tl.static_range(0, BM * BN // 256):
        rr = row + i // 4 // (BN // 32) * 16 + i % 4 * 4
        cc = col + i // 4 % (BN // 32) * 16
        offset = batch.to(tl.int64) * M * N + rr.to(tl.int64) * N + cc
        tl.store(C + offset, acc[i], (rr < M) & (cc < N))


@libentry()
@triton.jit
def regular_1_32_64_128_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    BM: tl.constexpr = 32
    BN: tl.constexpr = 64
    BK: tl.constexpr = 128
    V: tl.constexpr = 8
    ES: tl.constexpr = 2
    t = tl.arange(0, 256)
    lane = t % 64
    wave = t // 64
    wm = wave % 2
    wn = wave // 2
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    pid = tl.program_id(0)
    first = pid // (GROUP * gn) * GROUP
    gs = tl.minimum(GROUP, gm - first)
    m0 = (first + pid % (GROUP * gn) % gs) * BM
    n0 = pid % (GROUP * gn) // gs * BN
    batch = tl.program_id(1)
    shared = tle.gpu.alloc((8192,), tl.uint32, nv_mma_shared_layout=False)
    base = tle.gpu.local_ptr(shared, (0,)).to(tl.uint64).to(tl.uint32)
    aptr = (A + batch * M * K).to(tl.uint64)
    bptr = (B + batch * K * N).to(tl.uint64)
    a0 = aptr.to(tl.uint32)
    a1 = (aptr >> 32).to(tl.uint32)
    alim = tl.full((), M * K * ES, tl.uint32)
    b0 = bptr.to(tl.uint32)
    b1 = (bptr >> 32).to(tl.uint32)
    blim = tl.full((), K * N * ES, tl.uint32)
    inca = tl.full((), BK * ES, tl.uint32)
    incb = tl.full((), BK * N * ES, tl.uint32)
    tiles = tl.full((), K // BK, tl.uint32)
    x = (t + 0) * V
    ag0 = tl.where(
        m0 + x // BK < M, ((m0 + x // BK) * K + x % BK) * ES, -2147483648
    ).to(tl.uint32)
    as0 = base + ax(x // BK, x % BK, BK, V) * ES
    x = (t + 256) * V
    ag1 = tl.where(
        m0 + x // BK < M, ((m0 + x // BK) * K + x % BK) * ES, -2147483648
    ).to(tl.uint32)
    as1 = base + ax(x // BK, x % BK, BK, V) * ES
    x = (t + 0) * V
    bg0 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs0 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    x = (t + 256) * V
    bg1 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs1 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    x = (t + 512) * V
    bg2 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs2 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    x = (t + 768) * V
    bg3 = tl.where(n0 + x % BN < N, (x // BN * N + n0 + x % BN) * ES, -2147483648).to(
        tl.uint32
    )
    bs3 = base + BM * BK * ES + bx(x // BN, x % BN, BN, V) * ES
    ap0_0 = base + ax(wm * 16 + 0 + lane % 16, 0 + lane // 16 * 4, BK, V) * ES
    bp0_0 = (
        base + BM * BK * ES + bx(0 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_1 = base + ax(wm * 16 + 0 + lane % 16, 16 + lane // 16 * 4, BK, V) * ES
    bp0_1 = (
        base + BM * BK * ES + bx(16 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_2 = base + ax(wm * 16 + 0 + lane % 16, 32 + lane // 16 * 4, BK, V) * ES
    bp0_2 = (
        base + BM * BK * ES + bx(32 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_3 = base + ax(wm * 16 + 0 + lane % 16, 48 + lane // 16 * 4, BK, V) * ES
    bp0_3 = (
        base + BM * BK * ES + bx(48 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_4 = base + ax(wm * 16 + 0 + lane % 16, 64 + lane // 16 * 4, BK, V) * ES
    bp0_4 = (
        base + BM * BK * ES + bx(64 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_5 = base + ax(wm * 16 + 0 + lane % 16, 80 + lane // 16 * 4, BK, V) * ES
    bp0_5 = (
        base + BM * BK * ES + bx(80 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_6 = base + ax(wm * 16 + 0 + lane % 16, 96 + lane // 16 * 4, BK, V) * ES
    bp0_6 = (
        base + BM * BK * ES + bx(96 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    ap0_7 = base + ax(wm * 16 + 0 + lane % 16, 112 + lane // 16 * 4, BK, V) * ES
    bp0_7 = (
        base
        + BM * BK * ES
        + bx(112 + lane // 4, wn * 32 + 0 + lane % 4 * V, BN, V) * ES
    )
    acc = tl.inline_asm_elementwise(
        _ASM_1_32_64_128_2_2,
        _CON_1_32_64_128_2_2,
        [
            a0,
            a1,
            alim,
            b0,
            b1,
            blim,
            inca,
            incb,
            tiles,
            ag0,
            as0,
            ag1,
            as1,
            bg0,
            bs0,
            bg1,
            bs1,
            bg2,
            bs2,
            bg3,
            bs3,
            ap0_0,
            bp0_0,
            ap0_1,
            bp0_1,
            ap0_2,
            bp0_2,
            ap0_3,
            bp0_3,
            ap0_4,
            bp0_4,
            ap0_5,
            bp0_5,
            ap0_6,
            bp0_6,
            ap0_7,
            bp0_7,
        ],
        dtype=(
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
        ),
        is_pure=False,
        pack=1,
    )
    row = m0 + wm * (BM // 2) + lane // 16
    col = n0 + wn * (BN // 2) + lane % 16
    for i in tl.static_range(0, BM * BN // 256):
        rr = row + i // 4 // (BN // 32) * 16 + i % 4 * 4
        cc = col + i // 4 % (BN // 32) * 16
        offset = batch.to(tl.int64) * M * N + rr.to(tl.int64) * N + cc
        tl.store(C + offset, acc[i], (rr < M) & (cc < N))


@libentry()
@triton.jit
def regular_0_64_64_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 32, 2)


@libentry()
@triton.jit
def regular_1_64_64_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 32, 2)


@libentry()
@triton.jit
def regular_0_128_128_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 128, 128, 32, 2)


@libentry()
@triton.jit
def regular_1_128_128_32_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 128, 128, 32, 2)


@libentry()
@triton.jit
def regular_2_64_64_16_2_2(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr
):
    _native_dot(A, B, C, M, N, K, GROUP, 64, 64, 16, 2)


REGULAR_KERNELS = {
    (2, 64, 64, 16, 2, 2): regular_2_64_64_16_2_2,
    (0, 128, 128, 32, 2, 2): regular_0_128_128_32_2_2,
    (0, 64, 64, 32, 2, 2): regular_0_64_64_32_2_2,
    (1, 128, 128, 32, 2, 2): regular_1_128_128_32_2_2,
    (1, 64, 64, 32, 2, 2): regular_1_64_64_32_2_2,
    (2, 128, 128, 16, 2, 2): regular_2_128_128_16_2_2,
    (2, 64, 64, 128, 2, 2): regular_2_64_64_128_2_2,
    (0, 64, 64, 128, 2, 2): regular_0_64_64_128_2_2,
    (1, 32, 64, 128, 2, 2): regular_1_32_64_128_2_2,
}


NATIVE_REGULAR_KEYS = frozenset(
    [
        (0, 64, 64, 32, 2, 2),
        (0, 128, 128, 32, 2, 2),
        (1, 64, 64, 32, 2, 2),
        (1, 128, 128, 32, 2, 2),
        (2, 64, 64, 16, 2, 2),
        (2, 64, 64, 128, 2, 2),
        (2, 128, 128, 16, 2, 2),
    ]
)


def launch_regular(a, b, c, config):
    dt, bm, bn, bk, wm, wn = config
    batch, m, k = a.shape
    n = b.shape[2]
    return REGULAR_KERNELS[config][(triton.cdiv(m, bm) * triton.cdiv(n, bn), batch)](
        a,
        b,
        c,
        m,
        n,
        k,
        1 if max(m, n) <= 512 and k <= 512 else 8,
        num_warps=wm * wn,
        num_stages=1,
        allow_flush_denorm=True,
        enable_fp_fusion=True,
    )
