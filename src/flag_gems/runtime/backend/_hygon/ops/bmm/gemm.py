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

"""Generic and packed GEMM tuning families."""

import torch
import triton
import triton.language as tl

from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle


@triton.jit
def baddbmm_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGE_SHARED: tl.constexpr = False,
    UNROLL: tl.constexpr = 1,
    BATCH: tl.constexpr = 1,
):
    pid, batch = tl.program_id(0), tl.program_id(1)
    gm, gn = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % (GROUP_M * gn) % size
    pn = pid % (GROUP_M * gn) // size
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    cols = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    ap = A + batch * AB + rows[:, None] * AM + rk[None, :] * AK
    bp = B + batch * BB + rk[:, None] * BK + cols[None, :] * BN
    if STAGE_SHARED:
        shared_b = tle.gpu.alloc(
            (BLOCK_K, BLOCK_N), B.dtype.element_ty, scope=tle.gpu.smem
        )
        local_b = tle.gpu.local_ptr(shared_b)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ki in tl.range(tl.cdiv(K, BLOCK_K), loop_unroll_factor=UNROLL):
        am = ((M % BLOCK_M == 0) | (rm < M))[:, None]
        bn = ((N % BLOCK_N == 0) | (rn < N))[None, :]
        km = (K % BLOCK_K == 0) | (rk + ki * BLOCK_K < K)
        a = tl.load(ap, am & km[None, :], 0)
        b = tl.load(bp, bn & km[:, None], 0)
        if STAGE_SHARED:
            tl.store(local_b, b)
            b = tl.load(local_b)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
        ap += BLOCK_K * AK
        bp += BLOCK_K * BK
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc *= ALPHA
    if BETA != 0:
        acc += BETA * tl.load(
            Bias + batch * CB + rm[:, None] * CM + rn[None, :] * CN, mask, 0
        ).to(tl.float32)
    tl.store(Out + batch * OB + rm[:, None] * OM + rn[None, :] * ON, acc, mask)


def launch_compact_gemm(a, b, out, bias=None):
    # The two-warp configuration is intended for compact FP16/BF16
    # matrices with enough output tiles to occupy the device.
    if a.dtype not in (torch.float16, torch.bfloat16):
        return False
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if a.ndim == 3 else 1
    if (
        not (256 <= m <= 1024 and 128 <= n <= 1024 and 128 <= k <= 4096)
        or m % 32
        or n % 64
        or k % 128
    ):
        return False
    tiles = batch * (m // 32) * (n // 64)
    if not 64 <= tiles <= 512 or not a.is_contiguous() or not out.is_contiguous():
        return False
    a_strides = a.stride()
    b_strides = b.stride()
    out_strides = out.stride()
    column_major = b_strides[-2:] == (1, k)
    if not (
        b.is_contiguous() or (column_major and (b.ndim == 2 or b_strides[0] == k * n))
    ):
        return False
    if bias is None and (not column_major or tiles >= 256):
        return False
    if bias is not None and (bias.shape != out.shape or not bias.is_contiguous()):
        return False
    c_strides = out_strides if bias is None else bias.stride()
    if a.ndim == 2:
        a_strides = (0, *a_strides)
        b_strides = (0, *b_strides)
        out_strides = (0, *out_strides)
        c_strides = (0, *c_strides)
    bm = 64 if column_major and tiles >= 256 else 32
    bk = 64 if not column_major and tiles >= 256 else 128
    grid = (triton.cdiv(m, bm) * (n // 64), batch)
    baddbmm_kernel[grid](
        a,
        b,
        out if bias is None else bias,
        out,
        m,
        n,
        k,
        *a_strides,
        *b_strides,
        *c_strides,
        *out_strides,
        1,
        int(bias is not None),
        bm,
        64,
        bk,
        1,
        BATCH=batch,
        num_warps=4 if bm == 64 else 2,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return True
