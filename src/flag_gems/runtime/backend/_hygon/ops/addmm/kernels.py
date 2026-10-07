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
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    benchmark_mode="event",
    flagtune_op_name="addmm",
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    EVEN_M: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
    EVEN_K: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if EVEN_M:
        offs_am_c = tl.max_contiguous(
            tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M
        )
    else:
        offs_am_c = tl.max_contiguous(
            tl.multiple_of(offs_am % M, BLOCK_SIZE_M), BLOCK_SIZE_M
        )
    if EVEN_N:
        offs_bn_c = tl.max_contiguous(
            tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N
        )
    else:
        offs_bn_c = tl.max_contiguous(
            tl.multiple_of(offs_bn % N, BLOCK_SIZE_N), BLOCK_SIZE_N
        )

    # A is always row-major (mat1.contiguous() is guaranteed by _launch_addmm)
    a_ptrs = a_ptr + offs_am_c[:, None] * K + offs_k[None, :]
    if B_CONTIGUOUS:
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_bn_c[None, :]
        b_step = BLOCK_SIZE_K * N
    else:
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn_c[None, :] * stride_bn
        b_step = BLOCK_SIZE_K * stride_bk

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if EVEN_K:
        loop_end = K
    else:
        loop_end = tl.cdiv(K, BLOCK_SIZE_K) * BLOCK_SIZE_K - BLOCK_SIZE_K
    for k in range(0, loop_end, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if IS_FP64:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_step

    if not EVEN_K:
        rk = loop_end + offs_k
        mask_k = rk < K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        if IS_FP64:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    if not ALPHA_ONE:
        accumulator *= alpha

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if BETA_ZERO:
        c = accumulator
    else:
        if BIAS_SCALAR:
            bias = tl.load(i_ptr)
        elif BIAS_ROW:
            bias_ptrs = i_ptr + stride_in * offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)[None, :]
        elif BIAS_COL:
            bias_ptrs = i_ptr + stride_im * offs_cm
            bias = tl.load(bias_ptrs, mask=offs_cm < M, other=0.0)[:, None]
        else:
            i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
            if EVEN_M and EVEN_N:
                bias = tl.load(i_ptrs)
            else:
                bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
        if BETA_ONE:
            c = accumulator + bias
        else:
            c = accumulator + bias * beta
        c = c.to(bias.dtype)

    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _store_addmm_subtile(
    i_ptr,
    c_ptr,
    accumulator,
    rows,
    cols,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    ALPHA_ONE: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
):
    if not ALPHA_ONE:
        accumulator *= alpha
    if not BETA_ZERO:
        bias = tl.load(i_ptr + rows[:, None] * stride_im + cols[None, :] * stride_in)
        if BETA_ONE:
            accumulator += bias
        else:
            accumulator += beta * bias
        accumulator = accumulator.to(bias.dtype)
    tl.store(c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn, accumulator)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_rowmajor"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    benchmark_mode="event",
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_rowmajor_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    # The host materializes column-major B for aligned row-major half GEMMs.
    # Contiguous K pairs avoid gfx936 shared-to-dot scalar halfword reads.
    # Partition the accumulator, not K: each output is reduced by one CTA.
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)
    rk = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] + cols[None, :] * K
    acc00 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    acc01 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    acc10 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    acc11 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)

    for _ in tl.range(0, K // BLOCK_SIZE_K, loop_unroll_factor=4):
        a0 = tl.load(a_ptrs)
        a1 = tl.load(a_ptrs + (BLOCK_SIZE_M // 2) * K)
        b0 = tl.load(b_ptrs)
        b1 = tl.load(b_ptrs + (BLOCK_SIZE_N // 2) * K)
        acc00 = tl.dot(a0, b0, acc00, input_precision="ieee")
        acc01 = tl.dot(a0, b1, acc01, input_precision="ieee")
        acc10 = tl.dot(a1, b0, acc10, input_precision="ieee")
        acc11 = tl.dot(a1, b1, acc11, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc00,
        rows,
        cols,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc01,
        rows,
        cols + BLOCK_SIZE_N // 2,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc10,
        rows + BLOCK_SIZE_M // 2,
        cols,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc11,
        rows + BLOCK_SIZE_M // 2,
        cols + BLOCK_SIZE_N // 2,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )


@triton.jit
def _transpose_addmm_b(b_ptr, bt_ptr, K: tl.constexpr, N: tl.constexpr):
    rk = tl.program_id(0) * 64 + tl.arange(0, 64)
    rn = tl.program_id(1) * 64 + tl.arange(0, 64)
    mask = (rk[:, None] < K) & (rn[None, :] < N)
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask, other=0)
    tl.store(bt_ptr + rn[None, :] * K + rk[:, None], b, mask)


@triton.jit
def _addmm_compact_body(
    A,
    B,
    Bias,
    C,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    PREFETCH: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    BETA_ONE: tl.constexpr,
    COLUMN_B: tl.constexpr = False,
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    start = pid // (GROUP * nn) * GROUP
    if nm % GROUP == 0:
        gs = GROUP
    else:
        gs = tl.minimum(nm - start, GROUP)
    pm = start + pid % (GROUP * nn) % gs
    pn = pid % (GROUP * nn) // gs
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + m[:, None] * K + k[None, :]
    if COLUMN_B:
        bp = B + k[:, None] + n[None, :] * K
        b_step = BK
    else:
        bp = B + k[:, None] * N + n[None, :]
        b_step = BK * N
    acc = tl.zeros((BM, BN), tl.float32)
    if PREFETCH:
        a = tl.load(
            ap,
            ((m[:, None] < M) | (M % BM == 0)) & ((k[None, :] < K) | (K % BK == 0)),
            other=0,
        )
        b = tl.load(
            bp,
            ((k[:, None] < K) | (K % BK == 0)) & ((n[None, :] < N) | (N % BN == 0)),
            other=0,
        )
        for i in range(1, tl.cdiv(K, BK)):
            ap += BK
            bp += b_step
            na = tl.load(
                ap,
                ((m[:, None] < M) | (M % BM == 0))
                & (((i * BK + k)[None, :] < K) | (K % BK == 0)),
                other=0,
            )
            nb = tl.load(
                bp,
                (((i * BK + k)[:, None] < K) | (K % BK == 0))
                & ((n[None, :] < N) | (N % BN == 0)),
                other=0,
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")
            a, b = (na, nb)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    else:
        for i in tl.range(0, tl.cdiv(K, BK), loop_unroll_factor=1):
            a = tl.load(
                ap,
                ((m[:, None] < M) | (M % BM == 0))
                & (((i * BK + k)[None, :] < K) | (K % BK == 0)),
                other=0,
            )
            b = tl.load(
                bp,
                (((i * BK + k)[:, None] < K) | (K % BK == 0))
                & ((n[None, :] < N) | (N % BN == 0)),
                other=0,
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")
            ap += BK
            bp += b_step
    off = m[:, None] * stride_im + n[None, :] * stride_in
    mask = (m[:, None] < M) & (n[None, :] < N)
    if not ALPHA_ONE:
        acc *= alpha
    if not BETA_ZERO:
        bias = tl.load(Bias + off, mask, other=0)
        if BETA_ONE:
            acc += bias
        else:
            acc += beta * bias
        acc = acc.to(bias.dtype)
    tl.store(C + m[:, None] * stride_cm + n[None, :] * stride_cn, acc, mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_small"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    benchmark_mode="event",
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_small_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 1,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    _addmm_compact_body(
        a_ptr,
        b_ptr,
        i_ptr,
        c_ptr,
        alpha,
        beta,
        M,
        N,
        K,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        False,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_medium"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    benchmark_mode="event",
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_medium_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 1,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    _addmm_compact_body(
        a_ptr,
        b_ptr,
        i_ptr,
        c_ptr,
        alpha,
        beta,
        M,
        N,
        K,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        True,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_fp32"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    benchmark_mode="event",
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_fp32_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 1,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = pid // num_pid_in_group * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)
    rk = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + cols[None, :]
    acc00 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    acc01 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    acc10 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    acc11 = tl.zeros((BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2), tl.float32)
    for _ in tl.range(0, K // BLOCK_SIZE_K, loop_unroll_factor=1):
        a0 = tl.load(a_ptrs)
        a1 = tl.load(a_ptrs + BLOCK_SIZE_M // 2 * K)
        b0 = tl.load(b_ptrs)
        b1 = tl.load(b_ptrs + BLOCK_SIZE_N // 2)
        acc00 = tl.dot(a0, b0, acc00, input_precision="ieee")
        acc01 = tl.dot(a0, b1, acc01, input_precision="ieee")
        acc10 = tl.dot(a1, b0, acc10, input_precision="ieee")
        acc11 = tl.dot(a1, b1, acc11, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc00,
        rows,
        cols,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc01,
        rows,
        cols + BLOCK_SIZE_N // 2,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc10,
        rows + BLOCK_SIZE_M // 2,
        cols,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )
    _store_addmm_subtile(
        i_ptr,
        c_ptr,
        acc11,
        rows + BLOCK_SIZE_M // 2,
        cols + BLOCK_SIZE_N // 2,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_column"),
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=2,
    rep=8,
    benchmark_mode="event",
    flagtune_op_name="addmm",
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_column_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 1,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    _addmm_compact_body(
        a_ptr,
        b_ptr,
        i_ptr,
        c_ptr,
        alpha,
        beta,
        M,
        N,
        K,
        stride_im,
        stride_in,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        False,
        ALPHA_ONE,
        BETA_ZERO,
        BETA_ONE,
        True,
    )
