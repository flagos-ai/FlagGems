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
import triton.language.extra.smt as smt
import triton.language.extra.smt_rvisa as tle
from triton.language.extra.smt_rvisa import call as _sr_call
from triton.language.extra.smt_rvisa import index, mem

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _mm_fallback_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Scalar products also support dtypes/layouts rejected by the matrix engine.
    # Bound the launch grid on runtimes which support at most 512 programs.
    tiles_n = tl.cdiv(N, BLOCK_N)
    for task in range(tl.program_id(0), M * tiles_n, tl.num_programs(0)):
        row = task // tiles_n
        cols = task % tiles_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for k_start in range(tl.cdiv(K, BLOCK_K)):
            ks = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(A + row * stride_am + ks * stride_ak, ks < K, other=0).to(
                tl.float32
            )
            b = tl.load(
                B + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                (ks[:, None] < K) & (cols[None, :] < N),
                other=0,
            ).to(tl.float32)
            acc += tl.sum(a[:, None] * b, axis=0)
        tl.store(C + row * stride_cm + cols * stride_cn, acc, cols < N)


def _mm_fallback(a, b, out):
    M, K = a.shape
    N = b.shape[1]
    if M == 0 or N == 0:
        return out
    _mm_fallback_kernel[(min(M * triton.cdiv(N, 16), 512),)](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=16,
        BLOCK_K=32,
    )
    return out


_f16 = tle.f16
_f32 = tle.f32
# NOTE: dtype args to tle.* must be bare names that match _DTYPE_NAMES
# ({"f16","f32",...}). Use the un-prefixed aliases f16/f32 inside the raw
# kernel body so the AST produces ast.Name("f32"), not ast.Name("_f32").
f16 = _f16
f32 = _f32


# ---------------------------------------------------------------------------
# M==1 GEMV fast path (true vfwmacc.vf intrinsic, pure llvm-direct sibling).
#   C[0, n:n+VL] = sum_k x[k] * W[k, n:n+VL]
#   - x[k]    : scalar f16 load (llvm.load)
#   - W[k, :] : contiguous row vector load (llvm.load, vector<[4]xf16> = 64 f16)
#   - acc     : vector<[4]xf32>, vfwmacc.vf (scalar·vector widening FMA) per k
#   - fptrunc : f32→f16 vector narrowing (llvm.fptrunc), direct f16 output
# Pure llvm-direct → routes to LLVMDirectTextCodegen sibling path, bypasses
# spine-opt bufferization (which rejects llvm.call_intrinsic as "unknown memory
# side effects"). program_id resolved at runtime via spine_grid(ctx, axis).
# Direct f16 output — no f32 workspace, no host-side conversion. Each program
# handles one NB=64 N-block; grid = (N // NB,).
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_m1_raw(
    x_ptr: mem(f16),
    W_ptr: mem(f16),
    C_ptr: mem(f16, out=True),
    K: index,
    N: index,
    NB: index,
):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const(0.0, "vector<[4]xf32>")
    xbase = tle.llvm_base_ptr(x_ptr)
    wbase = tle.llvm_base_ptr(W_ptr)
    cbase = tle.llvm_base_ptr(C_ptr)
    n_start = tle.program_id(0) * NB
    acc = zero_acc
    for k in tle.range(0, K, 1):
        gx = tle.llvm_gep(xbase, k, "f16")
        xk = tle.call_intrinsic("llvm.load", [gx], result_type="f16")
        gw = tle.llvm_gep(wbase, k * N + n_start, "f16")
        wv = tle.call_intrinsic("llvm.load", [gw], result_type="vector<[4]xf16>")
        acc = tle.call_intrinsic(
            "llvm.riscv.vfwmacc.vf",
            [acc, xk, wv, zero_i64, vl, zero_i64],
            result_type="vector<[4]xf32>",
        )
    acc_h = tle.call_intrinsic("llvm.fptrunc", [acc], result_type="vector<[4]xf16>")
    gc = tle.llvm_gep(cbase, n_start, "f16")
    tle.call_intrinsic("llvm.store", [acc_h, gc], result_type="()")


@triton.jit(do_not_specialize=["K", "N", "NB"])
def _mm_m1_host(x, W, C, K, N, NB):
    _sr_call(mm_m1_raw, outputs=[], inputs=[x, W, C, K, N, NB])


# ---------------------------------------------------------------------------
# M==1 GEMV transposed-b fast path (vfwmacc.vv, pure llvm-direct sibling).
#   For b = W.T where W is stored (N, K) row-major. b has shape (K, N) with
#   stride (1, K) — i.e. b.stride(0) == 1. The kernel interprets the raw
#   pointer as (N, K) row-major (W's actual layout) and computes
#   C[0, n] = sum_k x[k] * W[n, k] via vfwmacc.vv (vector·vector widening FMA).
#   Each program handles NB output rows, 8 rows unrolled per inner iteration
#   for ILP. Same pattern as test_raw_mv_three_stage.py::mv_vfwmacc_ci_parallel.
#   No host-side copy — reads W's storage directly via strided-free row-major
#   access (W[n, k:k+64] is contiguous because K is W's inner dim).
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_m1_t_raw(
    x_ptr: mem(f16),
    W_ptr: mem(f16),
    C_ptr: mem(f16, out=True),
    K: index,
    N: index,
    NB: index,
):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    xbase = tle.llvm_base_ptr(x_ptr)
    wbase = tle.llvm_base_ptr(W_ptr)
    cbase = tle.llvm_base_ptr(C_ptr)
    n_base = tle.program_id(0) * NB
    n_end = n_base + NB
    for ni in tle.range(n_base, n_end, 8):
        acc0 = zero_acc
        acc1 = zero_acc
        acc2 = zero_acc
        acc3 = zero_acc
        acc4 = zero_acc
        acc5 = zero_acc
        acc6 = zero_acc
        acc7 = zero_acc
        for ki in tle.range(zero_i64, K, vl):
            gx = tle.llvm_gep(xbase, ki, "f16")
            vx = tle.call_intrinsic("llvm.load", [gx], result_type="vector<[4]xf16>")
            gw0 = tle.llvm_gep(wbase, ni * K + ki, "f16")
            gw1 = tle.llvm_gep(wbase, (ni + 1) * K + ki, "f16")
            gw2 = tle.llvm_gep(wbase, (ni + 2) * K + ki, "f16")
            gw3 = tle.llvm_gep(wbase, (ni + 3) * K + ki, "f16")
            gw4 = tle.llvm_gep(wbase, (ni + 4) * K + ki, "f16")
            gw5 = tle.llvm_gep(wbase, (ni + 5) * K + ki, "f16")
            gw6 = tle.llvm_gep(wbase, (ni + 6) * K + ki, "f16")
            gw7 = tle.llvm_gep(wbase, (ni + 7) * K + ki, "f16")
            vw0 = tle.call_intrinsic("llvm.load", [gw0], result_type="vector<[4]xf16>")
            vw1 = tle.call_intrinsic("llvm.load", [gw1], result_type="vector<[4]xf16>")
            vw2 = tle.call_intrinsic("llvm.load", [gw2], result_type="vector<[4]xf16>")
            vw3 = tle.call_intrinsic("llvm.load", [gw3], result_type="vector<[4]xf16>")
            vw4 = tle.call_intrinsic("llvm.load", [gw4], result_type="vector<[4]xf16>")
            vw5 = tle.call_intrinsic("llvm.load", [gw5], result_type="vector<[4]xf16>")
            vw6 = tle.call_intrinsic("llvm.load", [gw6], result_type="vector<[4]xf16>")
            vw7 = tle.call_intrinsic("llvm.load", [gw7], result_type="vector<[4]xf16>")
            acc0 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc0, vx, vw0, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc1 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc1, vx, vw1, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc2 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc2, vx, vw2, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc3 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc3, vx, vw3, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc4 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc4, vx, vw4, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc5 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc5, vx, vw5, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc6 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc6, vx, vw6, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
            acc7 = tle.call_intrinsic(
                "llvm.riscv.vfwmacc",
                [acc7, vx, vw7, zero_i64, vl, zero_i64],
                result_type="vector<[4]xf32>",
            )
        s0 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32"
        )
        s1 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32"
        )
        s2 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32"
        )
        s3 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32"
        )
        s4 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32"
        )
        s5 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32"
        )
        s6 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32"
        )
        s7 = tle.call_intrinsic(
            "llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32"
        )
        h0 = tle.call_intrinsic("llvm.fptrunc", [s0], result_type="f16")
        h1 = tle.call_intrinsic("llvm.fptrunc", [s1], result_type="f16")
        h2 = tle.call_intrinsic("llvm.fptrunc", [s2], result_type="f16")
        h3 = tle.call_intrinsic("llvm.fptrunc", [s3], result_type="f16")
        h4 = tle.call_intrinsic("llvm.fptrunc", [s4], result_type="f16")
        h5 = tle.call_intrinsic("llvm.fptrunc", [s5], result_type="f16")
        h6 = tle.call_intrinsic("llvm.fptrunc", [s6], result_type="f16")
        h7 = tle.call_intrinsic("llvm.fptrunc", [s7], result_type="f16")
        tle.call_intrinsic(
            "llvm.store", [h0, tle.llvm_gep(cbase, ni, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h1, tle.llvm_gep(cbase, ni + 1, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h2, tle.llvm_gep(cbase, ni + 2, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h3, tle.llvm_gep(cbase, ni + 3, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h4, tle.llvm_gep(cbase, ni + 4, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h5, tle.llvm_gep(cbase, ni + 5, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h6, tle.llvm_gep(cbase, ni + 6, "f16")], result_type="()"
        )
        tle.call_intrinsic(
            "llvm.store", [h7, tle.llvm_gep(cbase, ni + 7, "f16")], result_type="()"
        )


@triton.jit(do_not_specialize=["K", "N", "NB"])
def _mm_m1_t_host(x, W, C, K, N, NB):
    _sr_call(mm_m1_t_raw, outputs=[], inputs=[x, W, C, K, N, NB])


# ---------------------------------------------------------------------------
# M>1 mm transposed-b kernel (prefill). Extends mm_m1_t_raw with an M-row
# loop: C[m,n] = sum_k a[m,k]*W[n,k]. Each program handles NB columns × all
# M rows; 8 columns unrolled, M iterated. W rows reused across m via L1 cache.
# Same vfwmacc.vv pattern. Qwen prefill (M=18/32, b=W.T) hits this 100%.
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mm_mN_t_raw(
    a_ptr: mem(f16),
    W_ptr: mem(f16),
    C_ptr: mem(f16, out=True),
    M: index,
    K: index,
    N: index,
    NB: index,
    CS: index,
):
    vl = tle.llvm_const(64, "i64")
    zero_i64 = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    abase = tle.llvm_base_ptr(a_ptr)
    wbase = tle.llvm_base_ptr(W_ptr)
    cbase = tle.llvm_base_ptr(C_ptr)
    n_base = tle.program_id(0) * NB
    n_end = n_base + NB
    for ni in tle.range(n_base, n_end, 8):
        for m in tle.range(0, M, 1):
            acc0 = zero_acc
            acc1 = zero_acc
            acc2 = zero_acc
            acc3 = zero_acc
            acc4 = zero_acc
            acc5 = zero_acc
            acc6 = zero_acc
            acc7 = zero_acc
            for ki in tle.range(zero_i64, K, vl):
                ga = tle.llvm_gep(abase, m * K + ki, "f16")
                va = tle.call_intrinsic(
                    "llvm.load", [ga], result_type="vector<[4]xf16>"
                )
                gw0 = tle.llvm_gep(wbase, ni * K + ki, "f16")
                gw1 = tle.llvm_gep(wbase, (ni + 1) * K + ki, "f16")
                gw2 = tle.llvm_gep(wbase, (ni + 2) * K + ki, "f16")
                gw3 = tle.llvm_gep(wbase, (ni + 3) * K + ki, "f16")
                gw4 = tle.llvm_gep(wbase, (ni + 4) * K + ki, "f16")
                gw5 = tle.llvm_gep(wbase, (ni + 5) * K + ki, "f16")
                gw6 = tle.llvm_gep(wbase, (ni + 6) * K + ki, "f16")
                gw7 = tle.llvm_gep(wbase, (ni + 7) * K + ki, "f16")
                vw0 = tle.call_intrinsic(
                    "llvm.load", [gw0], result_type="vector<[4]xf16>"
                )
                vw1 = tle.call_intrinsic(
                    "llvm.load", [gw1], result_type="vector<[4]xf16>"
                )
                vw2 = tle.call_intrinsic(
                    "llvm.load", [gw2], result_type="vector<[4]xf16>"
                )
                vw3 = tle.call_intrinsic(
                    "llvm.load", [gw3], result_type="vector<[4]xf16>"
                )
                vw4 = tle.call_intrinsic(
                    "llvm.load", [gw4], result_type="vector<[4]xf16>"
                )
                vw5 = tle.call_intrinsic(
                    "llvm.load", [gw5], result_type="vector<[4]xf16>"
                )
                vw6 = tle.call_intrinsic(
                    "llvm.load", [gw6], result_type="vector<[4]xf16>"
                )
                vw7 = tle.call_intrinsic(
                    "llvm.load", [gw7], result_type="vector<[4]xf16>"
                )
                acc0 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc0, va, vw0, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc1 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc1, va, vw1, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc2 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc2, va, vw2, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc3 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc3, va, vw3, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc4 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc4, va, vw4, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc5 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc5, va, vw5, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc6 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc6, va, vw6, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
                acc7 = tle.call_intrinsic(
                    "llvm.riscv.vfwmacc",
                    [acc7, va, vw7, zero_i64, vl, zero_i64],
                    result_type="vector<[4]xf32>",
                )
            s0 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32"
            )
            s1 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32"
            )
            s2 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32"
            )
            s3 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32"
            )
            s4 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32"
            )
            s5 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32"
            )
            s6 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32"
            )
            s7 = tle.call_intrinsic(
                "llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32"
            )
            h0 = tle.call_intrinsic("llvm.fptrunc", [s0], result_type="f16")
            h1 = tle.call_intrinsic("llvm.fptrunc", [s1], result_type="f16")
            h2 = tle.call_intrinsic("llvm.fptrunc", [s2], result_type="f16")
            h3 = tle.call_intrinsic("llvm.fptrunc", [s3], result_type="f16")
            h4 = tle.call_intrinsic("llvm.fptrunc", [s4], result_type="f16")
            h5 = tle.call_intrinsic("llvm.fptrunc", [s5], result_type="f16")
            h6 = tle.call_intrinsic("llvm.fptrunc", [s6], result_type="f16")
            h7 = tle.call_intrinsic("llvm.fptrunc", [s7], result_type="f16")
            cn0 = m * CS + ni
            tle.call_intrinsic(
                "llvm.store", [h0, tle.llvm_gep(cbase, cn0, "f16")], result_type="()"
            )
            tle.call_intrinsic(
                "llvm.store",
                [h1, tle.llvm_gep(cbase, cn0 + 1, "f16")],
                result_type="()",
            )
            tle.call_intrinsic(
                "llvm.store",
                [h2, tle.llvm_gep(cbase, cn0 + 2, "f16")],
                result_type="()",
            )
            tle.call_intrinsic(
                "llvm.store",
                [h3, tle.llvm_gep(cbase, cn0 + 3, "f16")],
                result_type="()",
            )
            tle.call_intrinsic(
                "llvm.store",
                [h4, tle.llvm_gep(cbase, cn0 + 4, "f16")],
                result_type="()",
            )
            tle.call_intrinsic(
                "llvm.store",
                [h5, tle.llvm_gep(cbase, cn0 + 5, "f16")],
                result_type="()",
            )
            tle.call_intrinsic(
                "llvm.store",
                [h6, tle.llvm_gep(cbase, cn0 + 6, "f16")],
                result_type="()",
            )
            tle.call_intrinsic(
                "llvm.store",
                [h7, tle.llvm_gep(cbase, cn0 + 7, "f16")],
                result_type="()",
            )


@triton.jit(do_not_specialize=["M", "K", "N", "NB", "CS"])
def _mm_mN_t_host(a, W, C, M, K, N, NB, CS):
    _sr_call(mm_mN_t_raw, outputs=[], inputs=[a, W, C, M, K, N, NB, CS])


# ---------------------------------------------------------------------------
# tl.dot mm kernel (standard Triton dot, spine-mlir-main-rv64 lowering).
#   Fallback for M>1 prefill when smt.dot path unavailable. Uses input strides.
#   1.27x-2.42x faster than vfwmacc.vv. smt.dot path is 4x faster than this.
# ---------------------------------------------------------------------------
@triton.jit
def mm_tldot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(0, tl.cdiv(K, BLOCK_K)):
        a_bp = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, ki * BLOCK_K),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_bp = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(ki * BLOCK_K, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        a = tl.load(a_bp, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_bp, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
    c_bp = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_bp, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# smt.dot mm kernel (K3 matrix engine via linalg.mmt4d). Fastest path.
#   K-loop with unpack-before-accumulate: smt.dot returns packed MICRO-tile
#   result; must smt.view(..., (1,1)) to unpack to plain tensor before +=.
#   Direct packed accumulate (acc += smt.dot(a,b) without unpack) produces
#   wrong results — tl.zeros+smt.view init creates incompatible tensor format.
#   BK=256 fixed (K3 MICRO_K=8, 32 K-tiles per BK). 4x faster than tl.dot.
#   TCM ~256KB (within 384KB limit); noloop BK=K variant needed 608KB (over).
# ---------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_spacemit"),
    key=["M", "N", "K"],
    disable_aabs=True,
)
@triton.jit
def mm_smt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_bp = tl.make_block_ptr(
            base=a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            offsets=[pid_m * BLOCK_SIZE_M, ki * BLOCK_SIZE_K],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            order=[1, 0],
        )
        b_bp = tl.make_block_ptr(
            base=b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            offsets=[ki * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            order=[1, 0],
        )
        a_desc = smt.descriptor_load(a_bp, (0, 0))
        a = smt.view(a_desc, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
        b_desc = smt.descriptor_load(b_bp, (0, 0))
        b = smt.view(b_desc, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        partial = smt.dot(a, b)
        partial = smt.view(partial, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
        acc += partial
    c = acc.to(c_ptr.dtype.element_ty)
    c_bp = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_bp, c, boundary_check=(0, 1))


_TLDOT_BK = 256
_TLDOT_BM = 32
_SMT_BK = 256
_SMT_BM = 32


def _pick_smt_bn(N):
    """Per-shape BN. 256 for N>=2048, else 64. BN=512 crashes the kernel
    (tile size limit, rule 53 pattern) so large N is handled by chunked
    launches in _mm_smt_path instead of bumping BN."""
    if N >= 2048:
        return 256
    return 64


# BN=256, max grid_n=512 → max cols per kernel launch (rule 56: grid>512 corrupts).
_SMT_MAX_COLS_PER_LAUNCH = 256 * 512  # 131072


def _mm_smt_path(a, b, c, M, K, N):
    """smt.dot matrix-engine fast path for M>1 prefill. 4x faster than tl.dot
    for proj mm shapes (N≤4096). For very large N (lm_head N=151936) chunked
    launch with BN=256 is slower than vfwmacc.vv NB=512 (smt.dot BN=512 crashes
    — matrix engine tile limit stricter than vector unit), so caller skips
    smt.dot when N > _SMT_MAX_COLS_PER_LAUNCH.

    Returns True on success, False to fall back. Handles arbitrary N via
    chunked launches (each ≤131072 cols, BN=256) + per-chunk main/tail split."""
    bn = _pick_smt_bn(N)
    if N > _SMT_MAX_COLS_PER_LAUNCH:
        # Chunked smt.dot would be slower than vfwmacc.vv NB=512 — let caller fall back.
        return False
    grid_m = (M + _SMT_BM - 1) // _SMT_BM
    n_done = 0
    while n_done < N:
        chunk_n = min(_SMT_MAX_COLS_PER_LAUNCH, N - n_done)
        n_main = (chunk_n // bn) * bn
        n_tail = chunk_n - n_main
        if n_main > 0:
            grid_n = n_main // bn
            if grid_m * grid_n > 512:
                return False
            a_chunk = a
            b_chunk = b[:, n_done : n_done + n_main]
            c_chunk = c[:, n_done : n_done + n_main]
            mm_smt_kernel[(grid_m, grid_n)](
                a_chunk,
                b_chunk,
                c_chunk,
                M,
                n_main,
                K,
                a_chunk.stride(0),
                a_chunk.stride(1),
                b_chunk.stride(0),
                b_chunk.stride(1),
                c_chunk.stride(0),
                c_chunk.stride(1),
                BLOCK_SIZE_M=_SMT_BM,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=_SMT_BK,
            )
        if n_tail > 0:
            c_tail = c[:, n_done + n_main : n_done + n_main + n_tail]
            b_tail = b[:, n_done + n_main : n_done + n_main + n_tail]
            _mm_fallback(a, b_tail, c_tail)
        n_done += chunk_n
    return True


def _pick_tldot_bn(N):
    """Per-shape BN from sweep. Returns 0 if N too large (fall back).
    N>4096 falls back (BN=512 crashes tl.dot too — same K3 tile limit as smt.dot;
    vfwmacc.vv handles large N via _mm_transposed_tail)."""
    if N <= 1024:
        return 64
    elif N <= 2048:
        return 256
    elif N <= 4096:
        return 128
    return 0


def _mm_tldot_path(a, b, c, M, K, N):
    """tl.dot fallback for M>1 prefill, reading B with its original strides.
    Returns True on success, False to fall back. Only for proj mms."""
    bn = _pick_tldot_bn(N)
    if bn == 0:
        return False
    grid_m = (M + _TLDOT_BM - 1) // _TLDOT_BM
    grid_n = (N + bn - 1) // bn
    if grid_m * grid_n > 512:
        return False
    mm_tldot_kernel[(grid_m, grid_n)](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_TLDOT_BM,
        BLOCK_N=bn,
        BLOCK_K=_TLDOT_BK,
    )
    return True


def _mm_transposed_tail(a, b, c, M, K, N):
    """vfwmacc fast path for transposed b (b.stride(0)==1, b=W.T).
    Splits N into main tile (multiple of NB, vfwmacc) + tail (vfwmacc with
    shifted pointer views). Writes into c (pre-allocated (M,N)). Returns True.

    NB auto-bumped to keep grid <= 512 (K3 runtime limit: grid>512 corrupts —
    programs silently drop or write zeros). n_main=151552 (lm_head) with NB=512
    gives grid=296. n_tail=384 handled by vfwmacc kernel with grid=(1,), NB=n_tail.

    Tail uses strided views b[:, n_main:] / c[:, n_main:] whose data_ptr is
    shifted by n_main elements — kernel reads W[n_main+ni, ki] and writes
    c[m, n_main+ni] correctly. Constraint: n_tail % 8 == 0 (kernel 8-col unroll).
    """
    GRID_LIMIT = 512
    NB = 256
    if N // NB > GRID_LIMIT and N >= 512:
        NB = 512
    n_main = (N // NB) * NB
    n_tail = N - n_main
    if n_main > 0:
        if M == 1:
            grid = (n_main // NB,)
            _mm_m1_t_host[grid](a, b, c, K, n_main, NB=NB)
        else:
            grid = (n_main // NB,)
            _mm_mN_t_host[grid](a, b, c, M, K, n_main, NB=NB, CS=N)
    if n_tail > 0:
        # tail: vfwmacc kernel with shifted pointer views (no temp, no native addmm).
        # b[:, n_main:] has stride (1, K) and data_ptr = b.data_ptr() + n_main*K*2,
        # which is exactly W[n_main:, :] row-major — kernel's wbase[ni*K+ki] reads
        # W[n_main+ni, ki]. c[:, n_main:] similarly shifts cbase so writes land at
        # c[m, n_main+ni]. CS=N (full row stride) keeps c layout correct.
        b_tail = b[:, n_main:]
        c_tail = c[:, n_main:]
        # n_tail is always 8-aligned: callers gate N%8==0 and NB is
        # 256/512, so n_main and n_tail are both multiples of 8.
        if M == 1:
            _mm_m1_t_host[(1,)](a, b_tail, c_tail, K, n_tail, NB=n_tail)
        else:
            _mm_mN_t_host[(1,)](a, b_tail, c_tail, M, K, n_tail, NB=n_tail, CS=N)
    return True


def _check_mm_inputs(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise RuntimeError("mm expects two matrices with compatible dimensions")
    if a.dtype != b.dtype or a.device != b.device:
        raise RuntimeError("mm inputs must have the same dtype and device")
    if a.dtype not in (torch.float16, torch.float32):
        raise RuntimeError("SpacemiT mm only supports float16 and float32")
    return a.shape[0], a.shape[1], b.shape[1]


def mm(a, b):
    logger.debug("GEMS_SPACEMIT MM")
    M, K, N = _check_mm_inputs(a, b)
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # vfwmacc fast paths (pure llvm-direct sibling, no host-side copy):
    #   M==1 decode:  mm_m1_raw (vfwmacc.vf, contiguous b) or
    #                 mm_m1_t_raw (vfwmacc.vv, transposed b=W.T)
    #   M>1  prefill: mm_mN_t_raw (vfwmacc.vv, transposed b=W.T, M-row loop)
    # Qwen hits transposed b 100% (weights (out,in) row-major, mm(x, W.T)).
    # M==1 reads the original layout directly to avoid a weight copy.
    # Tail handling: N%8==0 suffices for the raw vector kernels.
    if a.dtype == torch.float16 and a.is_contiguous() and K > 0 and K % 64 == 0:
        if M == 1:
            if b.is_contiguous() and b.stride(1) == 1 and N % 64 == 0:
                NB = 64
                grid = (N // NB,)
                _mm_m1_host[grid](a, b, c, K, N, NB=NB)
                return c
            if b.stride(0) == 1 and b.stride(1) == K and N % 8 == 0:
                _mm_transposed_tail(a, b, c, M, K, N)
                return c
        else:
            # M>1 prefill: try smt.dot (matrix engine, 4x faster) first,
            # fall back to tl.dot, then vfwmacc. Only for proj mms (N<=4096).
            # SMT/tl.dot read contiguous and transposed B using its strides.
            # The raw vfwmacc path requires tightly packed rows of W=B.T.
            if (b.stride(0) == 1 or b.stride(1) == 1) and N % 8 == 0:
                if _mm_smt_path(a, b, c, M, K, N):
                    return c
                if _mm_tldot_path(a, b, c, M, K, N):
                    return c
                if b.stride(0) == 1 and b.stride(1) == K:
                    _mm_transposed_tail(a, b, c, M, K, N)
                    return c
    return _mm_fallback(a, b, c)


def mm_out(a, b, *, out):
    logger.debug("GEMS_SPACEMIT MM_OUT")
    M, K, N = _check_mm_inputs(a, b)
    if out.dtype != a.dtype or out.device != a.device:
        raise RuntimeError("mm output must match the input dtype and device")
    if out.shape != (M, N):
        out.resize_((M, N))

    if a.dtype == torch.float16 and a.is_contiguous() and K > 0 and K % 64 == 0:
        if M == 1 and out.is_contiguous():
            if b.is_contiguous() and b.stride(1) == 1 and N % 64 == 0:
                NB = 64
                grid = (N // NB,)
                _mm_m1_host[grid](a, b, out, K, N, NB=NB)
                return out
            if b.stride(0) == 1 and b.stride(1) == K and N % 8 == 0:
                _mm_transposed_tail(a, b, out, M, K, N)
                return out
        elif M > 1:
            # SMT/tl.dot use B's strides; the raw fallback needs packed W=B.T.
            if (b.stride(0) == 1 or b.stride(1) == 1) and N % 8 == 0:
                if _mm_smt_path(a, b, out, M, K, N):
                    return out
                if _mm_tldot_path(a, b, out, M, K, N):
                    return out
                if out.is_contiguous() and b.stride(0) == 1 and b.stride(1) == K:
                    _mm_transposed_tail(a, b, out, M, K, N)
                    return out

    return _mm_fallback(a, b, out)
