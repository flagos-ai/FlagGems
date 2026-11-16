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

"""Triton implementation of ``torch.inner`` (generalized inner product).

Contract mirrored from the reference:
  * both 0-dim         -> scalar product (output shape ())
  * dtype mismatch     -> RuntimeError("both inputs should have same dtype")
  * one operand 0-dim  -> scale the other operand (output shape = other.shape)
  * last dims differ   -> RuntimeError("inner() the last dimension must match ...")
  * both 1-dim         -> scalar dot product (output shape ())
  * otherwise          -> out[i..., j...] = sum_k a[i..., k] * b[j..., k]
                          output shape = a.shape[:-1] + b.shape[:-1]

Precision policy (measured on the MetaX C550 target with a debug probe):
  * fp16/bf16 : tensor-core tl.dot with fp32 accumulation.
  * fp32      : tl.dot(..., input_precision="ieee"); the backend default (tf32)
                loses ~1.8e-2 relative and is rejected.
  * fp64      : tl.dot produced catastrophically wrong elements on this
                backend, so an explicit broadcast/FMA accumulation in fp64 is
                used instead (verified exact against torch).
  * integer   : explicit broadcast accumulation in the operand dtype (modular
                arithmetic keeps wraparound semantics).

Both operands are K-major (the contraction dim is contiguous), so the B tile can
be loaded either as [BLOCK_K, BLOCK_N] (a "transposed" load) or as
[BLOCK_N, BLOCK_K] followed by ``tl.trans``.  Which one lets the backend use its
matrix units is measured, not assumed: the large fp16/bf16 GEMM path runs under
a bounded autotune whose config space includes both formulations.
"""

import torch
import triton
import triton.language as tl

_MAX_LEAD = 8

# accumulation modes (plain ints for the host; the kernels below use the
# matching literals because Triton kernels may not read non-constexpr globals)
_M_FP16 = 0  # tensor-core dot, fp32 accumulator
_M_FP32 = 1  # ieee dot, fp32 accumulator
_M_FP64 = 2  # explicit fp64 accumulate
_M_INT = 3  # explicit integer accumulate

# Bounded configuration space for the large fp16/bf16 GEMM path.  The first
# four configs load the B tile as [BLOCK_K, BLOCK_N]; the last four load it as
# [BLOCK_N, BLOCK_K] and transpose in registers.
_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": False},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "TRANS_B": False},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "TRANS_B": False},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": False},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": True},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "TRANS_B": True},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "TRANS_B": True},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": True},
        num_warps=8,
        num_stages=3,
    ),
    # Narrow tiles: the medium shapes only fill a fraction of the 104 SMs with
    # 128- or 64-wide tiles, so 256-block tilings are offered as well.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": True},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": True},
        num_warps=4,
        num_stages=3,
    ),
]

# fp32 needs its own list: an ieee dot on fp32 operands uses twice the shared
# memory per stage, so the fp16 frontier tiles are not all legal here.
_GEMM_F32_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": False},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": False},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": False},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": False},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "TRANS_B": False},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "TRANS_B": True},
        num_warps=8,
        num_stages=2,
    ),
]

# only large problems pay for the configuration search
_AUTOTUNE_MIN_WORK = 4 * 10**8


@triton.jit
def _swalk(row, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7):
    """Flat index -> storage offset for a collapsed (size, stride) dim list."""
    rem = row
    off = (rem % s7) * t7
    rem = rem // s7
    off += (rem % s6) * t6
    rem = rem // s6
    off += (rem % s5) * t5
    rem = rem // s5
    off += (rem % s4) * t4
    rem = rem // s4
    off += (rem % s3) * t3
    rem = rem // s3
    off += (rem % s2) * t2
    rem = rem // s2
    off += (rem % s1) * t1
    rem = rem // s1
    off += (rem % s0) * t0
    return off


@triton.jit
def _gemm_body(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    offs_m,
    offs_n,
    rowa,
    rowb,
    a_last,
    b_last,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_EVEN: tl.constexpr,
    MN_EVEN: tl.constexpr,
    MODE: tl.constexpr,
    TRANS_B: tl.constexpr,
    ALIGN_K: tl.constexpr,
):
    mask_m = offs_m < M
    mask_n = offs_n < N
    if ALIGN_K > 1:
        # rowa/rowb are exact multiples of the operands' row stride, so each
        # row start is vector-load aligned.  The compiler cannot infer that
        # from a runtime stride, and without it per-row loads are issued at
        # element granularity.
        rowa = tl.multiple_of(rowa, ALIGN_K)
        rowb = tl.multiple_of(rowb, ALIGN_K)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rowa[:, None] + offs_k[None, :] * a_last

    if MODE == 0 or MODE == 1:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=a_ptr.dtype.element_ty)

    if TRANS_B:
        b_ptrs = b_ptr + rowb[:, None] + offs_k[None, :] * b_last
        for k0 in range(0, K, BLOCK_K):
            if K_EVEN:
                if MN_EVEN:
                    a = tl.load(a_ptrs)
                    bt = tl.load(b_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
                    bt = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
            else:
                k_rem = K - k0
                if MN_EVEN:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
                    bt = tl.load(b_ptrs, mask=offs_k[None, :] < k_rem, other=0)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & (offs_k[None, :] < k_rem),
                        other=0,
                    )
                    bt = tl.load(
                        b_ptrs,
                        mask=mask_n[:, None] & (offs_k[None, :] < k_rem),
                        other=0,
                    )
            b = tl.trans(bt)
            if MODE == 0:
                acc = tl.dot(a, b, acc)
            elif MODE == 1:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
            a_ptrs += BLOCK_K * a_last
            b_ptrs += BLOCK_K * b_last
    else:
        b_ptrs = b_ptr + rowb[None, :] + offs_k[:, None] * b_last
        for k0 in range(0, K, BLOCK_K):
            if K_EVEN:
                if MN_EVEN:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
                    b = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
            else:
                k_rem = K - k0
                if MN_EVEN:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & (offs_k[None, :] < k_rem),
                        other=0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_k[:, None] < k_rem) & mask_n[None, :],
                        other=0,
                    )
            if MODE == 0:
                acc = tl.dot(a, b, acc)
            elif MODE == 1:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
            a_ptrs += BLOCK_K * a_last
            b_ptrs += BLOCK_K * b_last

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if MN_EVEN:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _swizzle(
    pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr
):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _inner_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_last,
    b_last,
    a_row,
    b_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_EVEN: tl.constexpr,
    MN_EVEN: tl.constexpr,
    MODE: tl.constexpr,
    TRANS_B: tl.constexpr,
    ALIGN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    _gemm_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        offs_m,
        offs_n,
        offs_m * a_row,
        offs_n * b_row,
        a_last,
        b_last,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_EVEN,
        MN_EVEN,
        MODE,
        TRANS_B,
        ALIGN_K,
    )


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K", "MODE"])
@triton.jit
def _inner_gemm_auto_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_last,
    b_last,
    a_row,
    b_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TRANS_B: tl.constexpr,
    K_EVEN: tl.constexpr,
    MN_EVEN: tl.constexpr,
    MODE: tl.constexpr,
    ALIGN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    _gemm_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        offs_m,
        offs_n,
        offs_m * a_row,
        offs_n * b_row,
        a_last,
        b_last,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_EVEN,
        MN_EVEN,
        MODE,
        TRANS_B,
        ALIGN_K,
    )


@triton.autotune(configs=_GEMM_F32_CONFIGS, key=["M", "N", "K", "MODE"])
@triton.jit
def _inner_gemm_auto_f32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_last,
    b_last,
    a_row,
    b_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TRANS_B: tl.constexpr,
    K_EVEN: tl.constexpr,
    MN_EVEN: tl.constexpr,
    MODE: tl.constexpr,
    ALIGN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    _gemm_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        offs_m,
        offs_n,
        offs_m * a_row,
        offs_n * b_row,
        a_last,
        b_last,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_EVEN,
        MN_EVEN,
        MODE,
        TRANS_B,
        ALIGN_K,
    )


@triton.jit
def _inner_gemm_strided_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_last,
    b_last,
    as0,
    as1,
    as2,
    as3,
    as4,
    as5,
    as6,
    as7,
    at0,
    at1,
    at2,
    at3,
    at4,
    at5,
    at6,
    at7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    bt0,
    bt1,
    bt2,
    bt3,
    bt4,
    bt5,
    bt6,
    bt7,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_EVEN: tl.constexpr,
    MN_EVEN: tl.constexpr,
    MODE: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rowa = _swalk(
        offs_m,
        as0,
        as1,
        as2,
        as3,
        as4,
        as5,
        as6,
        as7,
        at0,
        at1,
        at2,
        at3,
        at4,
        at5,
        at6,
        at7,
    )
    rowb = _swalk(
        offs_n,
        bs0,
        bs1,
        bs2,
        bs3,
        bs4,
        bs5,
        bs6,
        bs7,
        bt0,
        bt1,
        bt2,
        bt3,
        bt4,
        bt5,
        bt6,
        bt7,
    )
    _gemm_body(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        offs_m,
        offs_n,
        rowa,
        rowb,
        a_last,
        b_last,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_EVEN,
        MN_EVEN,
        MODE,
        TRANS_B,
        1,
    )


@triton.jit
def _inner_dot1_kernel(
    a_ptr, b_ptr, c_ptr, K, sa, sb, BLOCK: tl.constexpr, MODE: tl.constexpr
):
    """Whole contraction in one program, so the scalar dot costs one launch."""
    offs = tl.arange(0, BLOCK)
    if MODE == 0 or MODE == 1:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK,), dtype=a_ptr.dtype.element_ty)
    for k0 in range(0, K, BLOCK):
        cur = k0 + offs
        mask = cur < K
        a = tl.load(a_ptr + cur * sa, mask=mask, other=0)
        b = tl.load(b_ptr + cur * sb, mask=mask, other=0)
        if MODE == 0 or MODE == 1:
            acc += a.to(tl.float32) * b.to(tl.float32)
        else:
            acc += a * b
    tl.store(c_ptr, tl.sum(acc, axis=0).to(c_ptr.dtype.element_ty))


@triton.jit
def _inner_dot_kernel(
    a_ptr, b_ptr, mid_ptr, K, sa, sb, BLOCK: tl.constexpr, MODE: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MODE == 0 or MODE == 1:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK,), dtype=a_ptr.dtype.element_ty)
    step = num_programs * BLOCK
    for k0 in range(0, K, step):
        cur = k0 + offs
        mask = cur < K
        a = tl.load(a_ptr + cur * sa, mask=mask, other=0)
        b = tl.load(b_ptr + cur * sb, mask=mask, other=0)
        if MODE == 0 or MODE == 1:
            acc += a.to(tl.float32) * b.to(tl.float32)
        else:
            acc += a * b
    tl.store(mid_ptr + pid, tl.sum(acc, axis=0).to(mid_ptr.dtype.element_ty))


@triton.jit
def _inner_dot_reduce_kernel(mid_ptr, c_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    partials = tl.load(mid_ptr + offs, mask=offs < n, other=0)
    tl.store(c_ptr, tl.sum(partials, axis=0).to(c_ptr.dtype.element_ty))


@triton.jit
def _inner_scale_kernel(
    x_ptr,
    s_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    src = _swalk(offs, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7)
    x = tl.load(x_ptr + src, mask=mask, other=0)
    s = tl.load(s_ptr)
    tl.store(
        o_ptr + offs,
        x.to(o_ptr.dtype.element_ty) * s.to(o_ptr.dtype.element_ty),
        mask=mask,
    )


def _collapse_dims(shape, strides):
    """Drop size-1 dims and merge contiguous runs; returns outer->inner list."""
    dims = [(int(s), int(st)) for s, st in zip(shape, strides) if int(s) != 1]
    merged = []
    for s, st in reversed(dims):
        if merged and st == merged[-1][0] * merged[-1][1]:
            merged[-1] = (merged[-1][0] * s, merged[-1][1])
        else:
            merged.append((s, st))
    merged.reverse()
    return merged


def _pad_dims(dims):
    p = len(dims)
    sizes = [d[0] for d in dims] + [1] * (_MAX_LEAD - p)
    strides = [d[1] for d in dims] + [0] * (_MAX_LEAD - p)
    return sizes, strides


def _mode_of(dtype):
    if dtype == torch.float32:
        return _M_FP32
    if dtype == torch.float64:
        return _M_FP64
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return _M_FP16
    return _M_INT


def _select_gemm_config(M, N, K, mode):
    if mode >= _M_FP64:
        return 16, 16, 8, 8, 4, 2
    bm = min(64, max(16, triton.next_power_of_2(M)))
    bn = min(64, max(16, triton.next_power_of_2(N)))
    return bm, bn, 32, 8, 4, 3


def _sm_count(device):
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 104


def _launch_scale(x, s, out):
    n = out.numel()
    if n == 0:
        return out
    dims = _collapse_dims(x.shape, x.stride())
    if len(dims) > _MAX_LEAD:
        raise NotImplementedError("inner: rank too large for strided kernel")
    sizes, strides = _pad_dims(dims)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _inner_scale_kernel[grid](
        x,
        s,
        out,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def _launch_dot(a, b, out, K, mode):
    device = a.device
    is64 = mode == _M_FP64
    is_int = mode == _M_INT
    if K <= 65536:
        block = min(max(triton.next_power_of_2(K), 64), 8192)
        warps = 8 if block >= 2048 else 4
        _inner_dot1_kernel[(1,)](
            a,
            b,
            out,
            K,
            a.stride(0),
            b.stride(0),
            BLOCK=block,
            MODE=mode,
            num_warps=warps,
        )
        return out
    if is64:
        mid_dtype = torch.float64
        block = 128
    elif is_int:
        mid_dtype = a.dtype
        block = 256
    else:
        mid_dtype = torch.float32
        block = 256
    limit = max(1, 4 * _sm_count(device))
    n_prog = min(max(1, triton.cdiv(K, block)), limit)
    mid = torch.empty(n_prog, dtype=mid_dtype, device=device)
    _inner_dot_kernel[(n_prog,)](
        a, b, mid, K, a.stride(0), b.stride(0), BLOCK=block, MODE=mode, num_warps=4
    )
    _inner_dot_reduce_kernel[(1,)](
        mid, out, n_prog, BLOCK=triton.next_power_of_2(n_prog), num_warps=4
    )
    return out


def inner(self, other):
    a = self
    b = other

    adim = a.dim()
    bdim = b.dim()

    if adim == 0 and bdim == 0:
        out = torch.empty((), dtype=torch.result_type(a, b), device=a.device)
        return _launch_scale(a, b, out)

    if a.dtype != b.dtype:
        raise RuntimeError("both inputs should have same dtype")

    if adim == 0 or bdim == 0:
        if adim == 0:
            out = torch.empty(b.shape, dtype=b.dtype, device=b.device)
            return _launch_scale(b, a, out)
        out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
        return _launch_scale(a, b, out)

    if a.size(-1) != b.size(-1):
        raise RuntimeError(
            "inner() the last dimension must match on both input tensors but "
            f"got shapes {list(a.shape)} and {list(b.shape)}"
        )

    k = a.size(-1)
    out_shape = a.shape[:-1] + b.shape[:-1]

    m = 1
    for d in a.shape[:-1]:
        m *= d
    n = 1
    for d in b.shape[:-1]:
        n *= d

    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    if m == 0 or n == 0:
        return out

    mode = _mode_of(a.dtype)

    if adim == 1 and bdim == 1:
        return _launch_dot(a, b, out, k, mode)

    adims = _collapse_dims(a.shape[:-1], a.stride()[:-1])
    bdims = _collapse_dims(b.shape[:-1], b.stride()[:-1])
    if len(adims) > _MAX_LEAD or len(bdims) > _MAX_LEAD:
        raise NotImplementedError("inner: rank too large for strided kernel")

    a_last = a.stride(-1)
    b_last = b.stride(-1)

    if len(adims) <= 1 and len(bdims) <= 1:
        a_row = adims[0][1] if adims else 0
        b_row = bdims[0][1] if bdims else 0
        # Per-row starts stay 16-byte aligned when the contraction axis is
        # unit-stride and each row stride is a whole number of vectors.
        vec = 16 // a.element_size()
        if a_last == 1 and b_last == 1 and a_row % vec == 0 and b_row % vec == 0:
            align = vec
        else:
            align = 1
        if mode <= _M_FP32 and m * n * k >= _AUTOTUNE_MIN_WORK:
            # Every config's block size (32/64/128 rows, 32/64 contraction
            # lanes) divides the shape, so the tile masks are provably
            # redundant and are elided for the whole search space.
            even_all = (m % 128 == 0) and (n % 128 == 0) and (k % 64 == 0)
            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            )
            if mode == _M_FP16:
                _inner_gemm_auto_kernel[grid](
                    a,
                    b,
                    out,
                    m,
                    n,
                    k,
                    a_last,
                    b_last,
                    a_row,
                    b_row,
                    K_EVEN=even_all,
                    MN_EVEN=even_all,
                    MODE=mode,
                    ALIGN_K=align,
                )
            else:
                _inner_gemm_auto_f32_kernel[grid](
                    a,
                    b,
                    out,
                    m,
                    n,
                    k,
                    a_last,
                    b_last,
                    a_row,
                    b_row,
                    K_EVEN=even_all,
                    MN_EVEN=even_all,
                    MODE=mode,
                    ALIGN_K=align,
                )
            return out
        bm, bn, bk, group_m, warps, stages = _select_gemm_config(m, n, k, mode)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        mn_even = (m % bm) == 0 and (n % bn) == 0
        _inner_gemm_kernel[grid](
            a,
            b,
            out,
            m,
            n,
            k,
            a_last,
            b_last,
            a_row,
            b_row,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=group_m,
            K_EVEN=(k % bk) == 0,
            MN_EVEN=mn_even,
            MODE=mode,
            TRANS_B=False,
            ALIGN_K=align,
            num_warps=warps,
            num_stages=stages,
        )
        return out

    bm, bn, bk, group_m, warps, stages = _select_gemm_config(m, n, k, mode)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    mn_even = (m % bm) == 0 and (n % bn) == 0
    asz, ast = _pad_dims(adims)
    bsz, bst = _pad_dims(bdims)
    _inner_gemm_strided_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a_last,
        b_last,
        asz[0],
        asz[1],
        asz[2],
        asz[3],
        asz[4],
        asz[5],
        asz[6],
        asz[7],
        ast[0],
        ast[1],
        ast[2],
        ast[3],
        ast[4],
        ast[5],
        ast[6],
        ast[7],
        bsz[0],
        bsz[1],
        bsz[2],
        bsz[3],
        bsz[4],
        bsz[5],
        bsz[6],
        bsz[7],
        bst[0],
        bst[1],
        bst[2],
        bst[3],
        bst[4],
        bst[5],
        bst[6],
        bst[7],
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=group_m,
        K_EVEN=(k % bk) == 0,
        MN_EVEN=mn_even,
        MODE=mode,
        TRANS_B=False,
        num_warps=warps,
        num_stages=stages,
    )
    return out
