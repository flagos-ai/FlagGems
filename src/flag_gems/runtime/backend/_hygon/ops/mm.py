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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def mm_kernel(
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
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)

    # --------------------------
    # match naming: num_pid_m, num_pid_n
    # --------------------------
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # reorder for L2
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --------------------------
    # match naming: offs_am, offs_bn, offs_k
    # --------------------------
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # contiguous aligned offsets (ram/rbn → offs_am/offs_bn)
    offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
    offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    # previous K multiple
    # prev_k_mult = prev_multiple_of(K, BLOCK_K)
    prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K

    # accumulator
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --------------------------
    # main K loop
    # --------------------------
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k

        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak)
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn)
        )

        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)

        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # --------------------------
    # loop peel
    # --------------------------
    rk = prev_k_mult + offs_k
    mask_k = rk < K

    if IS_FP64:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
    else:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
        )

    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)

    if IS_FP64:
        accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
    else:
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # cast to output dtype
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # --------------------------
    # rematerialize offsets for store
    # (match naming: offs_cm, offs_cn)
    # --------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    tl.store(c_ptr, accumulator, mask=mask_store)


@libentry()
@triton.jit
def _mm_tiled(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = pid // tl.cdiv(N, BN)
    pn = pid % tl.cdiv(N, BN)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = step * BK + rk
        av = tl.load(
            A + rm[:, None] * SAM + kk[None, :] * SAK,
            mask=(rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            B + kk[:, None] * SBK + rn[None, :] * SBN,
            mask=(kk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _mm_vector(
    A,
    B,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK)
    av = tl.load(
        A + rm[:, None] * SAM + rk[None, :] * SAK,
        mask=(rm[:, None] < M) & (rk[None, :] < K),
        other=0.0,
    )
    bv = tl.load(B + rk * SBK, mask=rk < K, other=0.0)
    out = tl.sum(av * bv[None, :], axis=1)
    tl.store(C + rm, out, mask=rm < M)


@libentry()
@triton.jit
def _mm_rank_one_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    SAM: tl.constexpr,
    SBN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // N
    columns = offsets % N
    a = tl.load(A + rows * SAM, mask=offsets < M * N, other=0)
    b = tl.load(B + columns * SBN, mask=offsets < M * N, other=0)
    tl.store(C + offsets, a * b, mask=offsets < M * N)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm(a, b):
    logger.debug("GEMS_HYGON MM")
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[0]
        or a.dtype != b.dtype
        or a.device != b.device
        or a.dtype not in _ordered_datatypes
    ):
        return _mm_default(a, b)

    M, K = a.shape
    N = b.shape[1]
    if M == 0 or N == 0:
        return torch.empty((M, N), dtype=a.dtype, device=a.device)
    if K == 0:
        return torch.zeros((M, N), dtype=a.dtype, device=a.device)
    strides = a.stride() + b.stride()
    if any(stride < 0 for stride in strides):
        return _mm_default(a, b)
    if K == 1 and a.dtype in (torch.float32, torch.float64):
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
        with torch_device_fn.device(a.device):
            _mm_rank_one_kernel[(triton.cdiv(M * N, 256),)](
                a,
                b,
                out,
                M,
                N,
                a.stride(0),
                b.stride(1),
                BLOCK=256,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        return out

    # Bound the constexpr FP32 schedules and their 32-bit input address arithmetic.
    # The original implementation handles other types and larger address ranges.
    if (
        a.dtype != torch.float32
        or M > 1048576
        or N > 4096
        or K > 4096
        or M * N >= 2**31
        or (M - 1) * a.stride(0) + (K - 1) * a.stride(1) >= 2**31
        or (K - 1) * b.stride(0) + (N - 1) * b.stride(1) >= 2**31
    ):
        return _mm_default(a, b)
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    with torch_device_fn.device(a.device):
        if N == 1 and K <= 256:
            _mm_vector[(triton.cdiv(M, 16),)](
                a,
                b,
                out,
                M,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                BM=16,
                BK=triton.next_power_of_2(K),
                num_warps=4,
                num_stages=1,
            )
        else:
            if M >= 32768 and N >= 64:
                BM, BN, BK, warps = 64, 128, 64, 8
            elif M >= 1024 and N >= 64:
                BM, BN, BK, warps = 32, 64, (64 if K >= 256 and N <= 128 else 32), 4
            else:
                BM, BN, BK, warps = (
                    (16 if N <= 48 else 32),
                    (16 if N <= 48 else 32),
                    32,
                    (1 if N <= 48 else 4),
                )
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            _mm_tiled[grid](
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
                BM=BM,
                BN=BN,
                BK=BK,
                num_warps=warps,
                num_stages=1,
            )
    return out


def _mm_default(a, b):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
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
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            GROUP_M=8,
            num_stages=1,
            num_warps=4,
            num_ldmatrixes=0,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_HYGON MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
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
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            GROUP_M=8,
            num_stages=1,
            num_warps=4,
            num_ldmatrixes=0,
            IS_FP64=a.dtype == torch.float64,
        )
    return c
