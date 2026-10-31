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

"""Static MetaX GEMM configurations derived from measured candidates.

The public dispatcher admits FP32 only when TF32 is explicitly enabled.
Other inputs retain the existing implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_dot(a, b, acc, PREC: tl.constexpr, IS_F64: tl.constexpr):
    if IS_F64:
        return tl.dot(a, b, acc)
    elif PREC == 1:
        return tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="ieee")
    elif PREC == 2:
        return tl.dot(a, b, acc, out_dtype=tl.float32, input_precision="tf32")
    else:
        return tl.dot(a, b, acc, out_dtype=tl.float32)


@triton.jit
def _mm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sam: tl.constexpr,
    sak: tl.constexpr,
    sbk: tl.constexpr,
    sbn: tl.constexpr,
    scm: tl.constexpr,
    scn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    PREC: tl.constexpr,
    MODE: tl.constexpr,
    TN: tl.constexpr,
    IS_F64: tl.constexpr,
    MIXED: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GM * grid_n
    gid = pid // width
    gsize = tl.minimum(grid_m - gid * GM, GM)
    pid_m = gid * GM + pid % gsize
    pid_n = pid % width // gsize
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk0 = tl.arange(0, BK)
    if TN:
        if IS_F64:
            acc = tl.zeros((BN, BM), dtype=tl.float64)
        else:
            acc = tl.zeros((BN, BM), dtype=tl.float32)
        for k0 in range(0, K // BK * BK if MODE == 1 else K, BK):
            if MODE == 0:
                rk = k0 + rk0
                km = rk < K
                at = tl.load(
                    A + rk[:, None] * sak + rm[None, :] * sam,
                    mask=km[:, None] & (rm[None, :] < M),
                    other=0.0,
                )
                bt = tl.load(
                    B + rn[:, None] * sbn + rk[None, :] * sbk,
                    mask=km[None, :] & (rn[:, None] < N),
                    other=0.0,
                )
            else:
                at = tl.load(A + (k0 + rk0)[:, None] * sak + rm[None, :] * sam)
                bt = tl.load(B + rn[:, None] * sbn + (k0 + rk0)[None, :] * sbk)
            if MIXED:
                at = at.to(C.dtype.element_ty)
                bt = bt.to(C.dtype.element_ty)
            acc = _mm_dot(bt, at, acc, PREC, IS_F64)
        if MODE == 1:
            rk = K // BK * BK + rk0
            km = rk < K
            at = tl.load(
                A + rk[:, None] * sak + rm[None, :] * sam,
                mask=km[:, None] & (rm[None, :] < M),
                other=0.0,
            )
            bt = tl.load(
                B + rn[:, None] * sbn + rk[None, :] * sbk,
                mask=km[None, :] & (rn[:, None] < N),
                other=0.0,
            )
            if MIXED:
                at = at.to(C.dtype.element_ty)
                bt = bt.to(C.dtype.element_ty)
            acc = _mm_dot(bt, at, acc, PREC, IS_F64)
        out = tl.trans(acc.to(C.dtype.element_ty))
    else:
        if IS_F64:
            acc = tl.zeros((BM, BN), dtype=tl.float64)
        else:
            acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K // BK * BK if MODE == 1 else K, BK):
            if MODE == 0:
                rk = k0 + rk0
                km = rk < K
                a = tl.load(
                    A + rm[:, None] * sam + rk[None, :] * sak,
                    mask=km[None, :] & (rm[:, None] < M),
                    other=0.0,
                )
                b = tl.load(
                    B + rk[:, None] * sbk + rn[None, :] * sbn,
                    mask=km[:, None] & (rn[None, :] < N),
                    other=0.0,
                )
            else:
                a = tl.load(A + rm[:, None] * sam + (k0 + rk0)[None, :] * sak)
                b = tl.load(B + (k0 + rk0)[:, None] * sbk + rn[None, :] * sbn)
            if MIXED:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc = _mm_dot(a, b, acc, PREC, IS_F64)
        if MODE == 1:
            rk = K // BK * BK + rk0
            km = rk < K
            a = tl.load(
                A + rm[:, None] * sam + rk[None, :] * sak,
                mask=km[None, :] & (rm[:, None] < M),
                other=0.0,
            )
            b = tl.load(
                B + rk[:, None] * sbk + rn[None, :] * sbn,
                mask=km[:, None] & (rn[None, :] < N),
                other=0.0,
            )
            if MIXED:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc = _mm_dot(a, b, acc, PREC, IS_F64)
        out = acc.to(C.dtype.element_ty)
    cp = C + rm[:, None] * scm + rn[None, :] * scn
    if MODE == 0:
        tl.store(cp, out, mask=(rm < M)[:, None] & (rn < N)[None, :])
    else:
        tl.store(cp, out)


@triton.jit
def _mm_kernel_dyn(
    A,
    B,
    C,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    PREC: tl.constexpr,
    MODE: tl.constexpr,
    TN: tl.constexpr,
    IS_F64: tl.constexpr,
    MIXED: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GM * grid_n
    gid = pid // width
    gsize = tl.minimum(grid_m - gid * GM, GM)
    pid_m = gid * GM + pid % gsize
    pid_n = pid % width // gsize
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk0 = tl.arange(0, BK)
    if TN:
        if IS_F64:
            acc = tl.zeros((BN, BM), dtype=tl.float64)
        else:
            acc = tl.zeros((BN, BM), dtype=tl.float32)
        for k0 in range(0, K // BK * BK if MODE == 1 else K, BK):
            if MODE == 0:
                rk = k0 + rk0
                km = rk < K
                at = tl.load(
                    A + rk[:, None] * sak + rm[None, :] * sam,
                    mask=km[:, None] & (rm[None, :] < M),
                    other=0.0,
                )
                bt = tl.load(
                    B + rn[:, None] * sbn + rk[None, :] * sbk,
                    mask=km[None, :] & (rn[:, None] < N),
                    other=0.0,
                )
            else:
                at = tl.load(A + (k0 + rk0)[:, None] * sak + rm[None, :] * sam)
                bt = tl.load(B + rn[:, None] * sbn + (k0 + rk0)[None, :] * sbk)
            if MIXED:
                at = at.to(C.dtype.element_ty)
                bt = bt.to(C.dtype.element_ty)
            acc = _mm_dot(bt, at, acc, PREC, IS_F64)
        if MODE == 1:
            rk = K // BK * BK + rk0
            km = rk < K
            at = tl.load(
                A + rk[:, None] * sak + rm[None, :] * sam,
                mask=km[:, None] & (rm[None, :] < M),
                other=0.0,
            )
            bt = tl.load(
                B + rn[:, None] * sbn + rk[None, :] * sbk,
                mask=km[None, :] & (rn[:, None] < N),
                other=0.0,
            )
            if MIXED:
                at = at.to(C.dtype.element_ty)
                bt = bt.to(C.dtype.element_ty)
            acc = _mm_dot(bt, at, acc, PREC, IS_F64)
        out = tl.trans(acc.to(C.dtype.element_ty))
    else:
        if IS_F64:
            acc = tl.zeros((BM, BN), dtype=tl.float64)
        else:
            acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K // BK * BK if MODE == 1 else K, BK):
            if MODE == 0:
                rk = k0 + rk0
                km = rk < K
                a = tl.load(
                    A + rm[:, None] * sam + rk[None, :] * sak,
                    mask=km[None, :] & (rm[:, None] < M),
                    other=0.0,
                )
                b = tl.load(
                    B + rk[:, None] * sbk + rn[None, :] * sbn,
                    mask=km[:, None] & (rn[None, :] < N),
                    other=0.0,
                )
            else:
                a = tl.load(A + rm[:, None] * sam + (k0 + rk0)[None, :] * sak)
                b = tl.load(B + (k0 + rk0)[:, None] * sbk + rn[None, :] * sbn)
            if MIXED:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc = _mm_dot(a, b, acc, PREC, IS_F64)
        if MODE == 1:
            rk = K // BK * BK + rk0
            km = rk < K
            a = tl.load(
                A + rm[:, None] * sam + rk[None, :] * sak,
                mask=km[None, :] & (rm[:, None] < M),
                other=0.0,
            )
            b = tl.load(
                B + rk[:, None] * sbk + rn[None, :] * sbn,
                mask=km[:, None] & (rn[None, :] < N),
                other=0.0,
            )
            if MIXED:
                a = a.to(C.dtype.element_ty)
                b = b.to(C.dtype.element_ty)
            acc = _mm_dot(a, b, acc, PREC, IS_F64)
        out = acc.to(C.dtype.element_ty)
    cp = C + rm[:, None] * scm + rn[None, :] * scn
    if MODE == 0:
        tl.store(cp, out, mask=(rm < M)[:, None] & (rn < N)[None, :])
    else:
        tl.store(cp, out)


_TABLE_16 = (
    (32, 64, 64, 8, 4, 4, 1),
    (128, 128, 64, 8, 8, 3, 1),
    (128, 128, 32, 8, 4, 2, 1),
    (128, 256, 32, 8, 8, 2, 1),
)
_TABLE_BF16 = _TABLE_16
_TABLE_32 = (
    (32, 64, 32, 8, 4, 3, 0),
    (128, 128, 32, 8, 8, 3, 0),
    (128, 128, 32, 8, 8, 3, 0),
    (128, 128, 32, 8, 8, 3, 0),
)
_TABLE_64 = (
    (32, 32, 16, 8, 4, 2, 0),
    (32, 32, 16, 8, 4, 2, 0),
    (32, 32, 16, 8, 4, 2, 0),
    (32, 32, 16, 8, 4, 2, 0),
)
_TF32_MIN_K = 384
_DYN_BUCKET = 3


def _bucket(M, N):
    mx = M if M > N else N
    if mx <= 512:
        return 0
    if mx <= 1024:
        return 1
    if mx <= 2048:
        return 2
    return 3


def _aliased(a, b):
    try:
        return a.data_ptr() == b.data_ptr()
    except Exception:
        return True


def _select(dtype, M, N, K, aliased):
    b = _bucket(M, N)
    if dtype == torch.float64:
        return _TABLE_64[b] + (0, True)
    if dtype == torch.float32:
        prec = 2 if K >= _TF32_MIN_K and (not aliased) else 1
        return _TABLE_32[b] + (prec, False)
    if dtype == torch.bfloat16:
        return _TABLE_BF16[b] + (0, False)
    return _TABLE_16[b] + (0, False)


def tuned_mm(a, b):
    (M, K) = a.shape
    N = b.shape[1]
    if a.dtype == b.dtype:
        out_dtype = a.dtype
    else:
        out_dtype = torch.promote_types(a.dtype, b.dtype)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return c
    if K == 0:
        return torch.zeros((M, N), device=a.device, dtype=out_dtype)
    bucket = _bucket(M, N)
    (bm, bn, bk, gm, nw, ns, tn, prec, is_f64) = _select(
        out_dtype, M, N, K, _aliased(a, b)
    )
    sak = a.stride(1)
    sbk = b.stride(0)
    sbn = b.stride(1)
    if M % bm == 0 and N % bn == 0:
        mode = 2 if K % bk == 0 else 1
    else:
        mode = 0
    mixed = a.dtype != b.dtype
    grid = ((M + bm - 1) // bm * ((N + bn - 1) // bn),)
    if bucket == _DYN_BUCKET:
        _mm_kernel_dyn[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            sak,
            sbk,
            sbn,
            c.stride(0),
            c.stride(1),
            BM=bm,
            BN=bn,
            BK=bk,
            GM=gm,
            PREC=prec,
            MODE=mode,
            TN=tn,
            IS_F64=is_f64,
            MIXED=mixed,
            num_warps=nw,
            num_stages=ns,
        )
    else:
        _mm_kernel[grid](
            a,
            b,
            c,
            M=M,
            N=N,
            K=K,
            sam=a.stride(0),
            sak=sak,
            sbk=sbk,
            sbn=sbn,
            scm=c.stride(0),
            scn=c.stride(1),
            BM=bm,
            BN=bn,
            BK=bk,
            GM=gm,
            PREC=prec,
            MODE=mode,
            TN=tn,
            IS_F64=is_f64,
            MIXED=mixed,
            num_warps=nw,
            num_stages=ns,
        )
    return c
