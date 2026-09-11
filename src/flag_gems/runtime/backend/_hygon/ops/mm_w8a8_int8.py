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

"""Hygon W8A8 GEMM with INT8 dot products and overflow-safe long-K reduction."""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit
def _prequant_vector(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    for row in tl.static_range(M):
        acc = tl.zeros((BN, BK), tl.int32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + rk
            a = tl.load(A + row * K + k, k < K, other=0).to(tl.int32)
            b = tl.load(
                B + cols[:, None].to(tl.int64) * K + k[None, :],
                (cols[:, None] < N) & (k[None, :] < K),
                other=0,
            ).to(tl.int32)
            acc += a[None, :] * b
        dot = tl.sum(acc, 1)
        sa = tl.load(SA + row)
        sb = tl.load(SB + cols, cols < N, other=0)
        tl.store(C + row * N + cols, dot.to(tl.float32) * sa * sb, cols < N)


@libentry()
@triton.jit
def _grouped_int8(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ST: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = tl.cdiv(M, BM)
    pn = tl.cdiv(N, BN)
    if GROUP == 0:
        im = pid % pm
        jn = pid // pm
    else:
        group = pid // (GROUP * pn)
        first = group * GROUP
        size = tl.minimum(pm - first, GROUP)
        im = first + pid % size
        jn = (pid % (GROUP * pn)) // size
    rm = im * BM + tl.arange(0, BM)
    rn = jn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tle.gpu.pipeline(0, tl.cdiv(K, BK), num_stages=ST):
        k = start * BK + rk
        a = tl.load(
            A + rm[:, None] * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + rn[None, :] * K + k[:, None],
            (rn[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(tl.float32) * sa[:, None] * sb[None, :],
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _split_mm(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CHUNK: tl.constexpr,
    ST: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    split = tl.program_id(2)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tle.gpu.pipeline(0, tl.cdiv(CHUNK, BK), num_stages=ST):
        k = split * CHUNK + start * BK + rk
        a = tl.load(
            A + r[:, None].to(tl.int64) * K + k[None, :],
            (r[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + c[None, :].to(tl.int64) * K + k[:, None],
            (c[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(
        P + split.to(tl.int64) * M * N + r[:, None].to(tl.int64) * N + c[None, :],
        acc,
        (r[:, None] < M) & (c[None, :] < N),
    )


@libentry()
@triton.jit
def _split_reduce(
    P,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.int64)
    for s in range(SPLITS):
        acc += tl.load(
            P + s.to(tl.int64) * M * N + i.to(tl.int64), i < M * N, other=0
        ).to(tl.int64)
    sa = tl.load(SA + i // N, i < M * N, other=0)
    sb = tl.load(SB + i % N, i < M * N, other=0)
    tl.store(C + i, acc.to(tl.float32) * sa * sb, i < M * N)


@triton.jit
def _quantize_half_codes(x, peak, FP16: tl.constexpr):
    # For FP16/BF16, a non-tie normalized value is separated from a rounding
    # boundary by at least 1/(254*2047) relative error, larger than four FP32
    # unit roundoffs. Advancing the shared multiplier one ULP preserves these
    # codes and rounds the exact +/-63.5 tie to +/-64. Every other half-integer
    # tie requires factor 127 in the peak significand: retain correctly rounded
    # division for those rows and for the synthetic 1e-10 peak floor.
    bits = peak.to(tl.int32, bitcast=True)
    if FP16:
        mant = ((bits >> 13) & 1023) | 1024
    else:
        mant = ((bits >> 16) & 127) | 128
    fallback = (peak == 1e-10) | (mant % 127 == 0)
    if len(peak.shape) > 0:
        fallback = tl.sum(fallback.to(tl.int32)) > 0
    if fallback:
        q = libdevice.rint(tl.div_rn(x, peak) * 127.0).to(tl.int8)
    else:
        multiplier = tl.div_rn(127.0, peak)
        multiplier = (multiplier.to(tl.int32, bitcast=True) + 1).to(
            tl.float32, bitcast=True
        )
        q = libdevice.rint(x * multiplier).to(tl.int8)
    return q


@libentry()
@triton.jit
def _quantize_half_rows(
    A,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, BLOCK)
    a = tl.load(
        A + rows[:, None] * AS0 + k[None, :] * AS1,
        (rows[:, None] < M) & (k[None, :] < K),
        other=0,
    ).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(a), 1), 1e-10)
    q = _quantize_half_codes(a, peak[:, None], A.dtype.element_ty == tl.float16)
    tl.store(
        Q + rows[:, None] * K + k[None, :], q, (rows[:, None] < M) & (k[None, :] < K)
    )
    tl.store(S + rows, peak * (1.0 / 127.0), rows < M)


@libentry()
@triton.jit
def _mv_peak_parts(
    A,
    Peaks,
    K: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) // PARTS
    part = tl.program_id(0) % PARTS
    k = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + row.to(tl.int64) * AS0 + k.to(tl.int64) * AS1, k < K, other=0).to(
        tl.float32
    )
    peak = tl.max(tl.abs(a), 0)
    tl.store(Peaks + row * PARTS + part, peak)


@libentry()
@triton.jit
def _mv_quantized_parts(
    A,
    B,
    Peaks,
    Partial,
    K: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS: tl.constexpr,
    PEAK_PARTS: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) // PARTS
    part = tl.program_id(0) % PARTS
    ps = tl.arange(0, triton.next_power_of_2(PEAK_PARTS))
    peaks = tl.load(Peaks + row * PEAK_PARTS + ps, ps < PEAK_PARTS, other=0)
    peak = tl.maximum(tl.max(peaks, 0), 1e-10)
    k = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + row.to(tl.int64) * AS0 + k.to(tl.int64) * AS1, k < K, other=0).to(
        tl.float32
    )
    aq = _quantize_half_codes(a, peak, A.dtype.element_ty == tl.float16)
    b = tl.load(B + k.to(tl.int64) * BS, k < K, other=0)
    acc = tl.sum(aq.to(tl.int32) * b.to(tl.int32), 0)
    tl.store(Partial + row * PARTS + part, acc)


@libentry()
@triton.jit
def _mv_reduce(
    Peaks,
    Partial,
    SB,
    Out,
    M: tl.constexpr,
    PEAK_PARTS: tl.constexpr,
    PARTS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ps = tl.arange(0, triton.next_power_of_2(PEAK_PARTS))
    peak = tl.load(
        Peaks + rows[:, None] * PEAK_PARTS + ps[None, :],
        (rows[:, None] < M) & (ps[None, :] < PEAK_PARTS),
        other=0,
    )
    peak = tl.maximum(tl.max(peak, 1), 1e-10)
    ss = tl.arange(0, triton.next_power_of_2(PARTS))
    acc = tl.load(
        Partial + rows[:, None] * PARTS + ss[None, :],
        (rows[:, None] < M) & (ss[None, :] < PARTS),
        other=0,
    ).to(tl.int64)
    acc = tl.sum(acc, 1)
    sb = tl.load(SB)
    tl.store(Out + rows, acc.to(tl.float32) * (peak * (1.0 / 127.0)) * sb, rows < M)


@libentry()
@triton.jit
def _quantize_rows(
    X, Q, S, K: tl.constexpr, SR: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * SR + k * SK, k < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
    q = libdevice.rint(tl.div_rn(x, peak) * 127.0).to(tl.int8)
    tl.store(Q + row * K + k, q, k < K)
    tl.store(S + row, peak * (1.0 / 127.0))


@libentry()
@triton.jit
def _gemm_tle(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ST: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tle.gpu.pipeline(0, tl.cdiv(K, BK), num_stages=ST):
        k = start * BK + rk
        a = tle.load(
            A + rm[:, None] * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
            is_async=False,
        )
        off = rn[None, :] * K + k[:, None]
        b = tle.load(
            B + off, (rn[None, :] < N) & (k[:, None] < K), other=0, is_async=False
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(tl.float32) * sa[:, None] * sb[None, :],
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _quantize(
    X, Q, S, K: tl.constexpr, SR: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    peak = tl.full((BLOCK,), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK)):
        k = start * BLOCK + c
        x = tl.load(X + row * SR + k * SK, k < K, other=0).to(tl.float32)
        peak = tl.maximum(peak, tl.abs(x))
    maximum = tl.maximum(tl.max(peak, 0), 1.0e-10)
    tl.store(S + row, maximum * (1.0 / 127.0))
    for start in range(tl.cdiv(K, BLOCK)):
        k = start * BLOCK + c
        x = tl.load(X + row * SR + k * SK, k < K, other=0).to(tl.float32)
        normalized = tl.div_rn(x, maximum) * 127.0
        lower = tl.floor(normalized)
        fraction = normalized - lower
        odd = (lower.to(tl.int32) & 1) != 0
        rounded = lower + tl.where(
            (fraction > 0.5) | ((fraction == 0.5) & odd), 1.0, 0.0
        )
        q = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
        tl.store(Q + row * K + k, q, k < K)


@libentry()
@triton.jit
def _gemm(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        a = tl.load(
            A + rm[:, None].to(tl.int64) * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + rn[None, :].to(tl.int64) * K + k[:, None],
            (rn[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + rm[:, None].to(tl.int64) * N + rn[None, :],
        out,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _zero(C, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(C + i, 0.0, i < SIZE)


def _validate(a, b, scale_b):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_b)):
        raise TypeError("A, B and scale_b must be tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in _FLOATS or b.dtype != torch.int8:
        raise TypeError("A must be FP16/BF16/FP32 and B must be prequantized INT8")
    if a.device.type != "cuda" or any(x.device != a.device for x in (b, scale_b)):
        raise ValueError("A, B and scale_b must be on the same Hygon device")
    n = b.shape[1]
    if scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected per-column B scales with shape [N] or [1,N]")
    if scale_b.dtype != torch.float32 or not scale_b.is_contiguous():
        raise ValueError("scale_b must be a contiguous FP32 tensor")
    return a.shape[0], n, a.shape[1]


def _run(a, b, scale_b, out, m, n, k):
    if m == 0 or n == 0:
        return out
    with torch_device_fn.device(a.device):
        if k == 0:
            _zero[(triton.cdiv(m * n, 1024),)](out, m * n, 1024)
            return out
        if n == 1 and m >= 64 and k >= 65536 and a.dtype != torch.float32:
            # Keep the full-row peak but fuse quantization with vector products.
            # Every INT32 partial covers at most 16384 products; reduce in INT64.
            peak_parts = triton.cdiv(k, 8192)
            dot_parts = triton.cdiv(k, 16384)
            peaks = torch.empty((m, peak_parts), device=a.device, dtype=torch.float32)
            partial = torch.empty((m, dot_parts), device=a.device, dtype=torch.int32)
            _mv_peak_parts[(m * peak_parts,)](
                a, peaks, k, *a.stride(), peak_parts, 8192, num_warps=8
            )
            _mv_quantized_parts[(m * dot_parts,)](
                a,
                b,
                peaks,
                partial,
                k,
                *a.stride(),
                b.stride(0),
                peak_parts,
                dot_parts,
                16384,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _mv_reduce[(triton.cdiv(m, 16),)](
                peaks,
                partial,
                scale_b,
                out,
                m,
                peak_parts,
                dot_parts,
                16,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return out
        aq = torch.empty((m, k), device=a.device, dtype=torch.int8)
        sa = torch.empty((m,), device=a.device, dtype=torch.float32)
        # Keep signed 32-bit address arithmetic only when all offsets fit.
        fast_a = (
            (k <= 4096 or (m <= 8 and k <= 32768))
            and m * k < 2**31
            and (m - 1) * a.stride(0) + (k - 1) * a.stride(1) < 2**31
        )
        if fast_a and (m >= 128 or (m <= 8 and k > 4096)) and a.dtype != torch.float32:
            rows = 1
            quant_warps = 2 if 1024 <= m < 4096 else 4
            _quantize_half_rows[(triton.cdiv(m, rows),)](
                a,
                aq,
                sa,
                m,
                k,
                *a.stride(),
                rows,
                triton.next_power_of_2(k),
                num_warps=quant_warps,
                enable_fp_fusion=False,
            )
        elif fast_a:
            _quantize_rows[(m,)](
                a,
                aq,
                sa,
                k,
                *a.stride(),
                triton.next_power_of_2(k),
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _quantize[(m,)](
                a, aq, sa, k, *a.stride(), 1024, num_warps=4, enable_fp_fusion=False
            )
        # The caller supplies INT8 weights. Packing arbitrary strides, when
        # necessary, stays inside the operator; no weight quantization occurs.
        bq = b.t().contiguous()
        _launch_prequantized(aq, bq, sa, scale_b, out, m, n, k)
    return out


def mm_w8a8_int8(a, b, scale_b, *, out_dtype=None):
    """Multiply dynamically quantized floating A by prequantized INT8 B.

    A is finite FP16/BF16/FP32 [M,K], B is INT8 [K,N], and scale_b is
    contiguous FP32 [N] or [1,N], all on one Hygon device. B represents
    symmetric quantized weights with zero point 0, including code -128.
    Arbitrary matrix strides are supported; column-major B avoids packing.

    Each call and Graph replay computes A's per-row peak=max(abs(A),1e-10),
    scale=peak/127 and round-to-nearest-even codes (A/peak)*127 on the GPU.
    Activation quantization kernels, INT8 GEMM, and dequantization are all
    part of this operator and must all be timed. Weight preparation is offline.
    Output defaults to BF16. Long K uses bounded INT32 partial products and
    INT64 reduction. Forward inference only; autograd is not implemented.
    """
    m, n, k = _validate(a, b, scale_b)
    dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if dtype not in _FLOATS:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    return _run(a, b, scale_b, out, m, n, k)


def mm_w8a8_int8_out(a, b, scale_b, *, out):
    """Dynamic A quantization and INT8 GEMM into caller-owned output.

    Same inputs and timing scope as mm_w8a8_int8. Output may alias A,
    whose quantization finishes before GEMM, but not B or scale_b.
    """
    m, n, k = _validate(a, b, scale_b)
    if not isinstance(out, torch.Tensor) or out.dtype not in _FLOATS:
        raise TypeError("out must be an FP16, BF16 or FP32 tensor")
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must have shape [M,N], same device and be contiguous")
    if out.numel() and any(
        x.numel() and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in (b, scale_b)
    ):
        raise ValueError("out must not alias prequantized weights or scale_b")
    return _run(a, b, scale_b, out, m, n, k)


def _launch_prequantized(aq, bq, sa, sb, out, m, n, k):
    if m == 0 or n == 0:
        return out
    if k == 0:
        _zero[(triton.cdiv(m * n, 1024),)](out, m * n, 1024)
        return out
    if m == 1 and 4096 < k <= 32768:
        _prequant_vector[(triton.cdiv(n, 2),)](
            aq, bq, sa, sb, out, m, n, k, 2, 8192, num_warps=4
        )
    elif m <= 8 and n > 1024 and 1024 <= k <= 4096 and n * k < 2**31:
        bm = 16 if n <= 16384 else 32
        _grouped_int8[(triton.cdiv(n, 64),)](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            bm,
            64,
            256,
            2,
            0,
            num_warps=4,
            num_stages=2,
        )
    elif 2 <= m <= 8 and n >= 1024 and 4096 < k <= 32768 and n * k < 2**31:
        _grouped_int8[(triton.cdiv(n, 32),)](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            32,
            32,
            256,
            2,
            0,
            num_warps=4,
            num_stages=2,
        )
    elif 3 <= m <= 8 and 512 <= n <= 1024 and 1024 <= k <= 4096:
        _grouped_int8[(triton.cdiv(n, 16),)](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            16,
            16,
            256,
            2,
            0,
            num_warps=2,
            num_stages=2,
        )
    elif m <= 8 and n <= 1024 and 1024 <= k <= 4096:
        bn = 1 if m >= 3 else 2
        _prequant_vector[(triton.cdiv(n, bn),)](
            aq, bq, sa, sb, out, m, n, k, bn, 4096, num_warps=4
        )
    elif k > 131071 or (k >= 65536 and m >= 1024 and n >= 1024):
        # Even -128 * -128 cannot overflow a chunk's INT32 accumulator.
        # INT64 reduction preserves the integer result across long K. Splitting
        # also exposes more parallel work for large tiles with few output blocks.
        chunk = 32768
        parts = triton.cdiv(k, chunk)
        partial = torch.empty((parts, m, n), device=aq.device, dtype=torch.int32)
        bm, bn, nw = (256, 256, 8) if m >= 1024 and n >= 1024 else (64, 128, 4)
        _split_mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), parts)](
            aq, bq, partial, m, n, k, bm, bn, 128, chunk, 2, num_warps=nw, num_stages=2
        )
        _split_reduce[(triton.cdiv(m * n, 256),)](
            partial, sa, sb, out, m, n, parts, 256, num_warps=4
        )
    elif max(m * k, n * k, m * n) >= 2**31:
        _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            aq, bq, sa, sb, out, m, n, k, 32, 64, 64, num_warps=4, num_stages=1
        )
    elif (m >= 65 and n >= 1024 and k >= 1024 and (m >= 256 or n >= 2048)) or (
        m >= 8192 and 512 <= n < 1024 and k >= 1024
    ):
        if m >= 8192:
            if n < 1024:
                bm, bn, bk, nw, st, group = 256, 256, 128, 16, 2, 1
            elif n >= 8192:
                bm, bn, bk, nw, st, group = 256, 512, 64, 16, 2, 8
            else:
                bm, bn, bk, nw, st, group = 256, 256, 128, 8, 2, 8
        elif m >= 4096:
            bm, bn, bk, nw, st, group = 128, 128, 128, 8, 2, 8
        elif m >= 2048 and n >= 2048:
            bm, bn, bk, nw, st, group = 256, 256, 128, 16, 2, 0
        elif m == 256:
            if n <= 1024:
                bm, bn, bk, nw, st, group = 32, 64, 128, 4, 2, 0
            elif 16384 <= n <= 32768 and k < 4096:
                bm, bn, bk, nw, st, group = 256, 256, 128, 16, 2, 0
            else:
                bm, bn, bk, nw, st, group = 128, 128, 128, 4, 2, 0
        elif m < 256:
            if k > 4096 or n <= 8192:
                bm, bn, bk, nw, st, group = 64, 64, 128, 4, 2, 0
            else:
                bm, bn, bk, nw, st, group = 128, 128, 256, 4, 1, 0
        else:
            bm, bn, bk, nw, st, group = 64, 128, 128, 4, 1 if m >= 2048 else 2, 0
        _grouped_int8[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            st,
            group,
            num_warps=nw,
            num_stages=st,
        )
    else:
        bm, bn = (64, 128) if m >= 1024 else (32, 64)
        stages = 1 if m >= 2048 else 2
        _gemm_tle[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            bm,
            bn,
            128,
            stages,
            num_warps=4,
            num_stages=stages,
        )
    return out


def _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, *, out):
    """Private GEMM primitive; excludes activation quantization, not an API benchmark."""
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b, out)):
        raise TypeError("expected tensor inputs, scales and output")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    m, k = a.shape
    n = b.shape[1]
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("prequantized inputs must be INT8")
    if a.device.type != "cuda" or any(
        x.device != a.device for x in (b, scale_a, scale_b, out)
    ):
        raise ValueError("all tensors must be on the same Hygon device")
    if scale_a.shape not in ((m,), (m, 1)) or scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected per-row A scales and per-column B scales")
    if any(
        x.dtype != torch.float32 or not x.is_contiguous() for x in (scale_a, scale_b)
    ):
        raise ValueError("scales must be contiguous FP32 tensors")
    if out.shape != (m, n) or out.dtype not in _FLOATS or not out.is_contiguous():
        raise ValueError("out must be contiguous FP16/BF16/FP32 [M,N]")
    if out.numel() and any(
        x.numel() and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in (a, b, scale_a, scale_b)
    ):
        raise ValueError("prequantized output must not alias inputs or scales")
    with torch_device_fn.device(a.device):
        # Callers can supply these layouts directly. Other strides pay for
        # normalization inside this private primitive.
        aq = a.contiguous()
        bq = b.t().contiguous()
        return _launch_prequantized(aq, bq, scale_a, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    if out_dtype not in _FLOATS:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.ndim != 2
        or b.ndim != 2
    ):
        raise ValueError("expected matrix inputs")
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
    return _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, out=out)
