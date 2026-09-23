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
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

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
    Bias,
    HAS_BIAS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
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
        sa = tl.load(SA + row * (not A_SCALAR))
        sb = tl.load(SB + cols * (not B_SCALAR), cols < N, other=0)
        bias_value = 0.0
        if HAS_BIAS:
            bias_value = tl.load(Bias + cols, cols < N, other=0).to(tl.float32)
        tl.store(
            C + row * N + cols, (dot.to(tl.float32) * sa * sb) + bias_value, cols < N
        )


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
    Bias,
    HAS_BIAS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
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
    for start in tl.range(0, tl.cdiv(K, BK), num_stages=ST):
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
    sa = tl.load(SA + rm * (not A_SCALAR), rm < M, other=0)
    sb = tl.load(SB + rn * (not B_SCALAR), rn < N, other=0)
    bias_value = 0.0
    if HAS_BIAS:
        bias_value = tl.load(Bias + rn[None, :], rn[None, :] < N, other=0).to(
            tl.float32
        )
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        (acc.to(tl.float32) * sa[:, None] * sb[None, :]) + bias_value,
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
    for start in tl.range(0, tl.cdiv(CHUNK, BK), num_stages=ST):
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
    Bias,
    HAS_BIAS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.int64)
    for s in range(SPLITS):
        acc += tl.load(
            P + s.to(tl.int64) * M * N + i.to(tl.int64), i < M * N, other=0
        ).to(tl.int64)
    sa = tl.load(SA + (i // N) * (not A_SCALAR), i < M * N, other=0)
    sb = tl.load(SB + (i % N) * (not B_SCALAR), i < M * N, other=0)
    bias_value = 0.0
    if HAS_BIAS:
        bias_value = tl.load(Bias + i % N, i < M * N, other=0).to(tl.float32)
    tl.store(C + i, (acc.to(tl.float32) * sa * sb) + bias_value, i < M * N)


@libentry()
@triton.jit
def _gemm_tl_range(
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
    Bias,
    HAS_BIAS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tl.range(0, tl.cdiv(K, BK), num_stages=ST):
        k = start * BK + rk
        a = tle.load(
            A + rm[:, None] * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
            is_async=False,
        )
        off = rn[None, :] * K + k[:, None]
        b = tle.load(
            B + off,
            (rn[None, :] < N) & (k[:, None] < K),
            other=0,
            is_async=False,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm * (not A_SCALAR), rm < M, other=0)
    sb = tl.load(SB + rn * (not B_SCALAR), rn < N, other=0)
    bias_value = 0.0
    if HAS_BIAS:
        bias_value = tl.load(Bias + rn[None, :], rn[None, :] < N, other=0).to(
            tl.float32
        )
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        (acc.to(tl.float32) * sa[:, None] * sb[None, :]) + bias_value,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


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
    Bias,
    HAS_BIAS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
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
    sa = tl.load(SA + rm * (not A_SCALAR), rm < M, other=0)
    sb = tl.load(SB + rn * (not B_SCALAR), rn < N, other=0)
    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    bias_value = 0.0
    if HAS_BIAS:
        bias_value = tl.load(Bias + rn[None, :], rn[None, :] < N, other=0).to(
            tl.float32
        )
    tl.store(
        C + rm[:, None].to(tl.int64) * N + rn[None, :],
        (out) + bias_value,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _zero(
    C,
    SIZE: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    Bias,
    HAS_BIAS: tl.constexpr,
    A_SCALAR: tl.constexpr,
    B_SCALAR: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bias_value = 0.0
    if HAS_BIAS:
        bias_value = tl.load(Bias + i % N, i < SIZE, other=0).to(tl.float32)
    tl.store(C + i, (0.0) + bias_value, i < SIZE)


@libentry()
@triton.jit
def _mv_parts(A, B, P, M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    k = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + row.to(tl.int64) * K + k, k < K, other=0).to(tl.int32)
    b = tl.load(B + k, k < K, other=0).to(tl.int32)
    acc = tl.sum(a * b, 0)
    tl.store(P + part * M + row, acc)


def _launch_prequantized(aq, bq, sa, sb, out, m, n, k, bias):
    epilogue = dict(
        Bias=bias,
        HAS_BIAS=bias is not None,
        A_SCALAR=sa.numel() == 1,
        B_SCALAR=sb.numel() == 1,
        enable_fp_fusion=False,
    )
    if m == 0 or n == 0:
        return out
    if k == 0:
        _zero[(triton.cdiv(m * n, 1024),)](out, m * n, n, 1024, **epilogue)
        return out
    if n == 1 and k >= 65536:
        parts = triton.cdiv(k, 16384)
        partial = torch.empty((parts, m), device=aq.device, dtype=torch.int32)
        _mv_parts[(m, parts)](aq, bq, partial, m, k, 16384, num_warps=4)
        _split_reduce[(triton.cdiv(m, 256),)](
            partial, sa, sb, out, m, n, parts, 256, num_warps=4, **epilogue
        )
    elif m == 1 and 4096 < k <= 32768:
        _prequant_vector[(triton.cdiv(n, 2),)](
            aq, bq, sa, sb, out, m, n, k, 2, 8192, num_warps=4, **epilogue
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
            **epilogue,
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
            **epilogue,
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
            **epilogue,
        )
    elif m <= 8 and n <= 1024 and 1024 <= k <= 4096:
        bn = 1 if m >= 3 else 2
        _prequant_vector[(triton.cdiv(n, bn),)](
            aq, bq, sa, sb, out, m, n, k, bn, 4096, num_warps=4, **epilogue
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
            partial, sa, sb, out, m, n, parts, 256, num_warps=4, **epilogue
        )
    elif max(m * k, n * k, m * n) >= 2**31:
        _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            32,
            64,
            64,
            num_warps=4,
            num_stages=1,
            **epilogue,
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
            **epilogue,
        )
    else:
        bm, bn = (64, 128) if m >= 1024 else (32, 64)
        stages = 1 if m >= 2048 else 2
        _gemm_tl_range[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
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
            **epilogue,
        )
    return out


def _validate(a, b, scale_a, scale_b, bias):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b)):
        raise TypeError("A, B and scales must be tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized INT8")
    if a.device.type != "cuda" or any(
        x.device != a.device for x in (b, scale_a, scale_b)
    ):
        raise ValueError("inputs and scales must be on the same Hygon device")
    m, k = a.shape
    n = b.shape[1]
    for scale, shapes in [(scale_a, ((m,), (m, 1))), (scale_b, ((n,), (1, n)))]:
        if scale.numel() != 1 and scale.shape not in shapes:
            raise ValueError("expected scalar, per-row A or per-column B scales")
        if scale.dtype != torch.float32 or not scale.is_contiguous():
            raise ValueError("scales must be contiguous FP32 tensors")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.dtype not in _FLOATS:
            raise TypeError("bias must be an FP16/BF16/FP32 tensor")
        if bias.shape != (n,) or bias.device != a.device or not bias.is_contiguous():
            raise ValueError("bias must be contiguous [N] on the input device")
    return m, n, k


def _run(a, b, scale_a, scale_b, out, m, n, k, bias):
    with torch_device_fn.device(a.device):
        return _launch_prequantized(
            a.contiguous(), b.t().contiguous(), scale_a, scale_b, out, m, n, k, bias
        )


def mm_w8a8_int8(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
    """Compute (A @ B) * scale_a * scale_b + bias from prequantized INT8.

    A has shape [M,K], B [K,N], with symmetric zero-point-zero INT8 codes.
    FP32 scales are scalar or per-row A ([M], [M,1]) / per-column B
    ([N], [1,N]). Optional bias is contiguous [N]. All tensors must be on
    one Hygon device. Bias is added in FP32 before conversion to out_dtype.

    Quantization is the caller's responsibility and is excluded from timing.
    Arbitrary matrix strides are supported; contiguous A and column-major B
    avoid packing. Long K uses bounded INT32 partials and INT64 reduction.
    Forward inference only; autograd is not implemented.
    """
    m, n, k = _validate(a, b, scale_a, scale_b, bias)
    if out_dtype not in _FLOATS:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run(a, b, scale_a, scale_b, out, m, n, k, bias)


def mm_w8a8_int8_out(a, b, scale_a, scale_b, *, out, bias=None):
    """Prequantized INT8 GEMM into contiguous output; inputs must not alias out."""
    m, n, k = _validate(a, b, scale_a, scale_b, bias)
    if not isinstance(out, torch.Tensor) or out.dtype not in _FLOATS:
        raise TypeError("out must be an FP16/BF16/FP32 tensor")
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.numel() and any(
        x is not None
        and x.numel()
        and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in (a, b, scale_a, scale_b, bias)
    ):
        raise ValueError("out must not alias inputs, scales or bias")
    return _run(a, b, scale_a, scale_b, out, m, n, k, bias)
