# Copyright 2026 FlagOS Contributors
# Quantization mapping adapted from SageAttention (c)2024 SageAttention team.
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

"""Hopper SageAttention2: INT8 QK and E4M3 PV, per-thread scales.

This file contains real Triton kernels and never forwards to SageAttention.
Scope is explicit: SM90, FP16/BF16, D128, dense noncausal equal-head attention.
The original Sage implementation is only a validation reference.
"""

import torch
import triton
import triton.language as tl

from .native_mean import native_k_mean


@triton.jit
def _quant_q(
    X,
    Y,
    S,
    N: tl.constexpr,
    XZ: tl.constexpr,
    XN: tl.constexpr,
    XH: tl.constexpr,
    H: tl.constexpr,
    NQ: tl.constexpr,
    NS: tl.constexpr,
    D: tl.constexpr,
):
    block = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    group = tl.arange(0, 32)
    i = tl.arange(0, 2 * D)
    n = (
        block * 64
        + group[:, None] // 8 * 16
        + (i[None, :] // D) * 8
        + group[:, None] % 8
    )
    d = i % D
    x = tl.load(X + b * XZ + n * XN + h * XH + d[None, :], n < NQ, other=0).to(
        tl.float32
    )
    scale = tl.max(tl.abs(x), axis=1) / 127.0 + 0.0000001
    q = x / scale[:, None]
    q = q + 0.5 * tl.where(q >= 0, 1.0, -1.0)
    tl.store(Y + ((b * H + h) * NQ + n) * D + d[None, :], q.to(tl.int8), n < NQ)
    tl.store(S + (b * H + h) * NS + block * 32 + group, scale)


@triton.jit
def _quant_k(
    X,
    KM,
    Y,
    S,
    XZ: tl.constexpr,
    XN: tl.constexpr,
    XH: tl.constexpr,
    MZ: tl.constexpr,
    MH: tl.constexpr,
    H: tl.constexpr,
    NK: tl.constexpr,
    NS: tl.constexpr,
    D: tl.constexpr,
):
    block = tl.program_id(0) // 4
    sub = tl.program_id(0) % 4
    h = tl.program_id(1)
    b = tl.program_id(2)
    n0 = block * 128 + tl.arange(0, 16) * 8 + sub * 2
    n1 = n0 + 1
    d = tl.arange(0, D)
    mean = tl.load(KM + b * MZ + h * MH + d)
    x0 = tl.load(
        X + b * XZ + n0[:, None] * XN + h * XH + d[None, :], n0[:, None] < NK, other=0
    )
    x1 = tl.load(
        X + b * XZ + n1[:, None] * XN + h * XH + d[None, :], n1[:, None] < NK, other=0
    )
    # Native centers in the original dtype BEFORE the FP32 quantizer.
    x0 = tl.where(
        n0[:, None] < NK,
        (x0.to(tl.float32) - mean[None, :].to(tl.float32))
        .to(X.dtype.element_ty)
        .to(tl.float32),
        0.0,
    )
    x1 = tl.where(
        n1[:, None] < NK,
        (x1.to(tl.float32) - mean[None, :].to(tl.float32))
        .to(X.dtype.element_ty)
        .to(tl.float32),
        0.0,
    )
    scale = tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127.0 + 0.0000001
    q0 = x0 / scale
    q1 = x1 / scale
    q0 = q0 + 0.5 * tl.where(q0 >= 0, 1.0, -1.0)
    q1 = q1 + 0.5 * tl.where(q1 >= 0, 1.0, -1.0)
    tl.store(
        Y + ((b * H + h) * NK + n0[:, None]) * D + d[None, :],
        q0.to(tl.int8),
        n0[:, None] < NK,
    )
    tl.store(
        Y + ((b * H + h) * NK + n1[:, None]) * D + d[None, :],
        q1.to(tl.int8),
        n1[:, None] < NK,
    )
    tl.store(S + (b * H + h) * NS + block * 4 + sub, scale)


@triton.jit
def _v_partial(
    X,
    T,
    XZ: tl.constexpr,
    XN: tl.constexpr,
    XH: tl.constexpr,
    H: tl.constexpr,
    NK: tl.constexpr,
    NB: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    blk = tl.program_id(0)
    h = tl.program_id(2) % H
    b = tl.program_id(2) // H
    n = blk * BN + tl.arange(0, BN)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    x = tl.load(
        X + b * XZ + n[:, None] * XN + h * XH + d[None, :],
        (n[:, None] < NK) & (d[None, :] < D),
        other=0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=0)
    tl.store(T + ((b * H + h) * NB + blk) * D + d, maximum, d < D)


@triton.jit
def _v_scales(
    T,
    S,
    R,
    H: tl.constexpr,
    NB: tl.constexpr,
    D: tl.constexpr,
    RB: tl.constexpr,
    BD: tl.constexpr,
):
    bh = tl.program_id(1)
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    n = tl.arange(0, RB)
    x = tl.load(
        T + (bh * NB + n[:, None]) * D + d[None, :],
        (n[:, None] < NB) & (d[None, :] < D),
        other=0,
    )
    maximum = tl.max(x, axis=0)
    scale = tl.where(maximum > 0, maximum / 448.0, 1.0)
    inverse = tl.where(maximum > 0, 448.0 / maximum, 1.0)
    tl.store(S + bh * D + d, scale, d < D)
    tl.store(R + bh * D + d, inverse, d < D)


@triton.jit
def _quant_v(
    X,
    R,
    Y,
    XZ: tl.constexpr,
    XN: tl.constexpr,
    XH: tl.constexpr,
    H: tl.constexpr,
    NK: tl.constexpr,
    NP: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SAFE_STRIDED: tl.constexpr = False,
):
    h = tl.program_id(2) % H
    b = tl.program_id(2) // H
    physical = tl.program_id(0) * BN + tl.arange(0, BN)
    # Same within16 permutation as the native WGMMA FP8 operand.
    n = (
        physical // 16 * 16
        + (physical % 16) // 4 * 2
        + (physical % 4) // 2 * 8
        + physical % 2
    )
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    reciprocal = tl.load(R + (b * H + h) * D + d, d < D, other=1)
    if SAFE_STRIDED:
        # Load canonical token order before the FP8 operand permutation. The
        # permuted address expression over-vectorizes in pinned FlagTree for
        # row strides that are only 8-byte aligned (memcheck regression n=64).
        x = tl.load(
            X + b * XZ + physical[None, :] * XN + h * XH + d[:, None],
            (physical[None, :] < NK) & (d[:, None] < D),
            other=0,
        ).to(tl.float32)
        y = (x * reciprocal[:, None]).to(tl.float8e4nv)
        y = (
            y.reshape((BD, BN // 16, 2, 2, 2, 2))
            .permute((0, 1, 3, 4, 2, 5))
            .reshape((BD, BN))
        )
    else:
        x = tl.load(
            X + b * XZ + n[None, :] * XN + h * XH + d[:, None],
            (n[None, :] < NK) & (d[:, None] < D),
            other=0,
        ).to(tl.float32)
        y = (x * reciprocal[:, None]).to(tl.float8e4nv)
    tl.store(
        Y + ((b * H + h) * D + d[:, None]) * NP + physical[None, :],
        y,
        (physical[None, :] < NP) & (d[:, None] < D),
    )


def quantize_sage_sm90(query, key, value, *, tensor_layout="NHD"):
    """Internal quantization boundary exposed only for validation."""
    if tensor_layout == "HND":
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    elif tensor_layout != "NHD":
        raise ValueError("tensor_layout must be NHD or HND")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("expected4D inputs")
    b, nq, h, d = query.shape
    if (
        d != 128
        or key.shape[0] != b
        or key.shape[2:] != (h, d)
        or value.shape != key.shape
    ):
        raise ValueError("requiresD128, equal heads and matchingK/V")
    if any(
        x.device != query.device or x.dtype != query.dtype or x.stride(-1) != 1
        for x in (key, value)
    ):
        raise ValueError("requires matchingdtype/device and contiguouslastdimension")
    if (
        query.stride(-1) != 1
        or query.dtype not in (torch.float16, torch.bfloat16)
        or query.device.type != "cuda"
    ):
        raise ValueError("requiresCUDA FP16/BF16 and contiguouslastdimension")
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        raise NotImplementedError("experimentalSM90 only")
    nk = key.shape[1]
    if nq < 1 or nk < 1:
        raise ValueError("emptysequences unsupported")
    nqs = triton.cdiv(nq, 64) * 32
    nks = triton.cdiv(nk, 128) * 4
    npad = triton.cdiv(nk, 128) * 128
    qi = torch.empty((b, h, nq, d), device=query.device, dtype=torch.int8)
    ki = torch.empty((b, h, nk, d), device=query.device, dtype=torch.int8)
    qs = torch.empty((b, h, nqs), device=query.device, dtype=torch.float32)
    ks = torch.empty((b, h, nks), device=query.device, dtype=torch.float32)
    mean = native_k_mean(key)
    _quant_q[(triton.cdiv(nq, 64), h, b)](
        query, qi, qs, nq, *query.stride()[:3], h, nq, nqs, d
    )
    _quant_k[(triton.cdiv(nk, 128) * 4, h, b)](
        key,
        mean,
        ki,
        ks,
        *key.stride()[:3],
        mean.stride(0),
        mean.stride(2),
        h,
        nk,
        nks,
        d,
    )
    nb = triton.cdiv(nk, 256)
    partial = torch.empty((b, h, nb, d), device=query.device, dtype=torch.float32)
    vs = torch.empty((b, h, d), device=query.device, dtype=torch.float32)
    vr = torch.empty_like(vs)
    vf = torch.empty((b, h, d, npad), device=query.device, dtype=torch.float8_e4m3fn)
    _v_partial[(nb, triton.cdiv(d, 16), b * h)](
        value, partial, *value.stride()[:3], h, nk, nb, d, 256, 16
    )
    _v_scales[(triton.cdiv(d, 16), b * h)](
        partial, vs, vr, h, nb, d, triton.next_power_of_2(nb), 16
    )
    safe_strided = value.data_ptr() % 16 != 0 or any(
        (stride * value.element_size()) % 16 for stride in value.stride()[:3]
    )
    _quant_v[(triton.cdiv(npad, 256), triton.cdiv(d, 16), b * h)](
        value,
        vr,
        vf,
        *value.stride()[:3],
        h,
        nk,
        npad,
        d,
        256,
        16,
        SAFE_STRIDED=safe_strided,
    )
    return qi, qs, ki, ks, vf, vs, mean
