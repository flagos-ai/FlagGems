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

"""Portable per-thread INT8 Q/K quantization; BF16 V remains unquantized.

This file contains real Triton kernels and never forwards to SageAttention.
Scope: BF16, D128, dense noncausal equal-head attention.
The original Sage implementation is only a validation reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _k_mean_partial(
    X,
    T,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    XZ: tl.constexpr,
    XN: tl.constexpr,
    XH: tl.constexpr,
    NB: tl.constexpr,
    BN: tl.constexpr,
):
    blk = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    n = blk * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    x = tl.load(
        X + b * XZ + n[:, None] * XN + h * XH + d[None, :], n[:, None] < NK, 0
    ).to(tl.float32)
    part = tl.sum(x, axis=0)
    tl.store(T + (bh * NB + blk) * D + d, part)


@triton.jit
def _k_mean_finish(
    T,
    M,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NB: tl.constexpr,
    RB: tl.constexpr,
):
    bh = tl.program_id(0)
    n = tl.arange(0, RB)
    d = tl.arange(0, D)
    x = tl.load(T + (bh * NB + n[:, None]) * D + d[None, :], n[:, None] < NB, 0)
    mean = tl.sum(x, axis=0) / NK
    tl.store(M + bh * D + d, mean)


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


def quantize_qk(query, key, value, *, tensor_layout="NHD"):
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("expected 4D tensors")
    if tensor_layout == "HND":
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    elif tensor_layout != "NHD":
        raise ValueError("tensor_layout must be NHD or HND")
    b, nq, h, d = query.shape
    nk = key.shape[1]
    if (
        d != 128
        or key.shape[0] != b
        or key.shape[2:] != (h, d)
        or value.shape != key.shape
    ):
        raise ValueError("requires D128, equal heads and matching K/V")
    if query.device.type not in ("cuda", "musa") or query.dtype != torch.bfloat16:
        raise ValueError("requires BF16 CUDA/MUSA input")
    if any(
        x.device != query.device or x.dtype != query.dtype or x.stride(-1) != 1
        for x in (query, key, value)
    ):
        raise ValueError("requires matching dtype/device and contiguous last dimension")
    if min(b, h, nq, nk) < 1:
        raise ValueError("empty dimensions unsupported")
    nqs = triton.cdiv(nq, 64) * 32
    nks = triton.cdiv(nk, 128) * 4
    qi = torch.empty((b, h, nq, d), device=query.device, dtype=torch.int8)
    ki = torch.empty((b, h, nk, d), device=query.device, dtype=torch.int8)
    qs = torch.empty((b, h, nqs), device=query.device, dtype=torch.float32)
    ks = torch.empty((b, h, nks), device=query.device, dtype=torch.float32)
    nb = triton.cdiv(nk, 128)
    partial = torch.empty((b, h, nb, d), device=query.device, dtype=torch.float32)
    mean = torch.empty((b, 1, h, d), device=query.device, dtype=query.dtype)
    _k_mean_partial[(nb, b * h)](
        key, partial, nk, h, d, *key.stride()[:3], nb, 128, num_warps=4
    )
    _k_mean_finish[(b * h,)](
        partial, mean, nk, h, d, nb, triton.next_power_of_2(nb), num_warps=4
    )
    _quant_q[(triton.cdiv(nq, 64), h, b)](
        query, qi, qs, nq, *query.stride()[:3], h, nq, nqs, d, num_warps=4
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
        num_warps=4,
    )
    return qi, qs, ki, ks, value, mean
