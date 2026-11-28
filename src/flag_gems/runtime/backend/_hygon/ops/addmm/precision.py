# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Accurate split reductions for FP32 addmm with a long vector dimension."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to


@triton.jit
def _precise_vector(
    A,
    B,
    I,
    O,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    IM: tl.constexpr,
    IN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BO: tl.constexpr,
    RK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    o = tl.program_id(0) * BO + tl.arange(0, BO)
    m = o // N
    n = o % N
    k = tl.program_id(1) * RK + tl.arange(0, RK)
    dtype: tl.constexpr = tl.float64
    a = tl.load(
        A + m[:, None] * AM + k[None, :] * AK,
        (m[:, None] < M) & (k[None, :] < K),
        other=0,
    ).to(dtype)
    b = tl.load(
        B + n[:, None] * BN + k[None, :] * BK,
        (o[:, None] < M * N) & (k[None, :] < K),
        other=0,
    ).to(dtype)
    v = tl.sum(a * b, 1)
    if SPLIT:
        tl.store(P + tl.program_id(1) * M * N + o, v, o < M * N)
    else:
        if BETA != 0:
            bias = tl.load(I + m * IM + n * IN, o < M * N, other=0).to(dtype)
            v = v * ALPHA + bias * BETA
        else:
            v = v * ALPHA
        tl.store(O + m * OM + n * ON, v, o < M * N)


@triton.jit
def _precise_vector_finish(
    P,
    I,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    IM: tl.constexpr,
    IN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    o = tl.program_id(0)
    s = tl.arange(0, BLOCK)
    v = tl.load(P + s * M * N + o, s < S, other=0)
    value = tl.sum(v, 0) * ALPHA
    if BETA != 0:
        value += tl.load(I + (o // N) * IM + (o % N) * IN).to(tl.float64) * BETA
    tl.store(O + (o // N) * OM + (o % N) * ON, value)


def use_precise_vector(a, b):
    m, k = a.shape
    n = b.shape[1]
    # Bound both the number of reductions and the temporary split buffer.
    return (
        a.dtype == b.dtype == torch.float32
        and min(m, n) == 1
        and max(m, n) <= 512
        and k >= 4096
    )


def launch_precise_vector(bias, a, b, out, alpha, beta):
    m, k = a.shape
    n = b.shape[1]
    assert k == b.shape[0], "Incompatible dimensions"
    assert broadcastable_to(bias.shape, (m, n)), "Incompatible input shape"
    bias_view = bias.broadcast_to((m, n))
    block_k = 4096
    parts = triton.cdiv(k, block_k)
    block_out = 1 if m == 1 else 4
    with torch_device_fn.device(a.device):
        # FP32 products are exact in FP64. Keep both local and cross-block
        # sums in FP64, then round once after alpha/beta and the bias.
        partial = (
            torch.empty((parts, m * n), dtype=torch.float64, device=a.device)
            if parts > 1
            else out
        )
        _precise_vector[(triton.cdiv(m * n, block_out), parts)](
            a,
            b,
            bias,
            out,
            partial,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            *bias_view.stride(),
            *out.stride(),
            alpha,
            beta,
            block_out,
            block_k,
            parts > 1,
            num_warps=8 if m == 1 else 4,
            enable_fp_fusion=False,
        )
        if parts > 1:
            _precise_vector_finish[(m * n,)](
                partial,
                bias,
                out,
                m,
                n,
                parts,
                *bias_view.stride(),
                *out.stride(),
                alpha,
                beta,
                triton.next_power_of_2(parts),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out
