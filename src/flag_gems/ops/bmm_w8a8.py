# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton as tr
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@tr.jit
def channel_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ab: tl.constexpr,
    am: tl.constexpr,
    ak: tl.constexpr,
    bb: tl.constexpr,
    bk: tl.constexpr,
    bn: tl.constexpr,
    sab: tl.constexpr,
    sam: tl.constexpr,
    sbb: tl.constexpr,
    sbn: tl.constexpr,
    cb: tl.constexpr,
    cm: tl.constexpr,
    cn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    p = tl.program_id(0)
    batch = tl.program_id(1)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = p // (G * nn)
    size = tl.minimum(nm - group * G, G)
    pm = group * G + p % size
    pn = (p % (G * nn)) // size
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = A + batch * ab + m[:, None] * am + k[None, :] * ak
    bp = B + batch * bb + k[:, None] * bk + n[None, :] * bn
    acc = tl.full((BM, BN), 0, tl.int32)
    for i in range(tl.cdiv(K, BK)):
        av = tl.load(ap, (m[:, None] < M) & (k[None, :] + i * BK < K), other=0.0)
        bv = tl.load(bp, (k[:, None] + i * BK < K) & (n[None, :] < N), other=0.0)
        acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
        ap += BK * ak
        bp += BK * bk
    sa = tl.load(SA + batch * sab + m * sam, m < M, other=0.0).to(tl.float32)
    sb = tl.load(SB + batch * sbb + n * sbn, n < N, other=0.0).to(tl.float32)
    val = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + batch * cb + m[:, None] * cm + n[None, :] * cn,
        val,
        (m[:, None] < M) & (n[None, :] < N),
    )


def validate_inputs(A, B, A_scale, B_scale, out_dtype, out):
    if (
        A.ndim != 3
        or B.ndim != 3
        or A.shape[0] != B.shape[0]
        or A.shape[2] != B.shape[1]
    ):
        raise ValueError("Expected compatible batched matrices")
    batch, M, K = A.shape
    N = B.shape[2]
    if A_scale.shape not in ((batch, M), (batch, M, 1)) or B_scale.shape not in (
        (batch, N),
        (batch, 1, N),
    ):
        raise ValueError("Expected per-row A scales and per-output-channel B scales")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("Operands must both have INT8 dtype")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Expected FP16, BF16 or FP32 output")
    if A.device != B.device or A.device != A_scale.device or A.device != B_scale.device:
        raise ValueError("Inputs must share a device")
    if not A_scale.dtype.is_floating_point or not B_scale.dtype.is_floating_point:
        raise TypeError("Scales must be floating point")
    if out is None:
        out = torch.empty((batch, M, N), device=A.device, dtype=out_dtype)
    elif out.shape != (batch, M, N) or out.dtype != out_dtype or out.device != A.device:
        raise ValueError("Invalid output tensor")
    return out


def bmm_w8a8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    block_size=(128, 128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    *,
    config=None,
) -> torch.Tensor:
    """Channel-wise batched W8A8 matrix multiplication.

    A has shape (batch, M, K), B has shape (batch, K, N). A_scale is
    (batch, M) or (batch, M, 1); B_scale is (batch, N) or (batch, 1, N).
    INT8 accumulates in INT32. Scales are applied in FP32 before
    casting to out_dtype.
    block_size is retained for API compatibility and is unused here.
    """
    out = validate_inputs(A, B, A_scale, B_scale, out_dtype, out)
    batch, M, K = A.shape
    N = B.shape[2]
    if batch * M * N == 0:
        return out
    if config is None:
        if M < 1024:
            config = (64, 128, 64, 4, 1, 1)
        elif K >= 2048:
            config = (128, 128, 128, 8, 2, 8)
        else:
            config = (128, 128, 64, 4, 2, 8)
    bm, bn, bk, w, s, g = config
    with torch_device_fn.device(A.device):
        channel_kernel[(tr.cdiv(M, bm) * tr.cdiv(N, bn), batch)](
            A,
            B,
            A_scale,
            B_scale,
            out,
            M,
            N,
            K,
            *A.stride(),
            *B.stride(),
            A_scale.stride(0),
            A_scale.stride(1),
            B_scale.stride(0),
            B_scale.stride(-1),
            *out.stride(),
            bm,
            bn,
            bk,
            g,
            num_warps=w,
            num_stages=s,
        )
    return out
