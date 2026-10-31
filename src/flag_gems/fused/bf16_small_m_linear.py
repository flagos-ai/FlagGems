# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Forward-only BF16 linear specialization for small-M SM90 inputs.

The bounded schedule retains the complete K reduction in one CTA and a BF16
output boundary. The caller resolves device capability before graph capture.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def supported_metadata(
    *,
    m,
    n,
    k,
    input_dtype,
    weight_dtype,
    input_stride,
    weight_stride,
    device_sm,
    bias_present=False,
    same_device=True,
):
    """Pure metadata guard; resolve device_sm once, before graph capture."""
    return (
        device_sm == 90
        and 1 <= m <= 16
        and (n, k) == (1024, 6144)
        and input_dtype == weight_dtype == "bfloat16"
        and tuple(input_stride) == (6144, 1)
        and tuple(weight_stride) == (6144, 1)
        and not bias_present
        and same_device
    )


@triton.jit
def _linear_small_n_kernel(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rm = tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + rk
        a = tl.load(
            X + rm[:, None] * K + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + rn[None, :] * K + kk[:, None],
            (rn[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc += tl.dot(a, b, allow_tf32=False)
    tl.store(
        Y + rm[:, None] * N + rn[None, :],
        acc.to(tl.bfloat16),
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def bf16_small_m_linear(input, weight, *, device_sm, out=None):
    """Compute input @ weight.T for the supported inference domain.

    Call with gradients disabled. Unsupported metadata raises before launch;
    callers select their original linear implementation before invoking this API.
    """
    if input.ndim != 2 or weight.ndim != 2:
        raise ValueError("BF16 small-M linear requires 2-D input and weight")
    m, k = input.shape
    n = weight.shape[0]
    if (
        not supported_metadata(
            m=m,
            n=n,
            k=k,
            input_dtype=str(input.dtype).removeprefix("torch."),
            weight_dtype=str(weight.dtype).removeprefix("torch."),
            input_stride=input.stride(),
            weight_stride=weight.stride(),
            device_sm=device_sm,
            bias_present=False,
            same_device=input.device == weight.device,
        )
        or weight.shape[1] != k
        or input.device.type != "cuda"
    ):
        raise ValueError("unsupported BF16 small-M linear metadata")
    if out is None:
        out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    if (
        tuple(out.shape) != (m, n)
        or out.dtype != input.dtype
        or out.device != input.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [M,N] on the input device")
    if out.untyped_storage().data_ptr() in (
        input.untyped_storage().data_ptr(),
        weight.untyped_storage().data_ptr(),
    ):
        raise ValueError("out must not share input/weight storage")
    # One bounded configuration; no autotuning or allocation during replay.
    with torch.cuda.device(input.device):
        _linear_small_n_kernel[(triton.cdiv(n, 16),)](
            input,
            weight,
            out,
            m,
            n,
            k,
            BM=16,
            BN=16,
            BK=256,
            num_warps=4,
            num_stages=4,
        )
    return out
