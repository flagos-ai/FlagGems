# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit INT8 scaled GEMM; independent of ATen's FP8-only _scaled_mm."""

import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.fused.cutlass_scaled_mm import cutlass_scaled_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

GROUP_M = 8


def _heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("scaled_mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=2,
    rep=4,
)
@triton.heuristics({"EVEN_K": _heur_even_k})
@triton.jit
def scaled_mm_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = offs_m.to(tl.int64)
    offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            )
        if ACC_DTYPE == tl.int32:
            acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_DTYPE, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc.to(tl.float32)

    if SCALE_A_MODE == 0:
        scale_a = tl.full((BLOCK_M,), tl.load(ScaleA), dtype=tl.float32)
    else:
        scale_a = tl.load(ScaleA + offs_m, mask=offs_m < M, other=0.0)

    if SCALE_B_MODE == 0:
        scale_b = tl.full((BLOCK_N,), tl.load(ScaleB), dtype=tl.float32)
    else:
        scale_b = tl.load(ScaleB + offs_n, mask=offs_n < N, other=0.0)

    acc = acc * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _int8_scale(scale, size):
    if scale.dtype != torch.float32 or not scale.is_contiguous():
        raise ValueError("INT8 GEMM scales must be contiguous FP32")
    if scale.numel() not in (1, size):
        raise ValueError("INT8 GEMM scale must be scalar or per row/column")
    return scale.view(-1), int(scale.numel() != 1)


def scaled_mm_int8(a, b, scale_a, scale_b, bias=None, out_dtype=torch.bfloat16):
    """Forward CUDA INT8 `[M,K] x [K,N]` with FP32 scales and optional bias.

    Uses the existing specialized column-major path first. Exact Hopper tiles
    are opt-in for row-major/strided fallback; other shapes retain autotuning.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("INT8 GEMM requires compatible 2D matrices")
    if a.device.type != "cuda" or a.dtype != torch.int8 or b.dtype != torch.int8:
        raise NotImplementedError("scaled_mm_int8 supports CUDA signed INT8 inputs")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("INT8 GEMM output must be FP16, BF16 or FP32")
    m, k = a.shape
    n = b.shape[1]
    if k > 65536:
        raise NotImplementedError("INT8 GEMM K must be <=65536 to bound accumulation")
    for x in (b, scale_a, scale_b, bias):
        if x is not None and x.device != a.device:
            raise ValueError("INT8 GEMM tensors must share a device")
        if x is not None and x.requires_grad:
            raise NotImplementedError("INT8 GEMM supports forward inference only")
    scale_a, mode_a = _int8_scale(scale_a, m)
    scale_b, mode_b = _int8_scale(scale_b, n)
    if bias is not None and (bias.shape != (n,) or not bias.is_contiguous()):
        raise ValueError("INT8 GEMM bias must be a contiguous [N] vector")
    if bias is not None and bias.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("INT8 GEMM bias must have a floating dtype")
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    if m == 0 or n == 0:
        return out
    tile = None
    capability = torch.cuda.get_device_capability(a.device)
    with torch_device_fn.device(a.device):
        if (
            capability[0] == 9
            and k > 0
            and a.stride(1) == 1
            and b.stride(0) == 1
            and b.stride(1) % 16 == 0
            and n % 16 == 0
        ):
            cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
            return out
        if (
            capability == (9, 0)
            and os.getenv("FLAGGEMS_I8_SCALED_MM_SHAPE_TILES") == "1"
        ):
            if m <= 64 and (k, n) == (6144, 1536):
                tile = (16, 64, 128, 4, 3)
            elif m in (4096, 5089, 8192) and (k, n) in (
                (6144, 1536),
                (1024, 6144),
                (6144, 3072),
                (1536, 6144),
                (6144, 768),
            ):
                tile = (32, 64, 256, 4, 3)
        args = (
            a,
            b,
            scale_a,
            scale_b,
            bias,
            out,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            *out.stride(),
        )
        kwargs = dict(
            ACC_DTYPE=tl.int32,
            SCALE_A_MODE=mode_a,
            SCALE_B_MODE=mode_b,
            HAS_BIAS=bias is not None,
            GROUP_M=GROUP_M,
        )
        if tile:
            bm, bn, bk, warps, stages = tile
            scaled_mm_kernel.jit_function[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                *args,
                **kwargs,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                EVEN_K=k % bk == 0,
                num_warps=warps,
                num_stages=stages,
            )
        else:
            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            )
            scaled_mm_kernel[grid](*args, **kwargs)
    return out
