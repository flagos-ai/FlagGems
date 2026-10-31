# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .mm_w8a8_int8 import _grouped_int8, _launch_prequantized

logger = logging.getLogger(__name__)

MAX_INT32_REDUCTION = (2**31 - 1) // (128 * 128)


# Hygon DCU, torch 2.9.0 (ROCm, HIP 6.3.26045) / Triton 3.6.0, 2026-10-23.
# Values: BLOCK_M, BLOCK_N, num_warps, num_stages, GROUP_M, ACC_FP32, SWAP_AB,
# optionally TILE_K (defaults to min(128, max(32, next_power_of_2(K)))) and UNROLL
# (K-loop unroll factor for K > 256, defaults to 1).
EXACT_CONFIGS = {
    (8, 8, 1024, 4096): (16, 64, 4, 2, 8, False, False, 64, 4),
    (8, 32, 1024, 4096): (32, 64, 8, 2, 1, False, False, 128, 1),
}


def _get_int8_config(batch, m, n, k):
    key = (batch, m, n, k)
    acc_fp32, swap_ab, tile_k, unroll = False, False, None, 1
    if key in EXACT_CONFIGS:
        values = EXACT_CONFIGS[key]
        bm, bn, warps, stages, group, acc_fp32, swap_ab, tile_k, unroll = (
            *values,
            None,
            1,
        )[:9]
    elif n == 1024 and k >= 4096:
        if batch == 1:
            if k == 4096 and (4096 <= m <= 5120 or m >= 12288):
                bm, bn, warps, stages, group = 128, 256, 4, 1, 4
            elif m <= 32:
                bm, bn, warps, stages, group = 16, 32, 4, 2, 1
            elif m <= 224:
                bm, bn, warps, stages, group = 32, 32, 4, 2, 1
            elif m <= 320:
                bm, bn, warps, stages, group = 64, 64, 8, 2, 1
            elif m <= 512:
                bm, bn, warps, stages, group = 64, 64, 4, 2, 1
            else:
                bm, bn, warps, stages, group = 128, 128, 8, 2, 1
        elif m <= 64:
            bm = min(64, max(16, triton.next_power_of_2(m)))
            bn = 64 if k <= 4096 else 128
            warps = 4 if k <= 4096 else 8
            stages, group = 2, 1
            unroll = 2 if k <= 4096 else 1
        else:
            bm, bn, stages = 128, 128, 2
            warps = 8 if k <= 4096 else 4
            group = 1 if k <= 4096 and m > 128 else 8
            unroll = 2 if k <= 4096 and m <= 128 else 1
    elif n >= 2048 and m >= 128:
        if k <= 256:
            bm, bn, warps, stages, group = 64, 64, 4, 2, 1
        elif m >= 12288 and n >= 4096 and n % 256 == 0 and k <= 1024:
            bm, bn, warps, stages, group = 128, 256, 4, 1, 8
        else:
            bm, bn, warps, stages = 128, 128, 4, 2
            group = 8 if n >= 4096 else 1
    elif m <= 64:
        bm, bn, warps, stages, group = 32, 64, 4, 2, 1
    elif k >= 4096:
        bm, bn, warps, stages, group = 128, 128, 8, 1, 8
    else:
        bm, bn, warps, stages, group = 64, 64, 4, 2, 4
    result = dict(
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=warps,
        num_stages=stages,
        GROUP_M=group,
        ACC_FP32=acc_fp32,
        SWAP_AB=swap_ab,
        TILE_K=tile_k,
        UNROLL=unroll,
    )
    return result


@libentry()
@triton.jit
def bmm_channel_int8_kernel(
    A,
    W,
    AS,
    WS,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALIGNED: tl.constexpr,
    TILE_K: tl.constexpr,
    UNROLL: tl.constexpr = 1,
    SWAP_AB: tl.constexpr = False,
):
    batch = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BLOCK_M)
    nn = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    size_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, TILE_K)
    if (
        max(M * SA[1] + K * SA[2], K * SW[1] + N * SW[2], M * SO[1] + N * SO[2])
        >= 2**31
    ):
        rm = rm.to(tl.int64)
        rn = rn.to(tl.int64)
        rk = rk.to(tl.int64)
    if SWAP_AB:
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    for kb in tl.range(
        tl.cdiv(K, TILE_K),
        loop_unroll_factor=((K + TILE_K - 1) // TILE_K if K <= 256 else UNROLL),
    ):
        kk = kb * TILE_K + rk
        ap = A + batch * SA[0] + rm[:, None] * SA[1] + kk[None, :] * SA[2]
        wp = W + batch * SW[0] + kk[:, None] * SW[1] + rn[None, :] * SW[2]
        if ALIGNED:
            a = tl.load(ap)
            w = tl.load(wp)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(wp, (kk[:, None] < K) & (rn[None, :] < N), other=0)
        if SWAP_AB:
            acc = tl.dot(tl.trans(w), tl.trans(a), acc, out_dtype=tl.int32)
        else:
            acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    if SWAP_AB:
        acc = tl.trans(acc)
    a_scale = tl.load(AS + batch * SAS[0] + rm * SAS[1], rm < M, other=0)
    w_scale = tl.load(WS + batch * SWS[0] + rn * SWS[2], rn < N, other=0)
    acc = acc.to(tl.float32) * a_scale[:, None]
    acc = acc * w_scale[None, :]
    op = O + batch * SO[0] + rm[:, None] * SO[1] + rn[None, :] * SO[2]
    if ALIGNED:
        tl.store(op, acc)
    else:
        tl.store(op, acc, (rm[:, None] < M) & (rn[None, :] < N))


def bmm_w8a8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel-scaled INT8 BMM with INT32 accumulation on Hygon.

    A[B,M,K] is contiguous; W[B,K,N] is a transpose view of [B,N,K].
    FP32 scales have shapes [B,M] / [B,N], optionally [B,M,1] / [B,1,N].
    Inputs are prequantized; batch broadcasting and autograd are unsupported.
    """
    logger.debug("GEMS_HYGON BMM_W8A8")
    if activation.ndim != 3 or weight.ndim != 3:
        raise ValueError("expected activation[B,M,K] and weight[B,K,N]")
    batch, rows, reduction = activation.shape
    columns = weight.shape[2]
    if weight.shape[:2] != (batch, reduction):
        raise ValueError("batch and reduction dimensions must match")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be signed INT8 tensors")
    if activation.device.type != "cuda" or any(
        tensor.device != activation.device
        for tensor in (weight, activation_scale, weight_scale)
    ):
        raise ValueError("inputs and scales must be on the same Hygon device")
    if not activation.is_contiguous() or not weight.transpose(1, 2).is_contiguous():
        raise ValueError("expected contiguous activation and column-major weight")
    if activation_scale.shape not in ((batch, rows), (batch, rows, 1)):
        raise ValueError("activation_scale must have shape [B,M] or [B,M,1]")
    if weight_scale.shape not in ((batch, columns), (batch, 1, columns)):
        raise ValueError("weight_scale must have shape [B,N] or [B,1,N]")
    if any(
        scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (activation_scale, weight_scale)
    ):
        raise TypeError("scales must be contiguous FP32 tensors")
    if any(scale.requires_grad for scale in (activation_scale, weight_scale)):
        raise NotImplementedError("bmm_w8a8 is forward-only")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be bfloat16, float16 or float32")
    if out is None:
        out = torch.empty(
            (batch, rows, columns), dtype=out_dtype, device=activation.device
        )
    elif (
        out.shape != (batch, rows, columns)
        or out.dtype != out_dtype
        or out.device != activation.device
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError("out must be contiguous with matching shape, dtype and device")
    elif out.numel() and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (activation, weight, activation_scale, weight_scale)
    ):
        raise ValueError("out must not share storage with an input")
    if out.numel() == 0:
        return out
    if reduction > MAX_INT32_REDUCTION:
        raise NotImplementedError("reduction may overflow signed INT32 accumulation")
    with torch_device_fn.device(activation.device):
        if reduction == 0:
            return zero_(out)
        if (
            batch == 1
            and (rows <= 64 or 72 <= rows <= 512)
            and columns == 1024
            and reduction == 4096
        ):
            if rows in (16, 24, 32) or rows > 32:
                if rows > 224:
                    config = _get_int8_config(batch, rows, columns, reduction)
                    block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]
                    block_k = 2 * (config["TILE_K"] or 128)
                    warps, stages = config["num_warps"], config["num_stages"]
                    group_m = config["GROUP_M"]
                else:
                    block_m = 32 if rows > 32 else 16
                    block_n = 32 if rows >= 72 else 16
                    block_k, stages, group_m = 256, 2, 0
                    warps = 4 if 72 <= rows <= 160 else 2
                grouped_kernel = (
                    _parallel_grouped_int8 if 72 <= rows <= 512 else _grouped_int8
                )
                epilogue = (
                    dict(Bias=None, HAS_BIAS=False, A_SCALAR=False, B_SCALAR=False)
                    if rows < 72
                    else {}
                )
                grouped_kernel[
                    (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)
                ](
                    activation,
                    weight,
                    activation_scale,
                    weight_scale,
                    out,
                    rows,
                    columns,
                    reduction,
                    block_m,
                    block_n,
                    block_k,
                    stages,
                    group_m,
                    enable_fp_fusion=False,
                    num_warps=warps,
                    num_stages=stages,
                    **epilogue,
                )
                return out
            # Validated B=1 layouts already match the MM kernel's flat storage.
            return _launch_prequantized(
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                reduction,
                None,
            )
        config = _get_int8_config(batch, rows, columns, reduction)
        block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]
        block_k = config["TILE_K"] or min(
            128, max(32, triton.next_power_of_2(reduction))
        )
        bmm_channel_int8_kernel[
            (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n), batch)
        ](
            activation,
            weight,
            activation_scale,
            weight_scale,
            out,
            rows,
            columns,
            reduction,
            activation.stride(),
            weight.stride(),
            (activation_scale.stride(0), activation_scale.stride(1), 0),
            (weight_scale.stride(0), 0, weight_scale.stride(-1)),
            out.stride(),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ALIGNED=rows % block_m == 0
            and columns % block_n == 0
            and reduction % block_k == 0,
            TILE_K=block_k,
            GROUP_M=config["GROUP_M"],
            UNROLL=config["UNROLL"],
            SWAP_AB=config["SWAP_AB"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
    return out


@libentry()
@triton.jit
def _parallel_grouped_int8(
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
    rk = tl.arange(0, BK // 2)
    acc0 = tl.zeros((BM, BN), tl.int32)
    acc1 = tl.zeros((BM, BN), tl.int32)
    for start in tl.range(0, tl.cdiv(K, BK), num_stages=ST):
        k0 = start * BK + rk
        k1 = BK // 2 + k0
        a0 = tl.load(
            A + rm[:, None] * K + k0[None, :],
            (rm[:, None] < M) & (k0[None, :] < K),
            other=0,
        )
        b0 = tl.load(
            B + rn[None, :] * K + k0[:, None],
            (rn[None, :] < N) & (k0[:, None] < K),
            other=0,
        )
        a1 = tl.load(
            A + rm[:, None] * K + k1[None, :],
            (rm[:, None] < M) & (k1[None, :] < K),
            other=0,
        )
        b1 = tl.load(
            B + rn[None, :] * K + k1[:, None],
            (rn[None, :] < N) & (k1[:, None] < K),
            other=0,
        )
        acc0 = tl.dot(a0, b0, acc0, out_dtype=tl.int32)
        acc1 = tl.dot(a1, b1, acc1, out_dtype=tl.int32)
    acc = acc0 + acc1
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(tl.float32) * sa[:, None] * sb[None, :],
        (rm[:, None] < M) & (rn[None, :] < N),
    )
