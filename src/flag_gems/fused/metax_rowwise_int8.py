# SPDX-License-Identifier: Apache-2.0
"""Row-scaled & Group-2048 INT8 projections (including Wave64 Split-K) for MetaX C500."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _quantize_rows(
    X,
    Codes,
    Scales,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    m = col < K
    values = tl.load(X + row * STRIDE + col, mask=m, other=0.0).to(tl.float32)
    peak = tl.max(tl.abs(values), 0)
    scale = tl.where(peak > 0.0, peak * (1.0 / 127.0), 1.0)
    inv_scale = tl.where(peak > 0.0, 127.0 / peak, 0.0)
    q = tl.extra.cuda.libdevice.rint(values * inv_scale).to(tl.int32).to(tl.int8)
    tl.store(Codes + row * K + col, q, mask=m)
    tl.store(Scales + row, scale)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    codes = torch.empty((rows, width), dtype=torch.int8, device=x.device)
    scales = torch.empty(rows, dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(width)
    waves = 4 if block <= 2048 else 8
    with torch_device_fn.device(x.device):
        _quantize_rows[(rows,)](
            x,
            codes,
            scales,
            width,
            x.stride(0),
            block,
            num_warps=waves,
            num_stages=1,
        )
    return codes, scales


@triton.jit
def _rowwise_int8_mm(
    X,
    W,
    XScale,
    WScale,
    Y,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    if GROUP_M > 1 and tiles_m > 1:
        first_row_tile = pid // (GROUP_M * tiles_n) * GROUP_M
        group_size = tl.minimum(GROUP_M, tiles_m - first_row_tile)
        local = pid % (GROUP_M * tiles_n)
        pid_m = first_row_tile + local % group_size
        pid_n = local // group_size
    else:
        pid_m = pid // tiles_n
        pid_n = pid % tiles_n

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    read_row = (
        tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        if EVEN_M
        else tl.where(rm < M, rm, 0)
    )
    read_col = (
        tl.max_contiguous(tl.multiple_of(rn, BN), BN)
        if EVEN_N
        else tl.where(rn < N, rn, 0)
    )
    k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BK), BK), BK)

    a = X + read_row[:, None] * K + k[None, :]
    w = W + read_col[:, None] * K + k[None, :]
    acc = tl.full((BM, BN), 0, tl.int32)
    for _ in range(0, K // BK):
        av = tl.load(a)
        bv = tl.trans(tl.load(w))
        acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
        a += BK
        w += BK

    sx = tl.load(XScale + read_row, mask=rm < M, other=0.0)
    sw = tl.load(WScale + read_col, mask=rn < N, other=0.0)
    result = (acc.to(tl.float32) * (sx[:, None] * sw[None, :])).to(Y.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(Y + rm[:, None] * N + rn[None, :], result)
    else:
        tl.store(
            Y + rm[:, None] * N + rn[None, :],
            result,
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _rowwise_int8_g2048_mm(
    X,
    W,
    XScales,
    WScale,
    Y,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Single-kernel Group-2048 INT8 GEMM for K=6144 (3 independent per-row scales)."""
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    if GROUP_M > 1 and tiles_m > 1:
        first_row_tile = pid // (GROUP_M * tiles_n) * GROUP_M
        group_size = tl.minimum(GROUP_M, tiles_m - first_row_tile)
        local = pid % (GROUP_M * tiles_n)
        pid_m = first_row_tile + local % group_size
        pid_n = local // group_size
    else:
        pid_m = pid // tiles_n
        pid_n = pid % tiles_n

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    read_row = (
        tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        if EVEN_M
        else tl.where(rm < M, rm, 0)
    )
    read_col = (
        tl.max_contiguous(tl.multiple_of(rn, BN), BN)
        if EVEN_N
        else tl.where(rn < N, rn, 0)
    )
    rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BK), BK), BK)

    out_f32 = tl.zeros((BM, BN), dtype=tl.float32)
    ITERS_PER_GRP: tl.constexpr = 2048 // BK
    for g in tl.static_range(0, 3):
        a = X + read_row[:, None] * 6144 + (g * 2048 + rk)[None, :]
        w = W + read_col[:, None] * 6144 + (g * 2048 + rk)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for _ in range(0, ITERS_PER_GRP):
            acc = tl.dot(tl.load(a), tl.trans(tl.load(w)), acc, out_dtype=tl.int32)
            a += BK
            w += BK
        sxg = tl.load(XScales + g * M + read_row, mask=rm < M, other=0.0)
        out_f32 += acc.to(tl.float32) * sxg[:, None]

    sw = tl.load(WScale + read_col, mask=rn < N, other=0.0)
    res = (out_f32 * sw[None, :]).to(Y.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(Y + rm[:, None] * N + rn[None, :], res)
    else:
        tl.store(
            Y + rm[:, None] * N + rn[None, :],
            res,
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _rowwise_int8_mm_splitk(
    X,
    W,
    Part,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    tiles_n = tl.cdiv(N, BN)
    pid_m = pid // tiles_n
    pid_n = pid % tiles_n

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    read_row = (
        tl.max_contiguous(tl.multiple_of(rm, BM), BM)
        if EVEN_M
        else tl.where(rm < M, rm, 0)
    )
    read_col = (
        tl.max_contiguous(tl.multiple_of(rn, BN), BN)
        if EVEN_N
        else tl.where(rn < N, rn, 0)
    )
    rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BK), BK), BK)

    total_iters = K // BK
    iters_per_split = (total_iters + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * iters_per_split
    k_end = tl.minimum(total_iters, k_start + iters_per_split)

    a = X + read_row[:, None] * K + (k_start * BK + rk)[None, :]
    w = W + read_col[:, None] * K + (k_start * BK + rk)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(k_start, k_end):
        av = tl.load(a)
        bv = tl.trans(tl.load(w))
        acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
        a += BK
        w += BK

    p_ptrs = Part + pid_k * M * N + rm[:, None] * N + rn[None, :]
    if EVEN_M and EVEN_N:
        tl.store(p_ptrs, acc)
    else:
        tl.store(p_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _i32_splitk_dequant_reduce(
    Part,
    XScale,
    WScale,
    Y,
    M,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BS: tl.constexpr,
):
    idx = tl.program_id(0) * BS + tl.arange(0, BS)
    mn = M * N
    m = idx < mn
    acc = tl.zeros((BS,), dtype=tl.int32)
    for s in tl.static_range(0, SPLIT_K):
        acc += tl.load(Part + s * mn + idx, mask=m, other=0)
    r = idx // N
    c = idx % N
    sx = tl.load(XScale + r, mask=m, other=0.0)
    sw = tl.load(WScale + c, mask=m, other=0.0)
    res = (acc.to(tl.float32) * (sx * sw)).to(Y.dtype.element_ty)
    tl.store(Y + idx, res, mask=m)


@triton.jit
def _i32_splitk3_group_dequant_reduce(
    Part,
    XScales,
    WScale,
    Y,
    M,
    N: tl.constexpr,
    BS: tl.constexpr,
):
    idx = tl.program_id(0) * BS + tl.arange(0, BS)
    mn = M * N
    m = idx < mn
    r = idx // N
    c = idx % N
    p0 = tl.load(Part + 0 * mn + idx, mask=m, other=0).to(tl.float32)
    p1 = tl.load(Part + 1 * mn + idx, mask=m, other=0).to(tl.float32)
    p2 = tl.load(Part + 2 * mn + idx, mask=m, other=0).to(tl.float32)
    sx0 = tl.load(XScales + 0 * M + r, mask=m, other=0.0)
    sx1 = tl.load(XScales + 1 * M + r, mask=m, other=0.0)
    sx2 = tl.load(XScales + 2 * M + r, mask=m, other=0.0)
    sw = tl.load(WScale + c, mask=m, other=0.0)
    res = ((p0 * sx0 + p1 * sx1 + p2 * sx2) * sw).to(Y.dtype.element_ty)
    tl.store(Y + idx, res, mask=m)


def _select_int8_tile(rows: int, width: int, outputs: int):
    if rows <= 16:
        return 16, 64, 128, 4, 1, ""
    if rows <= 32:
        return 32, 64, 128, 8, 1, ""
    if rows <= 64:
        return 64, 64, 128, 8, 1, ""
    if rows <= 1024:
        bk = 128 if width == 6144 else 64
        return 128, 128, bk, 8, 8, ""
    if outputs == 2048 and width == 2048:
        return 128, 128, 64, 8, 8, ""
    if outputs == 2048 and width == 6144:
        return 128, 128, 128, 8, 8, "fullstage"
    if outputs == 12288 and width == 2048:
        return 256, 256, 64, 8, 8, "fullstage,storeCoalesce"
    if outputs == 2560 and width == 2048:
        return 256, 256, 64, 8, 8, "fullstage,unroll"
    return 256, 256, 64, 8, 8, "fullstage"


def project_codes(
    activation_codes: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    rows, width = activation_codes.shape
    outputs = weight_codes.shape[0]
    bm, bn, bk, waves, group_m, scenario = _select_int8_tile(rows, width, outputs)
    result = torch.empty(
        (rows, outputs), dtype=out_dtype, device=activation_codes.device
    )
    launch_kw = dict(num_warps=waves, num_stages=2, pipeline="basic")
    if scenario:
        launch_kw["scenario"] = scenario
    with torch_device_fn.device(activation_codes.device):
        _rowwise_int8_mm[(triton.cdiv(rows, bm) * triton.cdiv(outputs, bn),)](
            activation_codes,
            weight_codes,
            activation_scales,
            weight_scales,
            result,
            rows,
            width,
            outputs,
            bm,
            bn,
            bk,
            group_m,
            rows % bm == 0,
            outputs % bn == 0,
            **launch_kw,
        )
    return result


def project_codes_splitk_part(
    activation_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    splits: int,
) -> torch.Tensor:
    """Return unreduced [splits, M, N] int32 partials for fused downstream epilogue."""
    rows, width = activation_codes.shape
    outputs = weight_codes.shape[0]
    bm = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    bn = 64
    bk = 128
    waves = 4 if rows <= 16 else 8
    device = activation_codes.device
    part = torch.empty((splits, rows, outputs), dtype=torch.int32, device=device)
    with torch_device_fn.device(device):
        grid = (triton.cdiv(rows, bm) * triton.cdiv(outputs, bn), splits)
        _rowwise_int8_mm_splitk[grid](
            activation_codes,
            weight_codes,
            part,
            rows,
            width,
            outputs,
            bm,
            bn,
            bk,
            splits,
            rows % bm == 0,
            outputs % bn == 0,
            num_warps=waves,
            num_stages=2,
            pipeline="basic",
        )
    return part


def project_codes_splitk(
    activation_codes: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
    splits: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    rows, width = activation_codes.shape
    outputs = weight_codes.shape[0]
    if splits <= 1 or rows > 64:
        return project_codes(
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            out_dtype=out_dtype,
        )
    device = activation_codes.device
    part = project_codes_splitk_part(activation_codes, weight_codes, splits)
    result = torch.empty((rows, outputs), dtype=out_dtype, device=device)
    with torch_device_fn.device(device):
        mn = rows * outputs
        bs = 1024 if mn >= 262144 else 512
        rwaves = 8 if bs == 1024 else 4
        _i32_splitk_dequant_reduce[(triton.cdiv(mn, bs),)](
            part,
            activation_scales,
            weight_scales,
            result,
            rows,
            outputs,
            splits,
            bs,
            num_warps=rwaves,
            num_stages=1,
        )
    return result


def project_codes_g2048_k6144(
    activation_codes: torch.Tensor,
    activation_group_scales: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Group-2048 INT8 projection for K=6144 (Split-K=3 for M<=64, fused 3-group GEMM for M>64)."""
    rows, width = activation_codes.shape
    outputs = weight_codes.shape[0]
    device = activation_codes.device
    result = torch.empty((rows, outputs), dtype=out_dtype, device=device)
    with torch_device_fn.device(device):
        if rows <= 64:
            part = project_codes_splitk_part(activation_codes, weight_codes, 3)
            mn = rows * outputs
            bs = 1024 if mn >= 262144 else 512
            rwaves = 8 if bs == 1024 else 4
            _i32_splitk3_group_dequant_reduce[(triton.cdiv(mn, bs),)](
                part,
                activation_group_scales,
                weight_scales,
                result,
                rows,
                outputs,
                bs,
                num_warps=rwaves,
                num_stages=1,
            )
            return result
        bm, bn, bk, waves, group_m = 128, 128, 128, 8, 8
        _rowwise_int8_g2048_mm[(triton.cdiv(rows, bm) * triton.cdiv(outputs, bn),)](
            activation_codes,
            weight_codes,
            activation_group_scales,
            weight_scales,
            result,
            rows,
            outputs,
            bm,
            bn,
            bk,
            group_m,
            rows % bm == 0,
            outputs % bn == 0,
            num_warps=waves,
            num_stages=2,
            pipeline="basic",
        )
    return result


def project(
    x: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    activation_codes, activation_scales = quantize_rows(x)
    return project_codes(
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
        out_dtype=x.dtype,
    )
