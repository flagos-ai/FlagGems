# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Flat vector loads for contiguous add, retaining generic broadcasting."""

import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.add import add as _generic_add
from flag_gems.runtime import torch_device_fn


@triton.jit
def _transposed_add(
    A,
    B,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    AS: tl.constexpr,
    AM: tl.constexpr,
    AN: tl.constexpr,
    BS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ALPHA: tl.constexpr,
    TILE: tl.constexpr,
):
    batch = tl.program_id(2)
    m = tl.program_id(0) * TILE + tl.arange(0, TILE)
    n = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = (m[:, None] < M) & (n[None, :] < N)
    a = tl.load(A + batch * AS + m[:, None] * AM + n[None, :] * AN, valid, other=0)
    b = tl.load(B + batch * BS + m[:, None] * BM + n[None, :] * BN, valid, other=0)
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if (dtype == tl.float16 or dtype == tl.bfloat16) else dtype
    )
    tl.store(
        O + batch * M * N + m[:, None] * N + n[None, :],
        a.to(compute) + b.to(compute) * ALPHA,
        valid,
    )


@triton.jit
def _strided_add(
    A,
    B,
    O,
    SIZE: tl.constexpr,
    SHAPE: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cur = i
    ao = tl.full((BLOCK,), 0, tl.int64)
    bo = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = cur % SHAPE[d]
        ao += coord * AS[d]
        bo += coord * BS[d]
        cur = cur // SHAPE[d]
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if (dtype == tl.float16 or dtype == tl.bfloat16) else dtype
    )
    a = tl.load(A + ao, i < SIZE, other=0).to(compute)
    b = tl.load(B + bo, i < SIZE, other=0).to(compute)
    tl.store(O + i, a + b * ALPHA, i < SIZE)


@triton.jit
def _flat_add(
    A,
    B,
    O,
    SIZE: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALAR: tl.constexpr,
    PERIOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < SIZE
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if (dtype == tl.float16 or dtype == tl.bfloat16) else dtype
    )
    x = tl.load(A + i, valid, other=0).to(compute)
    if SCALAR:
        y = tl.full((), B, compute)
    else:
        if PERIOD == SIZE:
            j = i
        else:
            j = i % PERIOD
        y = tl.load(B + j, valid, other=0).to(compute)
    # Keep multiplication and addition separate, matching ATen add's alpha rule.
    value = x + y * ALPHA
    tl.store(O + i, value, valid)


def _launch_strided(a, b, alpha):
    shape = torch.broadcast_shapes(a.shape, b.shape)
    av, bv = a.broadcast_to(shape), b.broadcast_to(shape)
    size = av.numel()
    out = torch.empty(shape, dtype=a.dtype, device=a.device)
    if size:
        with torch_device_fn.device(a.device):
            tile = (
                len(shape) >= 2
                and a.shape == b.shape
                and a.stride(-1) == b.stride(-2) == 1
                and a.stride(-2) == b.stride(-1)
                and all(
                    a.stride(d) == a.stride(d + 1) * a.shape[d + 1]
                    and b.stride(d) == b.stride(d + 1) * b.shape[d + 1]
                    for d in range(a.ndim - 3)
                )
            )
            if tile:
                m, n = shape[-2:]
                batch = size // (m * n)
                _transposed_add[(triton.cdiv(m, 32), triton.cdiv(n, 32), batch)](
                    a,
                    b,
                    out,
                    m,
                    n,
                    a.stride(-3) if a.ndim >= 3 else 0,
                    *a.stride()[-2:],
                    b.stride(-3) if b.ndim >= 3 else 0,
                    *b.stride()[-2:],
                    alpha,
                    32,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                block = 64 if size <= 65536 else 512
                _strided_add[(triton.cdiv(size, block),)](
                    a,
                    b,
                    out,
                    size,
                    tuple(shape),
                    tuple(av.stride()),
                    tuple(bv.stride()),
                    alpha,
                    block,
                    num_warps=1 if size <= 65536 else 2,
                    enable_fp_fusion=False,
                )
    return out


def _legacy_add(a, b, *, alpha=1):
    if not isinstance(alpha, (int, float)):
        return _generic_add(a, b, alpha=alpha)
    if not isinstance(a, torch.Tensor) or a.is_complex() or a.dtype == torch.bool:
        return _generic_add(a, b, alpha=alpha)
    if not a.dtype.is_floating_point and not isinstance(alpha, int):
        return _generic_add(a, b, alpha=alpha)
    scalar = not isinstance(b, torch.Tensor)
    if scalar:
        if (
            not a.is_contiguous()
            or not isinstance(b, (int, float))
            or torch.result_type(a, b) != a.dtype
        ):
            return _generic_add(a, b, alpha=alpha)
        period = 1
    else:
        if b.device != a.device or b.dtype != a.dtype:
            return _generic_add(a, b, alpha=alpha)
        if not a.is_contiguous() or not b.is_contiguous():
            return _launch_strided(a, b, alpha)
        if b.shape == a.shape:
            period = a.numel()
        elif (b.numel() == 1 and b.ndim <= a.ndim) or (
            b.ndim == 1 and a.ndim and b.shape[0] == a.shape[-1]
        ):
            period = b.numel()
        else:
            return _launch_strided(a, b, alpha)
    if not isinstance(alpha, (int, float)):
        return _generic_add(a, b, alpha=alpha)
    out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    if a.numel():
        if a.dtype == torch.float32 and 24576 <= a.numel() <= 32768:
            block, warps = 256, 1
        elif a.numel() <= 32768:
            block, warps = 64, 1
        elif a.numel() <= 65536 or (a.dtype == torch.float64 and a.numel() >= 1048576):
            block, warps = 256, 1
        elif a.numel() <= 131072:
            block, warps = 512, 2
        else:
            block, warps = 1024, 4
        with torch_device_fn.device(a.device):
            _flat_add[(triton.cdiv(a.numel(), block),)](
                a,
                b,
                out,
                a.numel(),
                alpha,
                scalar,
                period,
                block,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return out


@triton.jit
def _ordered_offset(i, SH: tl.constexpr, ST: tl.constexpr):
    out = tl.full(i.shape, 0, tl.int64)
    for d in tl.static_range(len(SH) - 1, -1, -1):
        if ST[d] != 0:
            out += (i % SH[d]) * ST[d]
        i = i // SH[d]
    return out


@triton.jit
def _ordered_flat(
    A,
    B,
    O,
    N: tl.constexpr,
    SH: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    ALINEAR: tl.constexpr,
    BLINEAR: tl.constexpr,
    SCALAR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if dtype == tl.bfloat16 or dtype == tl.float16 else dtype
    )
    ao = i if ALINEAR else _ordered_offset(i, SH, AS)
    if (tl.program_id(0) + 1) * BLOCK <= N:
        a = tl.load(A + ao).to(compute)
    else:
        a = tl.load(A + ao, valid, other=0).to(compute)
    if SCALAR:
        b = tl.full((), B, compute)
    else:
        bo = i if BLINEAR else _ordered_offset(i, SH, BS)
        if (tl.program_id(0) + 1) * BLOCK <= N:
            b = tl.load(B + bo).to(compute)
        else:
            b = tl.load(B + bo, valid, other=0).to(compute)
    if (tl.program_id(0) + 1) * BLOCK <= N:
        tl.store(O + i, a + b * ALPHA)
    else:
        tl.store(O + i, a + b * ALPHA, valid)


@triton.jit
def _ordered_tile(
    A,
    B,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    PS: tl.constexpr,
    APS: tl.constexpr,
    BPS: tl.constexpr,
    OPS: tl.constexpr,
    AM: tl.constexpr,
    AN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    p = tl.program_id(2)
    m = tl.program_id(0) * TM + tl.arange(0, TM)
    n = tl.program_id(1) * TN + tl.arange(0, TN)
    valid = (m[:, None] < M) & (n[None, :] < N)
    a = tl.load(
        A + _ordered_offset(p, PS, APS) + m[:, None] * AM + n[None, :] * AN,
        valid,
        other=0,
    )
    b = tl.load(
        B + _ordered_offset(p, PS, BPS) + m[:, None] * BM + n[None, :] * BN,
        valid,
        other=0,
    )
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if dtype == tl.bfloat16 or dtype == tl.float16 else dtype
    )
    tl.store(
        O + _ordered_offset(p, PS, OPS) + m[:, None] * OM + n[None, :] * ON,
        a.to(compute) + b.to(compute) * ALPHA,
        valid,
    )


@triton.jit
def _ordered_row(
    A,
    B,
    O,
    COLS: tl.constexpr,
    SH: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    AC: tl.constexpr,
    BC: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if dtype == tl.bfloat16 or dtype == tl.float16 else dtype
    )
    if (tl.program_id(1) + 1) * BLOCK <= COLS:
        av = tl.load(A + _ordered_offset(row, SH, AS) + col * AC).to(compute)
    else:
        av = tl.load(
            A + _ordered_offset(row, SH, AS) + col * AC, col < COLS, other=0
        ).to(compute)
    if (tl.program_id(1) + 1) * BLOCK <= COLS:
        bv = tl.load(B + _ordered_offset(row, SH, BS) + col * BC).to(compute)
    else:
        bv = tl.load(
            B + _ordered_offset(row, SH, BS) + col * BC, col < COLS, other=0
        ).to(compute)
    if (tl.program_id(1) + 1) * BLOCK <= COLS:
        tl.store(O + row * COLS + col, av + bv * ALPHA)
    else:
        tl.store(O + row * COLS + col, av + bv * ALPHA, col < COLS)


@triton.jit
def _ordered_matrix(
    A,
    B,
    O,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    SH: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    AC: tl.constexpr,
    BC: tl.constexpr,
    ALPHA: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    row = tl.program_id(0) * TM + tl.arange(0, TM)
    col = tl.program_id(1) * TN + tl.arange(0, TN)
    valid = (row[:, None] < ROWS) & (col[None, :] < COLS)
    dtype: tl.constexpr = O.dtype.element_ty
    compute: tl.constexpr = (
        tl.float32 if dtype == tl.bfloat16 or dtype == tl.float16 else dtype
    )
    av = tl.load(
        A + _ordered_offset(row, SH, AS)[:, None] + col[None, :] * AC, valid, other=0
    ).to(compute)
    bv = tl.load(
        B + _ordered_offset(row, SH, BS)[:, None] + col[None, :] * BC, valid, other=0
    ).to(compute)
    tl.store(O + row[:, None] * COLS + col[None, :], av + bv * ALPHA, valid)


def add(a, b, *, alpha=1):
    scalar = not isinstance(b, torch.Tensor)
    if (
        not isinstance(a, torch.Tensor)
        or a.is_complex()
        or a.dtype == torch.bool
        or not isinstance(alpha, (int, float))
        or (
            scalar
            and (not isinstance(b, (int, float)) or torch.result_type(a, b) != a.dtype)
        )
        or (not scalar and (b.device != a.device or b.dtype != a.dtype))
        or (not a.dtype.is_floating_point and not isinstance(alpha, int))
    ):
        return _legacy_add(a, b, alpha=alpha)
    shape = (
        tuple(a.shape) if scalar else tuple(torch.broadcast_shapes(a.shape, b.shape))
    )
    av = a.broadcast_to(shape)
    bv = None if scalar else b.broadcast_to(shape)
    size = math.prod(shape)
    irregular = not av.is_contiguous() or (bv is not None and not bv.is_contiguous())
    if size > 8388608 and av.is_contiguous():
        return _legacy_add(a, b, alpha=alpha)
    if (
        size > 128
        and not (
            a.dtype == torch.float32 and 2097152 <= size <= 8388608 and not irregular
        )
        and (
            size < 262144
            or (size % 256 == 0 and not irregular and a.dtype != torch.int64)
        )
    ):
        return _legacy_add(a, b, alpha=alpha)
    order = sorted(
        range(len(shape)),
        key=lambda d: (
            av.stride(d)
            if av.stride(d) > 0
            else (bv.stride(d) if bv is not None and bv.stride(d) > 0 else 0)
        ),
        reverse=True,
    )
    strides = [0] * len(shape)
    dense_size = 1
    for d in reversed(order):
        strides[d] = dense_size
        dense_size *= shape[d]
    out = torch.empty_strided(shape, tuple(strides), dtype=a.dtype, device=a.device)
    if not size:
        return out
    axes = [d for d in order if shape[d] > 1]
    sh = tuple(shape[d] for d in axes)
    ast = tuple(av.stride(d) for d in axes)
    bst = () if scalar else tuple(bv.stride(d) for d in axes)
    ost = tuple(strides[d] for d in axes)
    aline = ast == ost
    bline = not scalar and bst == ost
    transpose = (
        len(shape) >= 2
        and av.stride(-2) == 1
        and av.stride(-1) >= shape[-2]
        and bv is not None
        and bv.stride(-1) == 1
    )
    with torch_device_fn.device(a.device):
        if transpose and 2097152 <= size <= 8388608:
            _ordered_tile[
                (
                    triton.cdiv(shape[-2], 128),
                    triton.cdiv(shape[-1], 16),
                    math.prod(shape[:-2]),
                )
            ](
                a,
                b,
                out,
                *shape[-2:],
                shape[:-2],
                tuple(av.stride()[:-2]),
                tuple(bv.stride()[:-2]),
                tuple(strides[:-2]),
                *av.stride()[-2:],
                *bv.stride()[-2:],
                *strides[-2:],
                alpha,
                128,
                16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif (
            not scalar
            and axes
            and a.dtype == torch.float32
            and not transpose
            and ast[-1] == bst[-1] == 1
            and len(axes) > 1
            and (not aline or not bline)
            and 64 <= sh[-1] <= 128
        ):
            # Short rows let one program reuse broadcast data across many rows.
            rows = math.prod(sh[:-1])
            cols = sh[-1]
            small_period = all(st == 0 for st in bst[:-2]) and sh[-2] <= 16
            tm, tn = (16, 128) if small_period else (4, 256)
            _ordered_matrix[(triton.cdiv(rows, tm), triton.cdiv(cols, tn))](
                a,
                b,
                out,
                rows,
                cols,
                sh[:-1],
                ast[:-1],
                bst[:-1],
                ast[-1],
                bst[-1],
                alpha,
                tm,
                tn,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif (
            not scalar
            and axes
            and a.dtype == torch.float32
            and not transpose
            and ast[-1] == bst[-1] == 1
            and not aline
            and size < 1500000
        ):
            _ordered_row[(math.prod(sh[:-1]), triton.cdiv(sh[-1], 1024))](
                a,
                b,
                out,
                sh[-1],
                sh[:-1],
                ast[:-1],
                bst[:-1],
                ast[-1],
                bst[-1],
                alpha,
                1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            if size <= 128:
                block, warps = 64, 1
            elif transpose:
                block, warps = (4096, 8) if size < 2097152 else (1024, 4)
            elif a.dtype == torch.int64:
                block, warps = 256, 4
            elif size >= 2000000 and aline and bline:
                block, warps = 512, 8
            elif (
                a.dtype == torch.float32
                and ast
                and (ast[-1] != 0 and not bline)
                and (not aline or (bst and bst[-1] == 1 and sh[-1] > 128))
            ):
                block, warps = 512, 8
            elif a.dtype == torch.float32 and aline and bline:
                block, warps = 2048, 4
            else:
                block, warps = 1024, 4
            _ordered_flat[(triton.cdiv(size, block),)](
                a,
                b,
                out,
                size,
                sh,
                ast,
                bst,
                aline,
                bline,
                scalar,
                alpha,
                block,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return out
