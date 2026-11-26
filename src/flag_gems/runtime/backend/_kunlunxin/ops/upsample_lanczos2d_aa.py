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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.upsample_lanczos2d_aa import (
    _reciprocal_scale,
    _support_and_taps,
    _validate,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# The tap loops are unrolled at trace time; the tap counts stay small, so the
# unrolled body stays within the per-program footprint.


@triton.jit
def _sinc_pi(x):
    """sin(pi*x)/(pi*x).

    Small arguments are evaluated by series so the ratio stays exact as x -> 0.
    """
    abs_x = tl.abs(x)
    pix = x * 3.141592653589793
    q = pix * pix
    series = 1.0 - q * (1.0 / 6.0) + q * q * (1.0 / 120.0)
    ratio = tl.sin(pix) / pix
    return tl.where(abs_x < 0.05, series, tl.where(abs_x == 0.0, 1.0, ratio))


@triton.jit
def _lanczos3(x):
    abs_x = tl.abs(x)
    return tl.where(abs_x < 3.0, _sinc_pi(x) * _sinc_pi(x / 3.0), 0.0)


@triton.jit
def _lanczos_weights_kernel(
    weights,
    index_mins,
    index_sizes,
    input_size,
    output_size,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    center = SCALE * (offsets.to(compute_dtype) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
    index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_size)
    index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)

    # One 2-D tile of (row, tap) instead of a per-tap loop: the tap axis is a
    # constexpr-bounded arange, so nothing is unrolled and the per-program
    # footprint stays flat in MAX_TAPS.
    TAPS_BLOCK: tl.constexpr = triton.next_power_of_2(MAX_TAPS)
    taps = tl.arange(0, TAPS_BLOCK)
    tap_index = taps[None, :]
    tap_lane = taps < MAX_TAPS
    tap_mask = mask[:, None] & tap_lane[None, :] & (tap_index < index_size[:, None])
    distance = (tap_index + index_min[:, None]).to(compute_dtype) - center[:, None]
    weight = tl.where(tap_mask, _lanczos3((distance + 0.5) * INVSCALE), 0.0)

    total_weight = tl.sum(weight, axis=1)
    total_weight = tl.where(total_weight != 0.0, total_weight, 1.0)
    tl.store(
        weights + offsets[:, None] * MAX_TAPS + tap_index,
        weight / total_weight[:, None],
        mask=mask[:, None] & tap_lane[None, :],
    )
    tl.store(index_mins + offsets, index_min, mask=mask)
    tl.store(index_sizes + offsets, index_size, mask=mask)


@triton.jit
def _lanczos_horizontal_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_w,
    output_w,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    row = offsets // output_w
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    if PRECOMPUTED:
        index_min = tl.load(index_mins + output_x, mask=mask).to(tl.int32)
        index_size = tl.load(index_sizes + output_x, mask=mask).to(tl.int32)
    else:
        center = SCALE * (output_x.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int32), 0)
        index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int32), input_w)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    rows_in = (numel + output_w - 1) // output_w
    for tap in tl.static_range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + output_x * MAX_TAPS + tap,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        # Clamped address and masked sample: a tap that is switched off must not
        # contribute anything, even if the load returns an unusable lane.
        sample_col = tl.minimum(index_min + tap, input_w - 1)
        sample_row = tl.minimum(row, rows_in - 1)
        sample = tl.load(
            input + sample_row * input_w + sample_col,
            mask=tap_mask,
            other=0.0,
        ).to(compute_dtype)
        weight = tl.where(tap_mask, weight, 0.0)
        value += tl.where(tap_mask, sample * weight, 0.0)
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lanczos_vertical_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_h,
    output_h,
    output_w,
    SCALE: tl.constexpr,
    SUPPORT: tl.constexpr,
    INVSCALE: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    PRECOMPUTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    output_y = (offsets // output_w) % output_h
    nc = offsets // (output_h * output_w)
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    if PRECOMPUTED:
        index_min = tl.load(index_mins + output_y, mask=mask).to(tl.int32)
        index_size = tl.load(index_sizes + output_y, mask=mask).to(tl.int32)
    else:
        center = SCALE * (output_y.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int32), 0)
        index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int32), input_h)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    nc_in = numel // (output_h * output_w)
    for tap in tl.static_range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + output_y * MAX_TAPS + tap,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        # Clamped address and masked sample: a tap that is switched off must not
        # contribute anything, even if the load returns an unusable lane.
        sample_row = tl.minimum(index_min + tap, input_h - 1)
        input_offset = (
            tl.minimum(nc, nc_in - 1) * input_h + sample_row
        ) * output_w + output_x
        sample = tl.load(input + input_offset, mask=tap_mask, other=0.0).to(
            compute_dtype
        )
        weight = tl.where(tap_mask, weight, 0.0)
        value += tl.where(tap_mask, sample * weight, 0.0)
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


# ---------------------------------------------------------------------------
# Dense weight-matrix path.
#
# Both passes of the separable filter are the same contraction a dense weight
# matrix expresses: out[o] = sum_i in[i] * W[o, i].  Materialising W (one row
# per output coordinate) turns every access into a contiguous tile load and the
# contraction into one matrix multiply.
#
# Addresses are never clamped: a clamped index makes the whole tile load lower
# to per-lane gathers and costs three orders of magnitude.  Instead the tile
# geometry is chosen so every access is in range by construction:
#   * the contraction axis is split into power-of-two chunks covering exactly
#     the input extent, so no lane reads past the input,
#   * the weight tile is padded up to a power of two on both axes and the
#     padding rows/columns are zero, so extra lanes contribute nothing,
#   * the intermediate and output buffers are padded to the tile shape, so
#     every store is unmasked and in range, and the result is narrowed on return.
# The tile lane axis is kept at 64 or more: narrower tiles lose stores.
#
# The weight tiles depend only on the shape and the scale, so they are built
# once and reused; the filtered run then costs one allocation and one launch.
# ---------------------------------------------------------------------------

_DENSE_MAX_ELEMS = 1 << 16
_DENSE_MIN_LANES = 64
# Small inputs still win on the dense path once the tile geometry is valid.
_DENSE_MIN_INPUT = 1 << 10
# One program per (n, c) plane only pays off when there are enough planes.
_DENSE_MIN_PLANES = 16
# bfloat16 keeps the tap path on this target.
_DENSE_DTYPES = (torch.float16, torch.float32)
# Row blocks in descending preference; the first one dividing the row count wins.
_DENSE_ROW_BLOCKS = (128, 64, 32, 16)
_DENSE_PLAN_CACHE = {}
_DENSE_PLAN_CACHE_LIMIT = 64


def _dense_block(size: int) -> int:
    # The weight tile needs at least 64 lanes; narrower tiles store incompletely.
    return max(_DENSE_MIN_LANES, triton.next_power_of_2(size))


def _dense_chunks(size: int) -> list:
    """Greedy power-of-two split of the contraction extent (arange needs pow2)."""
    parts = []
    while size > 0:
        part = 1 << (size.bit_length() - 1)
        parts.append(part)
        size -= part
    return parts


def _dense_row_block(rows: int):
    for block in _DENSE_ROW_BLOCKS:
        if rows % block == 0:
            return block
    return None


def _dense_axis_ok(input_size: int, output_size: int) -> bool:
    """One dense weight tile for this axis pair must fit and split into <= 2 chunks."""
    return (
        input_size % 16 == 0
        and input_size >= 16
        and len(_dense_chunks(input_size)) <= 2
        and _dense_block(input_size) * _dense_block(output_size) <= _DENSE_MAX_ELEMS
    )


@triton.jit
def _lanczos_dense_weights_kernel(
    weights,
    in_size,
    out_size,
    SCALE,
    SUPPORT,
    INVSCALE,
    MAX_TAPS: tl.constexpr,
    IB: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    """Write the normalised weights of output coordinate `o` into a dense tile.

    Element (in index i, output o) lands at `i * row_stride + o * col_stride`, so
    one launch serves both the row-major [in, out] and [out, in] layouts.  The
    buffer is zero-initialised and `IB` is padded past `in_size`, so lanes outside
    the window store their zero.
    """
    o = tl.program_id(0)
    center = SCALE * (o.to(tl.float32) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int32), 0)
    index_max = tl.minimum((center + SUPPORT + 0.5).to(tl.int32), in_size)
    index_size = tl.minimum(tl.maximum(index_max - index_min, 0), MAX_TAPS)
    ii = tl.arange(0, IB)
    window = (ii >= index_min) & (ii < index_min + index_size) & (ii < in_size)
    distance = (ii.to(tl.float32) - center + 0.5) * INVSCALE
    weight = tl.where(window, _lanczos3(distance), 0.0)
    total = tl.sum(weight, axis=0)
    total = tl.where(total != 0.0, total, 1.0)
    tl.store(weights + ii * ROW_STRIDE + o * COL_STRIDE, weight / total)


@triton.jit
def _lanczos_h_dot_kernel(
    x,
    wmat,
    y,
    R: tl.constexpr,
    IN_W: tl.constexpr,
    BN: tl.constexpr,
    C0: tl.constexpr,
    S0: tl.constexpr,
    C1: tl.constexpr,
    S1: tl.constexpr,
    HAS2: tl.constexpr,
    IS_F16: tl.constexpr,
):
    """C[R, BN] = x[R, IN_W] @ W[IN_W, BN], one launch over all input rows."""
    pid = tl.program_id(0)
    rr = pid * R + tl.arange(0, R)
    nn = tl.arange(0, BN)
    k0 = C0 + tl.arange(0, S0)
    if IS_F16:
        a = tl.load(x + rr[:, None] * IN_W + k0[None, :])
        b = tl.load(wmat + k0[:, None] * BN + nn[None, :])
        acc = tl.dot(a, b)
    else:
        a = tl.load(x + rr[:, None] * IN_W + k0[None, :]).to(tl.float32)
        b = tl.load(wmat + k0[:, None] * BN + nn[None, :])
        acc = tl.dot(a, b, input_precision="ieee")
    if HAS2:
        k1 = C1 + tl.arange(0, S1)
        if IS_F16:
            a1 = tl.load(x + rr[:, None] * IN_W + k1[None, :])
            b1 = tl.load(wmat + k1[:, None] * BN + nn[None, :])
            acc += tl.dot(a1, b1)
        else:
            a1 = tl.load(x + rr[:, None] * IN_W + k1[None, :]).to(tl.float32)
            b1 = tl.load(wmat + k1[:, None] * BN + nn[None, :])
            acc += tl.dot(a1, b1, input_precision="ieee")
    tl.store(y + rr[:, None] * BN + nn[None, :], acc.to(y.dtype.element_ty))


@triton.jit
def _lanczos_v_dot_kernel(
    mid,
    wmat,
    y,
    IN_H: tl.constexpr,
    BW: tl.constexpr,
    KH_STRIDE: tl.constexpr,
    M: tl.constexpr,
    C0: tl.constexpr,
    S0: tl.constexpr,
    C1: tl.constexpr,
    S1: tl.constexpr,
    HAS2: tl.constexpr,
    IS_F16: tl.constexpr,
):
    """C[M, BW] = W[M, IN_H] @ mid[IN_H, BW] for one (n, c) plane."""
    plane = tl.program_id(0)
    rr = tl.arange(0, M)
    cc = tl.arange(0, BW)
    k0 = C0 + tl.arange(0, S0)
    if IS_F16:
        b = tl.load(wmat + rr[:, None] * KH_STRIDE + k0[None, :])
        a = tl.load(mid + plane * IN_H * BW + k0[:, None] * BW + cc[None, :])
        acc = tl.dot(b, a)
    else:
        b = tl.load(wmat + rr[:, None] * KH_STRIDE + k0[None, :])
        a = tl.load(mid + plane * IN_H * BW + k0[:, None] * BW + cc[None, :]).to(
            tl.float32
        )
        acc = tl.dot(b, a, input_precision="ieee")
    if HAS2:
        k1 = C1 + tl.arange(0, S1)
        if IS_F16:
            b1 = tl.load(wmat + rr[:, None] * KH_STRIDE + k1[None, :])
            a1 = tl.load(mid + plane * IN_H * BW + k1[:, None] * BW + cc[None, :])
            acc += tl.dot(b1, a1)
        else:
            b1 = tl.load(wmat + rr[:, None] * KH_STRIDE + k1[None, :])
            a1 = tl.load(mid + plane * IN_H * BW + k1[:, None] * BW + cc[None, :]).to(
                tl.float32
            )
            acc += tl.dot(b1, a1, input_precision="ieee")
    tl.store(
        y + plane * M * BW + rr[:, None] * BW + cc[None, :],
        acc.to(y.dtype.element_ty),
    )


@triton.jit
def _lanczos_fused_kernel(
    x,
    wmat_h,
    wmat_w,
    y,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    BN: tl.constexpr,
    M: tl.constexpr,
    KH_STRIDE: tl.constexpr,
    C0: tl.constexpr,
    S0: tl.constexpr,
    C1: tl.constexpr,
    S1: tl.constexpr,
    HAS2: tl.constexpr,
    IS_F16: tl.constexpr,
):
    """Both contractions for one (n, c) plane, with no intermediate buffer.

    out[M, BN] = W_h[M, IN_H] @ (x[plane][IN_H, IN_W] @ W_w[IN_W, BN]).
    """
    plane = tl.program_id(0)
    rr = tl.arange(0, M)
    nn = tl.arange(0, BN)
    ii = tl.arange(0, IN_W)
    base = plane * IN_H
    k0 = C0 + tl.arange(0, S0)
    if IS_F16:
        wh = tl.load(wmat_h + rr[:, None] * KH_STRIDE + k0[None, :])
        a = tl.load(x + (k0[:, None] + base) * IN_W + ii[None, :])
        h = tl.dot(wh, a)
    else:
        wh = tl.load(wmat_h + rr[:, None] * KH_STRIDE + k0[None, :])
        a = tl.load(x + (k0[:, None] + base) * IN_W + ii[None, :]).to(tl.float32)
        h = tl.dot(wh, a, input_precision="ieee")
    if HAS2:
        k1 = C1 + tl.arange(0, S1)
        if IS_F16:
            wh1 = tl.load(wmat_h + rr[:, None] * KH_STRIDE + k1[None, :])
            a1 = tl.load(x + (k1[:, None] + base) * IN_W + ii[None, :])
            h += tl.dot(wh1, a1)
        else:
            wh1 = tl.load(wmat_h + rr[:, None] * KH_STRIDE + k1[None, :])
            a1 = tl.load(x + (k1[:, None] + base) * IN_W + ii[None, :]).to(tl.float32)
            h += tl.dot(wh1, a1, input_precision="ieee")
    if IS_F16:
        # The half-precision pass rounds its intermediate to the compute dtype,
        # which is what the reference composition does between the two passes.
        ww = tl.load(wmat_w + ii[:, None] * BN + nn[None, :])
        acc = tl.dot(h.to(y.dtype.element_ty), ww)
    else:
        ww = tl.load(wmat_w + ii[:, None] * BN + nn[None, :])
        acc = tl.dot(h, ww, input_precision="ieee")
    tl.store(
        y + plane * M * BN + rr[:, None] * BN + nn[None, :],
        acc.to(y.dtype.element_ty),
    )


def _dense_plan(input: torch.Tensor, output_size, align_corners, scales_h, scales_w):
    """Geometry plus weight tiles for one (dtype, shape, output, scale) combination."""
    key = (
        input.dtype,
        tuple(input.shape),
        tuple(output_size),
        bool(align_corners),
        scales_h,
        scales_w,
        input.device,
    )
    if key in _DENSE_PLAN_CACHE:
        return _DENSE_PLAN_CACHE[key]

    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    rows = n * c * input_h
    row_block = _dense_row_block(rows)
    plan = None
    if (
        input.dtype in _DENSE_DTYPES
        and input.numel() >= _DENSE_MIN_INPUT
        and row_block is not None
        and _dense_axis_ok(input_w, output_w)
        and _dense_axis_ok(input_h, output_h)
    ):
        scale_w = _reciprocal_scale(input_w, output_w, align_corners, scales_w)
        scale_h = _reciprocal_scale(input_h, output_h, align_corners, scales_h)
        support_w, taps_w = _support_and_taps(scale_w)
        support_h, taps_h = _support_and_taps(scale_h)
        invscale_w = 1.0 / scale_w if scale_w >= 1.0 else 1.0
        invscale_h = 1.0 / scale_h if scale_h >= 1.0 else 1.0
        w_dtype = torch.float16 if input.dtype == torch.float16 else torch.float32
        kpad_w = _dense_block(input_w)
        bn = _dense_block(output_w)
        kpad_h = _dense_block(input_h)
        m_h = _dense_block(output_h)
        kw_chunks = _dense_chunks(input_w)
        kh_chunks = _dense_chunks(input_h)
        with torch_device_fn.device(input.device):
            weights_w = torch.zeros((kpad_w, bn), device=input.device, dtype=w_dtype)
            _lanczos_dense_weights_kernel[(output_w,)](
                weights_w,
                input_w,
                output_w,
                scale_w,
                support_w,
                invscale_w,
                taps_w,
                kpad_w,
                ROW_STRIDE=bn,
                COL_STRIDE=1,
            )
            weights_h = torch.zeros((m_h, kpad_h), device=input.device, dtype=w_dtype)
            _lanczos_dense_weights_kernel[(output_h,)](
                weights_h,
                input_h,
                output_h,
                scale_h,
                support_h,
                invscale_h,
                taps_h,
                kpad_h,
                ROW_STRIDE=1,
                COL_STRIDE=kpad_h,
            )
        plan = (
            rows,
            row_block,
            bn,
            m_h,
            kpad_h,
            kw_chunks,
            kh_chunks,
            weights_w,
            weights_h,
            # Both contractions in one program need the whole input width in a
            # single tile, and one program per plane needs enough planes.
            len(kw_chunks) == 1 and n * c >= _DENSE_MIN_PLANES,
        )
    if len(_DENSE_PLAN_CACHE) >= _DENSE_PLAN_CACHE_LIMIT:
        _DENSE_PLAN_CACHE.clear()
    _DENSE_PLAN_CACHE[key] = plan
    return plan


def _lanczos_tap_contiguous(
    input: torch.Tensor,
    output_size,
    align_corners: bool,
    scales_h,
    scales_w,
) -> torch.Tensor:
    """Two-pass tap implementation, the fallback for geometries the dense plan declines."""
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    if input_h == output_h and input_w == output_w:
        return input.clone()

    scale_w = _reciprocal_scale(input_w, output_w, align_corners, scales_w)
    scale_h = _reciprocal_scale(input_h, output_h, align_corners, scales_h)
    support_w, taps_w = _support_and_taps(scale_w)
    support_h, taps_h = _support_and_taps(scale_h)
    is_uint8 = input.dtype == torch.uint8
    is_fp64 = input.dtype == torch.float64
    precompute_weights = n * c >= 128
    block_size = 256

    with torch_device_fn.device(input.device):
        if input_w != output_w:
            if precompute_weights:
                weights_w, index_mins_w, index_sizes_w = _make_lanczos_weights(
                    input_w,
                    output_w,
                    scale_w,
                    support_w,
                    taps_w,
                    input.device,
                    is_fp64,
                )
            else:
                weights_w = index_mins_w = index_sizes_w = input
            horizontal = torch.empty(
                (n, c, input_h, output_w), device=input.device, dtype=input.dtype
            )
            horizontal_numel = horizontal.numel()
            _lanczos_horizontal_kernel[(triton.cdiv(horizontal_numel, block_size),)](
                input,
                horizontal,
                weights_w,
                index_mins_w,
                index_sizes_w,
                horizontal_numel,
                input_w,
                output_w,
                SCALE=scale_w,
                SUPPORT=support_w,
                INVSCALE=1.0 / scale_w if scale_w >= 1.0 else 1.0,
                MAX_TAPS=taps_w,
                IS_UINT8=is_uint8,
                IS_FP64=is_fp64,
                PRECOMPUTED=precompute_weights,
                BLOCK_SIZE=block_size,
            )
        else:
            horizontal = input

        if input_h != output_h:
            if precompute_weights:
                weights_h, index_mins_h, index_sizes_h = _make_lanczos_weights(
                    input_h,
                    output_h,
                    scale_h,
                    support_h,
                    taps_h,
                    input.device,
                    is_fp64,
                )
            else:
                weights_h = index_mins_h = index_sizes_h = input
            output = torch.empty(
                (n, c, output_h, output_w), device=input.device, dtype=input.dtype
            )
            output_numel = output.numel()
            _lanczos_vertical_kernel[(triton.cdiv(output_numel, block_size),)](
                horizontal,
                output,
                weights_h,
                index_mins_h,
                index_sizes_h,
                output_numel,
                input_h,
                output_h,
                output_w,
                SCALE=scale_h,
                SUPPORT=support_h,
                INVSCALE=1.0 / scale_h if scale_h >= 1.0 else 1.0,
                MAX_TAPS=taps_h,
                IS_UINT8=is_uint8,
                IS_FP64=is_fp64,
                PRECOMPUTED=precompute_weights,
                BLOCK_SIZE=block_size,
            )
        else:
            output = horizontal
    return output


def _make_lanczos_weights(
    input_size: int,
    output_size: int,
    scale: float,
    support: float,
    taps: int,
    device: torch.device,
    is_fp64: bool,
):
    weights = torch.empty(
        (output_size, taps),
        device=device,
        dtype=torch.float64 if is_fp64 else torch.float32,
    )
    index_mins = torch.empty(output_size, device=device, dtype=torch.int64)
    index_sizes = torch.empty(output_size, device=device, dtype=torch.int64)
    block_size = 128
    _lanczos_weights_kernel[(triton.cdiv(output_size, block_size),)](
        weights,
        index_mins,
        index_sizes,
        input_size,
        output_size,
        SCALE=scale,
        SUPPORT=support,
        INVSCALE=1.0 / scale if scale >= 1.0 else 1.0,
        MAX_TAPS=taps,
        IS_FP64=is_fp64,
        BLOCK_SIZE=block_size,
    )
    return weights, index_mins, index_sizes


def _lanczos_contiguous(
    input: torch.Tensor,
    output_size,
    align_corners: bool,
    scales_h,
    scales_w,
) -> torch.Tensor:
    plan = _dense_plan(input, output_size, align_corners, scales_h, scales_w)
    if plan is None:
        return _lanczos_tap_contiguous(
            input, output_size, align_corners, scales_h, scales_w
        )
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    (
        rows,
        row_block,
        bn,
        m_h,
        kpad_h,
        kw_chunks,
        kh_chunks,
        weights_w,
        weights_h,
        fused,
    ) = plan
    is_f16 = input.dtype == torch.float16
    kw_c1 = kw_chunks[0] if len(kw_chunks) > 1 else 0
    kw_s1 = kw_chunks[1] if len(kw_chunks) > 1 else 1
    kh_c1 = kh_chunks[0] if len(kh_chunks) > 1 else 0
    kh_s1 = kh_chunks[1] if len(kh_chunks) > 1 else 1
    planes = n * c
    dtype = input.dtype

    with torch_device_fn.device(input.device):
        if fused:
            padded = torch.empty((planes, m_h, bn), device=input.device, dtype=dtype)
            _lanczos_fused_kernel[(planes,)](
                input,
                weights_h,
                weights_w,
                padded,
                IN_H=input_h,
                IN_W=input_w,
                BN=bn,
                M=m_h,
                KH_STRIDE=kpad_h,
                C0=0,
                S0=kh_chunks[0],
                C1=kh_c1,
                S1=kh_s1,
                HAS2=kh_c1 != 0,
                IS_F16=is_f16,
            )
        else:
            buf = torch.empty(
                (rows + planes * m_h) * bn, device=input.device, dtype=dtype
            )
            mid = buf[: rows * bn].view(rows, bn)
            padded = buf[rows * bn :].view(planes, m_h, bn)
            _lanczos_h_dot_kernel[(rows // row_block,)](
                input,
                weights_w,
                mid,
                R=row_block,
                IN_W=input_w,
                BN=bn,
                C0=0,
                S0=kw_chunks[0],
                C1=kw_c1,
                S1=kw_s1,
                HAS2=kw_c1 != 0,
                IS_F16=is_f16,
            )
            _lanczos_v_dot_kernel[(planes,)](
                mid,
                weights_h,
                padded,
                IN_H=input_h,
                BW=bn,
                KH_STRIDE=kpad_h,
                M=m_h,
                C0=0,
                S0=kh_chunks[0],
                C1=kh_c1,
                S1=kh_s1,
                HAS2=kh_c1 != 0,
                IS_F16=is_f16,
            )
    output = padded.view(n, c, m_h, bn)
    if m_h == output_h and bn == output_w:
        return output
    return output[:, :, :output_h, :output_w].contiguous()


def _upsample_lanczos2d_aa(
    input: torch.Tensor,
    output_size,
    align_corners: bool = False,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_LANCZOS2D_AA")
    output_size = _validate(input, output_size)
    channels_last = input.is_contiguous(memory_format=torch.channels_last)
    if input.shape[0] == 0:
        output = torch.empty(
            (*input.shape[:2], *output_size), device=input.device, dtype=input.dtype
        )
    else:
        output = _lanczos_contiguous(
            input.contiguous(),
            output_size,
            align_corners,
            scales_h,
            scales_w,
        )
    if channels_last:
        output = output.contiguous(memory_format=torch.channels_last)
    return output


def _upsample_lanczos2d_aa_out(
    input: torch.Tensor,
    output_size,
    align_corners: bool = False,
    scales_h=None,
    scales_w=None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_LANCZOS2D_AA_OUT")
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor on {input.device}, but got {out.device}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype}"
        )
    result = _upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _upsample_lanczos2d_aa_vec(
    input: torch.Tensor,
    output_size,
    align_corners: bool,
    scale_factors,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN _UPSAMPLE_LANCZOS2D_AA_VEC")
    if (output_size is None) == (scale_factors is None):
        raise RuntimeError("Must specify exactly one of output_size and scale_factors")
    if output_size is not None:
        return _upsample_lanczos2d_aa(input, tuple(output_size), align_corners)
    if len(scale_factors) != 2:
        raise RuntimeError("scale_factors must have two elements")
    output_size = (
        int(input.shape[-2] * scale_factors[0]),
        int(input.shape[-1] * scale_factors[1]),
    )
    return _upsample_lanczos2d_aa(
        input,
        output_size,
        align_corners,
        float(scale_factors[0]),
        float(scale_factors[1]),
    )
