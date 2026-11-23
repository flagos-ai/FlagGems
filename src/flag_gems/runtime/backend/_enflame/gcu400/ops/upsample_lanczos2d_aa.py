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
import math
from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)
device = device.name
sinpi = tl_extra_shim.sinpi

# ---------------------------------------------------------------------------
# gcu400 dense dot path (rewritten 2026-11-18, same rationale as _upsample_lanczos2d_aa_backward):
#   Separable Lanczos-AA forward pass = two GEMMs (in @ Wx^T along W, Wy @ · along H). The original two-pass
#   per-lane gather form (13-tap weight table / sample element-wise gather + flat element-wise kernel)
#   serializes gathers per lane on GCU400, pushing large-shape downsampling to 1.9-2.5ms (native einsum
#   with two GEMMs takes only 51-94us). Replaced with a single kernel using a dense-weight-matrix tl.dot chain:
#     tmp[BIH,BOW] = dot(in[BIH,BKW], wxT[BKW,BOW])   (contract along W)
#     acc[BOH,BOW] = dot(wy[BOH,BIH], tmp[BIH,BOW])   (contract along H)
#   All addressing is affine; the K dimension (H_in/W_in) is unrolled with tl.static_range (a gcu400 dynamic range loop
#   carrying an fp32 tl.dot accumulator miscompiles; see the comments in gcu400/ops/mm.py).
# Finalized by probing (V6W1): BIH/BKW/BOH cap=128, BOW cap=64, num_warps=1.
# ---------------------------------------------------------------------------
_DOT_NUM_SPC = 48  # GCU400 L600 fixed grid (number of SPs)
_DOT_MAX_K_BLOCKS = 16
_DOT_WEIGHT_CACHE_MAX = 128
_DOT_FP16_MIN_ELEMS = 65536
_DOT_BIH_CAP = 128
_DOT_BOH_CAP = 128
_DOT_BOW_CAP = 64
_DOT_BKW_CAP = 128
_DOT_DSM_BUDGET = 917504 // 2  # Libra per-program DSM limit (448KB)


@triton.jit
def _lanczos_aa_filter(x):
    pix = math.pi * x
    sinc = tl.where(x == 0.0, 1.0, sinpi(x) / pix)
    sinc_three = tl.where(x == 0.0, 1.0, sinpi(x / 3.0) / (pix / 3.0))
    return tl.where(x < 3.0, sinc * sinc_three, 0.0)


@triton.jit
def _f2i(x):
    _LO: tl.constexpr = -2147483648.0
    _HI: tl.constexpr = 2147483520.0
    return tl.minimum(tl.maximum(x, _LO), _HI).to(tl.int32)


@triton.jit
def _dense_weights_kernel(
    w_ptr,
    out_size,
    in_size,
    scale,
    support,
    invscale,
    BLOCK_I: tl.constexpr,
    I_BLOCKS: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    """Computes dense Lanczos-AA weight rows [out_size, in_size] per output row (zero outside the window)."""
    pid = tl.program_id(0)
    for oi in range(pid, out_size, NUM_SPC):
        center = scale * (oi + 0.5)
        xmin = tl.maximum(_f2i(center - support + 0.5), 0)
        xmax = tl.minimum(_f2i(center + support + 0.5), in_size)
        total = 0.0
        for i0 in tl.static_range(I_BLOCKS):
            ii = i0 * BLOCK_I + tl.arange(0, BLOCK_I)
            arg = tl.abs((ii.to(tl.float32) - center + 0.5) * invscale)
            raw = _lanczos_aa_filter(arg)
            w = tl.where((ii >= xmin) & (ii < xmax) & (arg < 3.0), raw, 0.0)
            total += tl.sum(w)
        for i0 in tl.static_range(I_BLOCKS):
            ii = i0 * BLOCK_I + tl.arange(0, BLOCK_I)
            in_m = ii < in_size
            arg = tl.abs((ii.to(tl.float32) - center + 0.5) * invscale)
            raw = _lanczos_aa_filter(arg)
            w = tl.where((ii >= xmin) & (ii < xmax) & (arg < 3.0), raw, 0.0)
            w = tl.where(total != 0.0, w / total, 0.0)
            tl.store(
                w_ptr + oi.to(tl.int64) * in_size + ii,
                w.to(w_ptr.dtype.element_ty),
                mask=in_m,
            )


@triton.jit
def _fused_dot_forward_kernel(
    in_ptr,
    out_ptr,
    wy_ptr,
    wx_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    T_OH,
    T_OW,
    TOTAL_WORK,
    BIH: tl.constexpr,
    BOH: tl.constexpr,
    BOW: tl.constexpr,
    BKW: tl.constexpr,
    KH_BLOCKS: tl.constexpr,
    KW_BLOCKS: tl.constexpr,
    HAS_H: tl.constexpr,
    HAS_W: tl.constexpr,
    MODE: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    """out[nc,oh,ow] = sum_ih sum_ih wy[oh,ih] * wx[ow,iw] * in[nc,ih,iw].

    Two-stage dot chain in one kernel: tmp = in @ Wx^T (along W), acc = Wy @ tmp (along H).
    MODE=0: all-fp32 dot (exact, input/weights widened to fp32 in-kernel);
    MODE=1: dot uses tensor cores in the input dtype and tmp is cast back before dot2.
    All accesses are affine: Wy / Wx are dense weight matrices (zeroed outside the window), the
    transposed weight tile uses 2-D loads with row-stride=1 / col-stride=W_in, and in/out are
    """
    pid = tl.program_id(0)
    for wid in range(pid, TOTAL_WORK, NUM_SPC):
        nc = wid // (T_OH * T_OW)
        r = wid - nc * (T_OH * T_OW)
        oh_t = r // T_OW
        ow_t = r - oh_t * T_OW
        oh = oh_t * BOH + tl.arange(0, BOH)
        ow = ow_t * BOW + tl.arange(0, BOW)
        oh_m = oh < H_out
        ow_m = ow < W_out
        nc64 = nc.to(tl.int64)
        acc = tl.zeros((BOH, BOW), dtype=tl.float32)
        if HAS_H:
            for ihb in tl.static_range(KH_BLOCKS):
                ih = ihb * BIH + tl.arange(0, BIH)
                ih_m = ih < H_in
                if HAS_W:
                    tmp = tl.zeros((BIH, BOW), dtype=tl.float32)
                    for iwb in tl.static_range(KW_BLOCKS):
                        iw = iwb * BKW + tl.arange(0, BKW)
                        iw_m = iw < W_in
                        a = tl.load(
                            in_ptr + (nc64 * H_in + ih)[:, None] * W_in + iw[None, :],
                            mask=ih_m[:, None] & iw_m[None, :],
                            other=0.0,
                        )
                        b = tl.load(
                            wx_ptr + ow[None, :].to(tl.int64) * W_in + iw[:, None],
                            mask=iw_m[:, None] & ow_m[None, :],
                            other=0.0,
                        )
                        if MODE == 0:
                            tmp += tl.dot(
                                a.to(tl.float32),
                                b.to(tl.float32),
                                out_dtype=tl.float32,
                                allow_tf32=False,
                            )
                        else:
                            tmp += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
                else:
                    tmp = tl.load(
                        in_ptr + (nc64 * H_in + ih)[:, None] * W_in + ow[None, :],
                        mask=ih_m[:, None] & ow_m[None, :],
                        other=0.0,
                    ).to(tl.float32)
                wy_t = tl.load(
                    wy_ptr + oh[:, None].to(tl.int64) * H_in + ih[None, :],
                    mask=oh_m[:, None] & ih_m[None, :],
                    other=0.0,
                )
                if MODE == 1:
                    acc += tl.dot(
                        wy_t, tmp.to(wy_t.dtype), out_dtype=tl.float32, allow_tf32=False
                    )
                else:
                    acc += tl.dot(
                        wy_t.to(tl.float32),
                        tmp,
                        out_dtype=tl.float32,
                        allow_tf32=False,
                    )
        else:
            # H_out == H_in: row pass-through (the BOH tile also serves as the input row tile)
            if HAS_W:
                for iwb in tl.static_range(KW_BLOCKS):
                    iw = iwb * BKW + tl.arange(0, BKW)
                    iw_m = iw < W_in
                    a = tl.load(
                        in_ptr + (nc64 * H_in + oh)[:, None] * W_in + iw[None, :],
                        mask=oh_m[:, None] & iw_m[None, :],
                        other=0.0,
                    )
                    b = tl.load(
                        wx_ptr + ow[None, :].to(tl.int64) * W_in + iw[:, None],
                        mask=iw_m[:, None] & ow_m[None, :],
                        other=0.0,
                    )
                    if MODE == 0:
                        acc += tl.dot(
                            a.to(tl.float32),
                            b.to(tl.float32),
                            out_dtype=tl.float32,
                            allow_tf32=False,
                        )
                    else:
                        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            else:
                acc = tl.load(
                    in_ptr + (nc64 * H_in + oh)[:, None] * W_in + ow[None, :],
                    mask=oh_m[:, None] & ow_m[None, :],
                    other=0.0,
                ).to(tl.float32)
        tl.store(
            out_ptr + (nc64 * H_out + oh)[:, None] * W_out + ow[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=oh_m[:, None] & ow_m[None, :],
        )


@triton.jit
def _lanczos3(x):
    abs_x = tl.abs(x)
    pix = x * 3.141592653589793
    pix_over_three = pix / 3.0
    sinc_x = tl.where(abs_x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_x_over_three = tl.where(
        abs_x == 0.0, 1.0, tl.sin(pix_over_three) / pix_over_three
    )
    return tl.where(abs_x < 3.0, sinc_x * sinc_x_over_three, 0.0)


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
    # GCU: weights are stored transposed as [MAX_TAPS, output_size] so that the
    # per-tap weight loads in the resampling kernels (`tap * output_size + x`)
    # are contiguous along the flattened output index instead of stride-taps
    # gathers (measured 1.68x on the horizontal pass, GCU400).
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    center = SCALE * (offsets.to(compute_dtype) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
    index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_size)
    index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)

    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        weight = tl.where(tap_mask, weight, 0.0)
        tl.store(weights + tap * output_size + offsets, weight, mask=mask)
        total_weight += weight

    total_weight = tl.where(total_weight != 0.0, total_weight, 1.0)
    for tap in range(MAX_TAPS):
        weight = tl.load(weights + tap * output_size + offsets, mask=mask)
        tl.store(
            weights + tap * output_size + offsets,
            weight / total_weight,
            mask=mask,
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
        index_min = tl.load(index_mins + output_x, mask=mask)
        index_size = tl.load(index_sizes + output_x, mask=mask)
    else:
        center = SCALE * (output_x.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
        index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_w)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + tap * output_w + output_x,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        sample = tl.load(
            input + row * input_w + index_min + tap,
            mask=tap_mask,
            other=0.0,
        ).to(compute_dtype)
        weight = tl.where(tap_mask, weight, 0.0)
        value += sample * weight
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
        index_min = tl.load(index_mins + output_y, mask=mask)
        index_size = tl.load(index_sizes + output_y, mask=mask)
    else:
        center = SCALE * (output_y.to(compute_dtype) + 0.5)
        index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int64), 0)
        index_size = tl.minimum((center + SUPPORT + 0.5).to(tl.int64), input_h)
        index_size = tl.minimum(tl.maximum(index_size - index_min, 0), MAX_TAPS)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        if PRECOMPUTED:
            weight = tl.load(
                weights + tap * output_h + output_y,
                mask=tap_mask,
                other=0.0,
            )
        else:
            weight = _lanczos3((tap + index_min - center + 0.5) * INVSCALE)
        input_offset = (nc * input_h + index_min + tap) * output_w + output_x
        sample = tl.load(input + input_offset, mask=tap_mask, other=0.0).to(
            compute_dtype
        )
        weight = tl.where(tap_mask, weight, 0.0)
        value += sample * weight
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def _lanczos_vertical_main_kernel(
    input,
    output,
    weights,
    input_h,
    output_h,
    output_w,
    y_first,
    y_last,
    size_ref,
    K: tl.constexpr,
    B0: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    T_POW2: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # GCU fast path for integer downsample scales: inside the main segment
    # [y_first, y_last) every output row has the full tap window, so
    # index_min = K * y + B0 is affine and the tap weights (and their sum) are
    # identical for all rows. Each program loads a [BLOCK_Y, BLOCK_X] output
    # tile; the per-tap sample load address is a fully affine 2D tile (row
    # stride K * output_w, contiguous along x), which the GCU vectorizes.
    # Border rows (clamped index or shrunken window => different weights and
    # normalization) are handled by _lanczos_vertical_border_kernel.
    pid = tl.program_id(0)
    num_y_tiles = tl.cdiv(y_last - y_first, BLOCK_Y)
    num_x_tiles = tl.cdiv(output_w, BLOCK_X)
    nc = pid // (num_y_tiles * num_x_tiles)
    pid_yx = pid % (num_y_tiles * num_x_tiles)
    pid_y = pid_yx // num_x_tiles
    pid_x = pid_yx % num_x_tiles
    y = y_first + pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    t = tl.arange(0, T_POW2)
    ymask = y < y_last
    xmask = x < output_w
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    row_norm = tl.load(
        weights + t * output_h + y_first, mask=t < size_ref, other=0.0
    ).to(compute_dtype)
    total = tl.sum(row_norm)
    affine_min = K * y + B0
    m2 = ymask[:, None] & xmask[None, :]
    value = tl.zeros((BLOCK_Y, BLOCK_X), dtype=compute_dtype)
    for tap in range(MAX_TAPS):
        w_tap = tl.sum(tl.where(t == tap, row_norm, 0.0))
        sample = tl.load(
            input + (nc * input_h + affine_min + tap)[:, None] * output_w + x[None, :],
            mask=m2,
            other=0.0,
        ).to(compute_dtype)
        value += sample * w_tap
    value = value / tl.where(total != 0.0, total, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(
        output + (nc * output_h + y)[:, None] * output_w + x[None, :],
        value,
        mask=m2,
    )


@triton.jit
def _lanczos_vertical_border_kernel(
    input,
    output,
    weights,
    index_mins,
    index_sizes,
    numel,
    input_h,
    output_h,
    output_w,
    y_first,
    y_last,
    border_count,
    MAX_TAPS: tl.constexpr,
    IS_UINT8: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Covers the border rows y < y_first and y >= y_last where the tap window
    # is clamped/shrunken; flat per-element form identical to the generic
    # vertical kernel except for the border-row index remapping.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    output_x = offsets % output_w
    brow = (offsets // output_w) % border_count
    nc = offsets // (border_count * output_w)
    output_y = tl.where(brow < y_first, brow, brow - y_first + y_last)
    compute_dtype: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    index_min = tl.load(index_mins + output_y, mask=mask)
    index_size = tl.load(index_sizes + output_y, mask=mask)
    total_weight = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    value = tl.zeros((BLOCK_SIZE,), dtype=compute_dtype)
    for tap in range(MAX_TAPS):
        tap_mask = mask & (tap < index_size)
        weight = tl.load(
            weights + tap * output_h + output_y,
            mask=tap_mask,
            other=0.0,
        )
        input_offset = (nc * input_h + index_min + tap) * output_w + output_x
        sample = tl.load(input + input_offset, mask=tap_mask, other=0.0).to(
            compute_dtype
        )
        weight = tl.where(tap_mask, weight, 0.0)
        value += sample * weight
        total_weight += weight

    value /= tl.where(total_weight != 0.0, total_weight, 1.0)
    if IS_UINT8:
        value = tl.floor(tl.minimum(tl.maximum(value, 0.0), 255.0) + 0.5)
    tl.store(
        output + (nc * output_h + output_y) * output_w + output_x, value, mask=mask
    )


def _get_dense_weights(input_size, output_size, align_corners, scale_arg, dtype, dev):
    """Dense [out, in] Lanczos-AA weights (the host caches the device tensor, launching the
    weight kernel only on a cache miss; the bench reference einsum weights are cached too).
    """
    cache = getattr(_get_dense_weights, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        _get_dense_weights._cache = cache
    key = (
        int(input_size),
        int(output_size),
        bool(align_corners),
        None if scale_arg is None else float(scale_arg),
        dtype,
        str(dev),
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    scale = _reciprocal_scale(input_size, output_size, align_corners, scale_arg)
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    weight = torch.zeros((output_size, input_size), dtype=dtype, device=dev)
    BLOCK_I = 128
    _dense_weights_kernel[(_DOT_NUM_SPC,)](
        weight,
        output_size,
        input_size,
        scale,
        support,
        invscale,
        BLOCK_I=BLOCK_I,
        I_BLOCKS=triton.cdiv(input_size, BLOCK_I),
        NUM_SPC=_DOT_NUM_SPC,
        num_warps=4,
    )
    if len(cache) >= _DOT_WEIGHT_CACHE_MAX:
        cache.popitem(last=False)
    cache[key] = weight
    return weight


def _np2_min16(x, cap):
    return min(max(triton.next_power_of_2(max(int(x), 1)), 16), cap)


def _dot_path_eligible(input, output_h, output_w, align_corners):
    if input.device.type != "gcu":
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if align_corners and (output_h == 1 or output_w == 1):
        return False
    input_h, input_w = input.shape[-2:]
    if input_h > _DOT_MAX_K_BLOCKS * _DOT_BIH_CAP:
        return False
    if input_w > _DOT_MAX_K_BLOCKS * _DOT_BKW_CAP:
        return False
    return True


def _dot_forward_impl(
    input,
    output_size,
    align_corners,
    scales_h,
    scales_w,
    out_tensor=None,
):
    """Dense dot path: a single kernel launch performs the separable Lanczos-AA forward."""
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    nc = n * c
    dev = input.device
    has_h = input_h != output_h
    has_w = input_w != output_w
    work_elems = nc * max(input_h * input_w, output_h * output_w)
    use_f16_dot = (
        input.dtype in (torch.float16, torch.bfloat16)
        and work_elems >= _DOT_FP16_MIN_ELEMS
    )
    weight_dtype = input.dtype if use_f16_dot else torch.float32
    mode = 1 if use_f16_dot else 0
    wy = (
        _get_dense_weights(
            input_h, output_h, align_corners, scales_h, weight_dtype, dev
        )
        if has_h
        else input
    )
    wx = (
        _get_dense_weights(
            input_w, output_w, align_corners, scales_w, weight_dtype, dev
        )
        if has_w
        else input
    )
    bih = _np2_min16(input_h, _DOT_BIH_CAP) if has_h else 16
    boh = _np2_min16(output_h, _DOT_BOH_CAP)
    bow = _np2_min16(output_w, _DOT_BOW_CAP)
    bkw = _np2_min16(input_w, _DOT_BKW_CAP) if has_w else 16
    kh_blocks = triton.cdiv(input_h, bih) if has_h else 1
    kw_blocks = triton.cdiv(input_w, bkw) if has_w else 1
    # Peak DSM budget (fp32 worst case): a + b + tmp + wy + acc
    dsm_bytes = (bih * bkw + bkw * bow + bih * bow + boh * bih + boh * bow) * 4
    assert dsm_bytes <= _DOT_DSM_BUDGET, f"DSM {dsm_bytes} > {_DOT_DSM_BUDGET}"
    if out_tensor is not None and out_tensor.is_contiguous():
        out_flat = out_tensor.view(nc, output_h, output_w)
    else:
        out_flat = torch.empty((nc, output_h, output_w), dtype=input.dtype, device=dev)
    t_oh = triton.cdiv(output_h, boh)
    t_ow = triton.cdiv(output_w, bow)
    in_flat = input.reshape(nc, input_h, input_w)
    _fused_dot_forward_kernel[(_DOT_NUM_SPC,)](
        in_flat,
        out_flat,
        wy,
        wx,
        input_h,
        input_w,
        output_h,
        output_w,
        t_oh,
        t_ow,
        nc * t_oh * t_ow,
        BIH=bih,
        BOH=boh,
        BOW=bow,
        BKW=bkw,
        KH_BLOCKS=kh_blocks,
        KW_BLOCKS=kw_blocks,
        HAS_H=has_h,
        HAS_W=has_w,
        MODE=mode,
        NUM_SPC=_DOT_NUM_SPC,
        num_warps=1,
    )
    if out_tensor is not None and out_tensor.is_contiguous():
        return out_tensor
    return out_flat.reshape(n, c, output_h, output_w)


def _reciprocal_scale(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale: Optional[float],
) -> float:
    if align_corners:
        return (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    if scale is not None and scale > 0.0:
        return 1.0 / scale
    return input_size / output_size


def _support_and_taps(scale: float) -> Tuple[float, int]:
    support = 3.0 * scale if scale >= 1.0 else 3.0
    return support, math.ceil(support) * 2 + 1


def _validate(input: torch.Tensor, output_size: Sequence[int]) -> Tuple[int, int]:
    if input.device.type != device:
        raise RuntimeError(f"Expected input on {device}, but got {input.device.type}")
    if input.ndim != 4:
        raise RuntimeError(
            f"It is expected input_size equals to 4, but got size {input.ndim}"
        )
    if len(output_size) != 2:
        raise RuntimeError(
            f"It is expected output_size equals to 2, but got size {len(output_size)}"
        )
    if not input.is_floating_point() and input.dtype != torch.uint8:
        raise RuntimeError(
            f'"compute_index_ranges_weights" not implemented for {input.dtype}'
        )

    output_h, output_w = int(output_size[0]), int(output_size[1])
    if min(input.shape[-2], input.shape[-1], output_h, output_w) <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if input.shape[1] == 0:
        raise RuntimeError("Non-empty 4D data tensor expected")
    return output_h, output_w


def _make_lanczos_weights(
    input_size: int,
    output_size: int,
    scale: float,
    support: float,
    taps: int,
    device: torch.device,
    is_fp64: bool,
):
    # Transposed layout [taps, output_size]: per-tap weight gathers along the
    # flattened output index become contiguous loads (GCU400, see kernel note).
    weights = torch.empty(
        (taps, output_size),
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


def _int_scale_segment(
    input_size: int,
    output_size: int,
    scale: float,
    support: float,
    taps: int,
):
    """Return (K, B0, first, last, size_ref) for an exact integer scale >= 1.

    Inside [first, last) every output index has the full (unclamped) tap
    window, so index_min = K * y + B0 is affine and the tap weights are
    identical across the segment. Every row of the segment is exhaustively
    verified against the device-side formulas (trunc semantics included);
    returns None on any mismatch so the caller falls back to the generic
    kernel. Pure host-side scalar arithmetic on shape metadata.
    """
    K = round(scale)
    if K < 1 or abs(scale - K) > 1e-9:
        return None
    sizes = []
    for out_idx in range(output_size):
        center = K * (out_idx + 0.5)
        index_min = max(int(center - support + 0.5), 0)
        index_max = min(int(center + support + 0.5), input_size)
        sizes.append(min(max(index_max - index_min, 0), taps))
    size_ref = sizes[output_size // 2]
    first = 0
    while first < output_size and sizes[first] != size_ref:
        first += 1
    last = first
    while last < output_size and sizes[last] == size_ref:
        last += 1
    if last - first < 2:
        return None
    B0 = int(K * (first + 0.5) - support + 0.5) - K * first
    for out_idx in range(first, last):
        raw_min = int(K * (out_idx + 0.5) - support + 0.5)
        if raw_min != K * out_idx + B0 or raw_min < 0:
            return None
        if sizes[out_idx] != size_ref:
            return None
    return K, B0, first, last, size_ref


def _upsample_lanczos2d_aa_contiguous(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scales_h: Optional[float],
    scales_w: Optional[float],
) -> torch.Tensor:
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    if input_h == output_h and input_w == output_w:
        return input.clone()

    # gcu400 main path: a single kernel with a dense dot chain. For huge input dimensions beyond the static-unroll limit
    # on the K dimension (H_in/W_in),
    # fall back to the legacy gather path (its tap unrolling is independent of the input dimension).
    if _dot_path_eligible(input, output_h, output_w, align_corners):
        return _dot_forward_impl(
            input, (output_h, output_w), align_corners, scales_h, scales_w, None
        )

    scale_w = _reciprocal_scale(input_w, output_w, align_corners, scales_w)
    scale_h = _reciprocal_scale(input_h, output_h, align_corners, scales_h)
    support_w, taps_w = _support_and_taps(scale_w)
    support_h, taps_h = _support_and_taps(scale_h)
    is_uint8 = input.dtype == torch.uint8
    is_fp64 = input.dtype == torch.float64
    # GCU400: the transposed weight table plus the integer-scale vertical fast
    # path amortize for any non-tiny batch; the generic cutoff (n*c >= 128)
    # leaves mid-size batches on the slow on-the-fly weight computation.
    precompute_weights = n * c >= 16
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
            seg = (
                _int_scale_segment(input_h, output_h, scale_h, support_h, taps_h)
                if precompute_weights
                else None
            )
            if seg is not None:
                K, B0, y_first, y_last, size_ref = seg
                block_y, block_x = 32, min(triton.next_power_of_2(output_w), 64)
                num_y_tiles = triton.cdiv(y_last - y_first, block_y)
                num_x_tiles = triton.cdiv(output_w, block_x)
                _lanczos_vertical_main_kernel[(n * c * num_y_tiles * num_x_tiles,)](
                    horizontal,
                    output,
                    weights_h,
                    input_h,
                    output_h,
                    output_w,
                    y_first,
                    y_last,
                    size_ref,
                    K=K,
                    B0=B0,
                    MAX_TAPS=taps_h,
                    T_POW2=triton.next_power_of_2(taps_h),
                    IS_UINT8=is_uint8,
                    IS_FP64=is_fp64,
                    BLOCK_Y=block_y,
                    BLOCK_X=block_x,
                )
                border_count = y_first + (output_h - y_last)
                if border_count > 0:
                    border_numel = n * c * border_count * output_w
                    _lanczos_vertical_border_kernel[
                        (triton.cdiv(border_numel, block_size),)
                    ](
                        horizontal,
                        output,
                        weights_h,
                        index_mins_h,
                        index_sizes_h,
                        border_numel,
                        input_h,
                        output_h,
                        output_w,
                        y_first,
                        y_last,
                        border_count,
                        MAX_TAPS=taps_h,
                        IS_UINT8=is_uint8,
                        IS_FP64=is_fp64,
                        BLOCK_SIZE=block_size,
                    )
            else:
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


def _upsample_lanczos2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_LANCZOS2D_AA")
    output_size = _validate(input, output_size)
    channels_last = input.is_contiguous(memory_format=torch.channels_last)
    if input.shape[0] == 0:
        output = torch.empty(
            (*input.shape[:2], *output_size), device=input.device, dtype=input.dtype
        )
    else:
        output = _upsample_lanczos2d_aa_contiguous(
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
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _UPSAMPLE_LANCZOS2D_AA_OUT")
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor on {input.device}, but got {out.device}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype}"
        )
    output_size = _validate(input, output_size)
    # dot path + contiguous out: first resize to the target shape, then have the kernel write directly into the caller's
    # buffer,
    # eliminating one full allocation and one device-to-device copy (non-contiguous out follows the original copy path).
    if (
        _dot_path_eligible(input, output_size[0], output_size[1], align_corners)
        and input.shape[0] != 0
        and (out.is_contiguous() or out.numel() == 0)
    ):
        out.resize_((input.shape[0], input.shape[1], output_size[0], output_size[1]))
        input = input.contiguous()
        return _dot_forward_impl(
            input, output_size, align_corners, scales_h, scales_w, out
        )
    result = _upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _upsample_lanczos2d_aa_vec(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _UPSAMPLE_LANCZOS2D_AA_VEC")
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
