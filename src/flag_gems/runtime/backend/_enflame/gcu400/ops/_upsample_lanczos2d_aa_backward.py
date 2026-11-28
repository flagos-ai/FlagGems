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

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_WEIGHT_CACHE_MAX_ENTRIES = 128
sinpi = tl_extra_shim.sinpi

# Final parameters: chosen by Step 4 round sweeps with fp16 veto, see this optimization report.
PERSISTENT_GRID = 192
BLOCK_IW_CAP = 256
_FUSE_THRESHOLD = 1048576
_WARPS_BIG = 4
_ASCEND_FUSE_MAX_PROGRAM_ITERATIONS = 512
_ASCEND_FUSE_ALWAYS_FILTER_AREA = 1

# ---------------------------------------------------------------------------
# gcu400 dense dot path (rewritten 2026-11-18):
#   Separable-filter backward = two GEMMs (grad @ Wx^T along W, Wy^T @ · along H). The original gather-tap
#   form (dynamic cross-row loop × weight-table gather with per-lane computed addresses × statically unrolled 8x8 masked
#   loads) serializes gathers lane by lane on GCU400, dragging large shapes down to 3.3-7.2ms (native takes only
#   26-57us, see the sibling upsample_bilinear2d_backward case). Switched to a dense weight-matrix
#   tl.dot chain (three-chain BIH input-row tile x [Wy^T | grad] x Wx), all affine-addressed.
#   Constraint: the K dimension (H_out/W_out) is unrolled via tl.static_range - a dynamic range loop on gcu400
#   carrying an fp32 tl.dot accumulator miscompiles (see comments in gcu400/ops/mm.py), so extremely large
#   H_out/W_out (number of unrolled blocks > _DOT_MAX_K_BLOCKS) falls back to the legacy gather path.
_DOT_NUM_SPC = 48  # GCU400 L600 fixed grid (number of SPs)
_DOT_MAX_K_BLOCKS = 16
_DOT_WEIGHT_CACHE_MAX = 128
_DOT_FP16_MIN_ELEMS = 65536
_DOT_WARPS2_MIN_ELEMS = 262144
_DOT_BIH_CAP = 128
_DOT_BIW_CAP = 128
_DOT_BK_CAP = 64


@triton.jit
def _lanczos_aa_filter(x):
    """Lanczos-3 filter used by PyTorch's antialiased implementation."""
    pix = math.pi * x
    sinc = tl.where(x == 0.0, 1.0, sinpi(x) / pix)
    sinc_three = tl.where(x == 0.0, 1.0, sinpi(x / 3.0) / (pix / 3.0))
    return tl.where(x < 3.0, sinc * sinc_three, 0.0)


@triton.jit
def _f2i(x):
    """float -> int32 with clamping to avoid undefined overflow."""
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
def _fused_dot_backward_kernel(
    grad_ptr,
    gi_ptr,
    wy_ptr,
    wx_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    T_IH,
    T_IW,
    TOTAL_WORK,
    BIH: tl.constexpr,
    BIW: tl.constexpr,
    BKH: tl.constexpr,
    BKW: tl.constexpr,
    KH_BLOCKS: tl.constexpr,
    KW_BLOCKS: tl.constexpr,
    MODE: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    """grad_in[ih, iw] = sum_oh sum_ow wy[oh, ih] * wx[ow, iw] * grad[oh, ow].

    Two-stage dot chain in one kernel: tmp = Wy^T @ grad (along H), acc = tmp @ Wx (along W).
    MODE=0: all-fp32 dot (exact, grad/weights widened to fp32 in-kernel);
    MODE=1: dot1 uses tensor cores in the grad dtype and tmp is cast back before dot2.
    All accesses are affine: Wy^T and Wx are dense weight matrices (zeroed outside the
    window) and grad / grad_in are row-major contiguous tiles.
    """
    pid = tl.program_id(0)
    for wid in range(pid, TOTAL_WORK, NUM_SPC):
        nc = wid // (T_IH * T_IW)
        r = wid - nc * (T_IH * T_IW)
        ih_t = r // T_IW
        iw_t = r - ih_t * T_IW
        ih = ih_t * BIH + tl.arange(0, BIH)
        iw = iw_t * BIW + tl.arange(0, BIW)
        ih_m = ih < H_in
        iw_m = iw < W_in
        nc64 = nc.to(tl.int64)
        acc = tl.zeros((BIH, BIW), dtype=tl.float32)
        for kwb in tl.static_range(KW_BLOCKS):
            ow = kwb * BKW + tl.arange(0, BKW)
            ow_m = ow < W_out
            tmp = tl.zeros((BIH, BKW), dtype=tl.float32)
            for khb in tl.static_range(KH_BLOCKS):
                oh = khb * BKH + tl.arange(0, BKH)
                oh_m = oh < H_out
                a = tl.load(
                    wy_ptr + ih[:, None] + oh[None, :].to(tl.int64) * H_in,
                    mask=ih_m[:, None] & oh_m[None, :],
                    other=0.0,
                )
                g = tl.load(
                    grad_ptr + (nc64 * H_out + oh)[:, None] * W_out + ow[None, :],
                    mask=oh_m[:, None] & ow_m[None, :],
                    other=0.0,
                )
                if MODE == 0:
                    tmp += tl.dot(
                        a.to(tl.float32),
                        g.to(tl.float32),
                        out_dtype=tl.float32,
                        allow_tf32=False,
                    )
                else:
                    tmp += tl.dot(a, g, out_dtype=tl.float32, allow_tf32=False)
            b = tl.load(
                wx_ptr + ow[:, None].to(tl.int64) * W_in + iw[None, :],
                mask=ow_m[:, None] & iw_m[None, :],
                other=0.0,
            )
            if MODE == 1:
                acc += tl.dot(
                    tmp.to(b.dtype), b, out_dtype=tl.float32, allow_tf32=False
                )
            else:
                acc += tl.dot(
                    tmp, b.to(tl.float32), out_dtype=tl.float32, allow_tf32=False
                )
        tl.store(
            gi_ptr + (nc64 * H_in + ih)[:, None] * W_in + iw[None, :],
            acc.to(gi_ptr.dtype.element_ty),
            mask=ih_m[:, None] & iw_m[None, :],
        )


@triton.jit
def _precompute_weights_kernel(
    weight_ptr,
    start_ptr,
    output_size,
    input_size,
    scale,
    support,
    invscale,
    MAX_KSIZE: tl.constexpr,
    BLOCK_KSIZE: tl.constexpr,
):
    oi = tl.program_id(0)
    if oi >= output_size:
        return

    offs = tl.arange(0, BLOCK_KSIZE)
    offs_mask = offs < MAX_KSIZE
    center = scale * (oi + 0.5)
    xmin = tl.maximum(_f2i(center - support + 0.5), 0)
    xsize = tl.minimum(_f2i(center + support + 0.5), input_size) - xmin
    xsize = tl.minimum(tl.maximum(xsize, 0), MAX_KSIZE)
    xmin_f = xmin.to(tl.float32)

    arg = tl.abs((offs.to(tl.float32) + xmin_f - center + 0.5) * invscale)
    raw_w = tl.where((offs < xsize) & offs_mask, _lanczos_aa_filter(arg), 0.0)
    total = tl.sum(raw_w, axis=0)
    weights = tl.where(total != 0.0, raw_w / total, 0.0)

    tl.store(weight_ptr + oi.to(tl.int64) * MAX_KSIZE + offs, weights, mask=offs_mask)
    tl.store(start_ptr + oi, xmin)


@triton.jit
def _fused_precomputed_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    wx_ptr,
    wx_start_ptr,
    wy_ptr,
    wy_start_ptr,
    H_in,
    H_out,
    W_in,
    W_out,
    support_h,
    support_w,
    inv_h_scale,
    inv_w_scale,
    stride_go_nc,
    TOTAL_ROWS,
    GRID_ROWS: tl.constexpr,
    BLOCK_IW: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_KSIZE_H: tl.constexpr,
    MAX_KSIZE_W: tl.constexpr,
):
    """Single-pass input gather; fixed-size grid with in-kernel row striding decouples program count from problem
    size."""
    pid_col = tl.program_id(1)
    row_tile = tl.program_id(0)

    for pid_row in range(row_tile, TOTAL_ROWS, GRID_ROWS):
        nc = pid_row // H_in
        ih = pid_row % H_in
        iws = pid_col * BLOCK_IW + tl.arange(0, BLOCK_IW)
        iw_mask = iws < W_in

        oh_start = tl.maximum(
            _f2i((ih.to(tl.float32) + 0.5 - support_h) * inv_h_scale - 0.5), 0
        )
        ow_starts = tl.maximum(
            _f2i((iws.to(tl.float32) + 0.5 - support_w) * inv_w_scale - 0.5), 0
        )
        accum = tl.zeros((BLOCK_IW,), dtype=tl.float32)
        go_nc_base = nc.to(tl.int64) * stride_go_nc

        for d_ow in tl.static_range(MAX_OW):
            ow = ow_starts + d_ow
            ow_valid = iw_mask & (ow < W_out)
            ow_safe = tl.minimum(ow, W_out - 1)
            xmin = tl.load(wx_start_ptr + ow_safe, mask=ow_valid, other=0)
            kx = iws - xmin
            x_valid = ow_valid & (kx >= 0) & (kx < MAX_KSIZE_W)
            kx_safe = tl.minimum(tl.maximum(kx, 0), MAX_KSIZE_W - 1)
            wx = tl.load(
                wx_ptr + ow_safe.to(tl.int64) * MAX_KSIZE_W + kx_safe.to(tl.int64),
                mask=x_valid,
                other=0.0,
            )

            for d_oh in tl.static_range(MAX_OH):
                oh = oh_start + d_oh
                oh_valid = oh < H_out
                oh_safe = tl.minimum(oh, H_out - 1)
                ymin = tl.load(wy_start_ptr + oh_safe, mask=oh_valid, other=0)
                ky = ih - ymin
                y_valid = oh_valid & (ky >= 0) & (ky < MAX_KSIZE_H)
                ky_safe = tl.minimum(tl.maximum(ky, 0), MAX_KSIZE_H - 1)
                wy = tl.load(
                    wy_ptr + oh_safe.to(tl.int64) * MAX_KSIZE_H + ky_safe.to(tl.int64),
                    mask=y_valid,
                    other=0.0,
                )
                valid = x_valid & y_valid
                grad = tl.load(
                    grad_out_ptr
                    + go_nc_base
                    + oh_safe.to(tl.int64) * W_out
                    + ow_safe.to(tl.int64),
                    mask=valid,
                    other=0.0,
                )
                accum += wx * wy * grad

        grad_in_offset = pid_row.to(tl.int64) * W_in + iws.to(tl.int64)
        tl.store(
            grad_in_ptr + grad_in_offset,
            accum.to(grad_in_ptr.dtype.element_ty),
            mask=iw_mask,
        )


@triton.jit
def _pass1_w_gather_nchw_kernel_plain(
    grad_out_ptr,  # [NC, H_out, W_out] flat
    buf_ptr,  # [NC, H_out, W_in]  flat (output)
    wx_ptr,  # [W_out, MAX_KSIZE_W]
    wx_start_ptr,  # [W_out]
    W_in,
    W_out,
    support_w,
    inv_w_scale,
    BLOCK_IW: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_KSIZE_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in
    iw_f = iws.to(tl.float32)

    go_base = pid_row.to(tl.int64) * W_out
    buf_base = pid_row.to(tl.int64) * W_in

    ow_starts = tl.maximum(_f2i((iw_f + 0.5 - support_w) * inv_w_scale - 0.5), 0)

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    for d_ow in tl.static_range(MAX_OW):
        ow = ow_starts + d_ow
        ow_valid = iw_mask & (ow >= 0) & (ow < W_out)
        ow_safe = tl.maximum(tl.minimum(ow, W_out - 1), 0)

        xmin = tl.load(wx_start_ptr + ow_safe)
        k = iws - xmin
        in_range = ow_valid & (k >= 0) & (k < MAX_KSIZE_W)
        k_safe = tl.minimum(tl.maximum(k, 0), MAX_KSIZE_W - 1)
        wx_raw = tl.load(
            wx_ptr + ow_safe.to(tl.int64) * MAX_KSIZE_W + k_safe.to(tl.int64)
        )
        wx = tl.where(in_range, wx_raw, 0.0)

        g = tl.load(
            grad_out_ptr + go_base + ow_safe.to(tl.int64), mask=in_range, other=0.0
        )
        accum += wx * g

    tl.store(buf_ptr + buf_base + iws.to(tl.int64), accum, mask=iw_mask)


@triton.jit
def _pass2_h_gather_nchw_kernel_plain(
    buf_ptr,  # [NC, H_out, W_in] flat (input)
    grad_in_ptr,  # [NC, H_in,  W_in] flat (output)
    wy_ptr,  # [H_out, MAX_KSIZE_H]
    wy_start_ptr,  # [H_out]
    H_in,
    W_in,
    H_out,
    support_h,
    inv_h_scale,
    stride_buf_hw,  # = H_out * W_in
    BLOCK_IW: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_KSIZE_H: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    nc = pid_row // H_in
    ih = pid_row % H_in
    ih_f = ih.to(tl.float32)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in

    oh_start = tl.maximum(_f2i((ih_f + 0.5 - support_h) * inv_h_scale - 0.5), 0)

    buf_nc_base = nc.to(tl.int64) * stride_buf_hw

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    for d_oh in tl.static_range(MAX_OH):
        oh = oh_start + d_oh
        oh_valid = (oh >= 0) & (oh < H_out)
        oh_safe = tl.maximum(tl.minimum(oh, H_out - 1), 0)

        ymin = tl.load(wy_start_ptr + oh_safe)
        k = ih - ymin
        ih_in_range = oh_valid & (k >= 0) & (k < MAX_KSIZE_H)
        k_safe = tl.minimum(tl.maximum(k, 0), MAX_KSIZE_H - 1)
        wy_raw = tl.load(
            wy_ptr + oh_safe.to(tl.int64) * MAX_KSIZE_H + k_safe.to(tl.int64)
        )
        wy = tl.where(ih_in_range, wy_raw, 0.0)

        buf_off = buf_nc_base + oh_safe.to(tl.int64) * W_in + iws.to(tl.int64)
        b = tl.load(buf_ptr + buf_off, mask=iw_mask & ih_in_range, other=0.0)

        accum += wy * b

    gi_off = pid_row.to(tl.int64) * W_in + iws.to(tl.int64)
    tl.store(
        grad_in_ptr + gi_off,
        accum.to(grad_in_ptr.dtype.element_ty),
        mask=iw_mask,
    )


def _compute_scale(input_size, output_size, align_corners, scale=None):
    if align_corners:
        return float(input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    else:
        return (
            (1.0 / scale)
            if (scale is not None and scale > 0)
            else float(input_size) / output_size
        )


def _lanczos_aa_filter_scalar(x):
    x = abs(float(x))
    if x == 0.0:
        return 1.0
    if x >= 3.0:
        return 0.0
    pix = math.pi * x
    return (math.sin(pix) / pix) * (math.sin(pix / 3.0) / (pix / 3.0))


def _weight_matrix_cache_key(
    input_size, output_size, align_corners, scale, dtype, device, transpose
):
    return (
        int(input_size),
        int(output_size),
        bool(align_corners),
        None if scale is None else float(scale),
        dtype,
        str(device),
        bool(transpose),
    )


def _get_weight_matrix(
    input_size, output_size, align_corners, scale_arg, dtype, device, transpose=False
):
    cache = getattr(_get_weight_matrix, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        _get_weight_matrix._cache = cache
    key = _weight_matrix_cache_key(
        input_size, output_size, align_corners, scale_arg, dtype, device, transpose
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    scale = _compute_scale(input_size, output_size, align_corners, scale_arg)
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0

    weight_cpu = torch.zeros((output_size, input_size), dtype=dtype)
    for oi in range(output_size):
        center = scale * (oi + 0.5)
        xmin = max(math.floor(center - support + 0.5), 0)
        xmax = min(math.floor(center + support + 0.5), input_size)
        values = [
            _lanczos_aa_filter_scalar((ii - center + 0.5) * invscale)
            for ii in range(xmin, xmax)
        ]
        total = sum(values)
        if total != 0.0:
            for ii, value in zip(range(xmin, xmax), values):
                weight_cpu[oi, ii] = value / total

    if transpose:
        weight_cpu = weight_cpu.t().contiguous()
    weight = weight_cpu.to(device=device, dtype=dtype)
    if len(cache) >= _WEIGHT_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
    cache[key] = weight
    return weight


def _gemm_backward_path(
    grad_output, output_size, input_size, align_corners, scales_h, scales_w
):
    N, C, H_in, W_in = input_size
    H_out, W_out = output_size
    NC = N * C

    compute_dtype = (
        torch.float64 if grad_output.dtype == torch.float64 else torch.float32
    )
    wx = _get_weight_matrix(
        W_in, W_out, align_corners, scales_w, compute_dtype, grad_output.device
    )
    if NC == 1:
        wy_t = _get_weight_matrix(
            H_in,
            H_out,
            align_corners,
            scales_h,
            compute_dtype,
            grad_output.device,
            transpose=True,
        )

        grad_w = grad_output.contiguous().reshape(H_out, W_out).to(compute_dtype)
        tmp = flag_gems.mm(grad_w, wx)
        out = flag_gems.mm(wy_t, tmp)
        return out.reshape(N, C, H_in, W_in).to(grad_output.dtype)

    wy = _get_weight_matrix(
        H_in, H_out, align_corners, scales_h, compute_dtype, grad_output.device
    )

    grad_w = grad_output.contiguous().reshape(NC * H_out, W_out).to(compute_dtype)
    tmp = flag_gems.mm(grad_w, wx).reshape(NC, H_out, W_in)
    tmp_h = tmp.transpose(1, 2).contiguous().reshape(NC * W_in, H_out)
    out = flag_gems.mm(tmp_h, wy).reshape(NC, W_in, H_in).transpose(1, 2).contiguous()
    return out.reshape(N, C, H_in, W_in).to(grad_output.dtype)


def _should_use_gemm_path(device_type, NC, H_in, W_in, H_out, W_out, align_corners):
    if align_corners and (H_out == 1 or W_out == 1):
        return False
    if device_type == "npu":
        return True
    if device_type == "cuda":
        tiny = NC * max(H_in * W_in, H_out * W_out) < 8192
        channel_heavy_downsample = NC >= 128 and H_out < H_in and W_out < W_in
        very_large_upsample = NC >= (1 << 17) and H_out >= H_in and W_out >= W_in
        return tiny or channel_heavy_downsample or very_large_upsample
    return False


def _should_use_fused_path(
    device_type, total_elems, NC, H_in, W_in, BLOCK_IW, MAX_OH, MAX_OW
):
    filter_area = MAX_OH * MAX_OW
    if device_type == "npu":
        if filter_area <= _ASCEND_FUSE_ALWAYS_FILTER_AREA:
            return True
        w_tiles = triton.cdiv(W_in, BLOCK_IW)
        fused_program_iterations = NC * H_in * w_tiles * filter_area
        return fused_program_iterations <= _ASCEND_FUSE_MAX_PROGRAM_ITERATIONS
    return total_elems <= _FUSE_THRESHOLD


def _persistent_grid(total_rows, col_tiles):
    """Cap grid rows at PERSISTENT_GRID and stride over all rows in-kernel by GRID_ROWS."""
    max_rows = max(1, PERSISTENT_GRID // col_tiles)
    return min(total_rows, max_rows)


def _get_dense_weights(
    input_size, output_size, align_corners, scale_arg, dtype, device
):
    """Dense [out, in] Lanczos-AA weights (host caches the device tensor, one kernel on cache miss)."""
    cache = getattr(_get_dense_weights, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        _get_dense_weights._cache = cache
    key = _weight_matrix_cache_key(
        input_size, output_size, align_corners, scale_arg, dtype, device, False
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    scale = _compute_scale(input_size, output_size, align_corners, scale_arg)
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    weight = torch.zeros((output_size, input_size), dtype=dtype, device=device)
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


def _dot_backward_impl(
    grad_output,
    grad_out_flat,
    input_size,
    H_out,
    W_out,
    align_corners,
    scales_h,
    scales_w,
    out_tensor,
):
    """Dense dot path: a single kernel launch performs the separable backward with affine accesses."""
    N, C, H_in, W_in = input_size
    NC = N * C
    dev = grad_output.device
    work_elems = NC * H_in * W_in
    use_f16_dot = (
        grad_output.dtype in (torch.float16, torch.bfloat16)
        and work_elems >= _DOT_FP16_MIN_ELEMS
    )
    weight_dtype = grad_output.dtype if use_f16_dot else torch.float32
    mode = 1 if use_f16_dot else 0
    wy = _get_dense_weights(H_in, H_out, align_corners, scales_h, weight_dtype, dev)
    wx = _get_dense_weights(W_in, W_out, align_corners, scales_w, weight_dtype, dev)

    if out_tensor is not None and out_tensor.is_contiguous():
        grad_in_flat = out_tensor.view(NC, H_in, W_in)
    else:
        grad_in_flat = torch.empty(NC, H_in, W_in, dtype=grad_output.dtype, device=dev)

    BIH = _np2_min16(H_in, _DOT_BIH_CAP)
    BIW = _np2_min16(W_in, _DOT_BIW_CAP)
    BKH = _np2_min16(H_out, _DOT_BK_CAP)
    BKW = _np2_min16(W_out, _DOT_BK_CAP)
    T_IH = triton.cdiv(H_in, BIH)
    T_IW = triton.cdiv(W_in, BIW)
    nw = 2 if work_elems >= _DOT_WARPS2_MIN_ELEMS else 4
    _fused_dot_backward_kernel[(_DOT_NUM_SPC,)](
        grad_out_flat,
        grad_in_flat,
        wy,
        wx,
        H_in,
        W_in,
        H_out,
        W_out,
        T_IH,
        T_IW,
        NC * T_IH * T_IW,
        BIH=BIH,
        BIW=BIW,
        BKH=BKH,
        BKW=BKW,
        KH_BLOCKS=triton.cdiv(H_out, BKH),
        KW_BLOCKS=triton.cdiv(W_out, BKW),
        MODE=mode,
        NUM_SPC=_DOT_NUM_SPC,
        num_warps=nw,
    )
    if out_tensor is not None and out_tensor.is_contiguous():
        return out_tensor
    result = grad_in_flat.reshape(N, C, H_in, W_in)
    if out_tensor is not None:
        out_tensor.resize_(result.shape).copy_(result)
        return out_tensor
    return result


def _backward_impl(
    grad_output, output_size, input_size, align_corners, scales_h, scales_w, out_tensor
):
    N, C, H_in, W_in = map(int, input_size)
    H_out, W_out = map(int, output_size)
    NC = N * C

    if NC == 0 or H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
        empty = grad_output.new_zeros(input_size)
        if out_tensor is not None:
            out_tensor.resize_(tuple(input_size)).zero_()
            return out_tensor
        return empty

    grad_out_flat = grad_output.contiguous().reshape(NC, H_out, W_out)

    # gcu400 main path: dense dot. Huge output dimensions beyond the static-unroll limit of the K dimension
    # (H_out/W_out)
    # fall back to the legacy gather path (whose tap-unroll scale is independent of the output dimension).
    if (
        triton.cdiv(H_out, _DOT_BK_CAP) <= _DOT_MAX_K_BLOCKS
        and triton.cdiv(W_out, _DOT_BK_CAP) <= _DOT_MAX_K_BLOCKS
        and grad_output.device.type == "gcu"
    ):
        return _dot_backward_impl(
            grad_output,
            grad_out_flat,
            (N, C, H_in, W_in),
            H_out,
            W_out,
            align_corners,
            scales_h,
            scales_w,
            out_tensor,
        )
    return _legacy_backward_impl(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        out_tensor,
    )


def _legacy_backward_impl(
    grad_output, output_size, input_size, align_corners, scales_h, scales_w, out_tensor
):
    N, C, H_in, W_in = map(int, input_size)
    H_out, W_out = map(int, output_size)
    NC = N * C

    if NC == 0 or H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
        empty = grad_output.new_zeros(input_size)
        if out_tensor is not None:
            out_tensor.resize_(tuple(input_size)).zero_()
            return out_tensor
        return empty

    grad_out_flat = grad_output.contiguous().reshape(NC, H_out, W_out)

    h_scale = _compute_scale(H_in, H_out, align_corners, scales_h)
    w_scale = _compute_scale(W_in, W_out, align_corners, scales_w)

    INTERP_SIZE = 6
    support_h = (INTERP_SIZE * 0.5) * h_scale if h_scale >= 1.0 else INTERP_SIZE * 0.5
    support_w = (INTERP_SIZE * 0.5) * w_scale if w_scale >= 1.0 else INTERP_SIZE * 0.5
    invscale_h = 1.0 / h_scale if h_scale >= 1.0 else 1.0
    invscale_w = 1.0 / w_scale if w_scale >= 1.0 else 1.0

    MAX_KSIZE_H = math.ceil(support_h) * 2 + 1
    MAX_KSIZE_W = math.ceil(support_w) * 2 + 1

    _EPS = 1e-10
    inv_h_scale = 1.0 / max(h_scale, _EPS)
    inv_w_scale = 1.0 / max(w_scale, _EPS)

    MAX_OH = min(math.ceil(2 * support_h * inv_h_scale) + 2, max(H_out, 1))
    MAX_OW = min(math.ceil(2 * support_w * inv_w_scale) + 2, max(W_out, 1))

    BLOCK_IW = min(triton.next_power_of_2(max(W_in, 1)), BLOCK_IW_CAP)
    if BLOCK_IW < 32:
        BLOCK_IW = 32
    nw = 1 if BLOCK_IW <= 32 else (2 if BLOCK_IW <= 64 else _WARPS_BIG)

    col_tiles = triton.cdiv(W_in, BLOCK_IW)
    total_elems = NC * max(H_in * W_in, H_out * W_out)
    use_fused = _should_use_fused_path(
        grad_output.device.type,
        total_elems,
        NC,
        H_in,
        W_in,
        BLOCK_IW,
        MAX_OH,
        MAX_OW,
    )

    # Output tensor reuse: the out variant writes kernel results directly into the caller's grad_input,
    # saving one full-size allocation and one full device-to-device copy.
    if out_tensor is not None and out_tensor.is_contiguous():
        grad_in_flat = out_tensor.view(NC, H_in, W_in)
    else:
        grad_in_flat = torch.empty(
            NC, H_in, W_in, dtype=grad_output.dtype, device=grad_output.device
        )

    wy = torch.empty(
        max(H_out, 1), MAX_KSIZE_H, dtype=torch.float32, device=grad_output.device
    )
    wx = torch.empty(
        max(W_out, 1), MAX_KSIZE_W, dtype=torch.float32, device=grad_output.device
    )
    wy_start = torch.empty(max(H_out, 1), dtype=torch.int32, device=grad_output.device)
    wx_start = torch.empty(max(W_out, 1), dtype=torch.int32, device=grad_output.device)

    _precompute_weights_kernel[(H_out,)](
        wy,
        wy_start,
        H_out,
        H_in,
        h_scale,
        support_h,
        invscale_h,
        MAX_KSIZE=MAX_KSIZE_H,
        BLOCK_KSIZE=triton.next_power_of_2(MAX_KSIZE_H),
    )
    _precompute_weights_kernel[(W_out,)](
        wx,
        wx_start,
        W_out,
        W_in,
        w_scale,
        support_w,
        invscale_w,
        MAX_KSIZE=MAX_KSIZE_W,
        BLOCK_KSIZE=triton.next_power_of_2(MAX_KSIZE_W),
    )

    if use_fused:
        total_rows = NC * H_in
        grid_rows = _persistent_grid(total_rows, col_tiles)
        _fused_precomputed_backward_kernel[(grid_rows, col_tiles)](
            grad_out_flat,
            grad_in_flat,
            wx,
            wx_start,
            wy,
            wy_start,
            H_in,
            H_out,
            W_in,
            W_out,
            support_h,
            support_w,
            inv_h_scale,
            inv_w_scale,
            H_out * W_out,
            total_rows,
            GRID_ROWS=grid_rows,
            BLOCK_IW=BLOCK_IW,
            MAX_OH=MAX_OH,
            MAX_OW=MAX_OW,
            MAX_KSIZE_H=MAX_KSIZE_H,
            MAX_KSIZE_W=MAX_KSIZE_W,
            num_warps=nw,
        )
    else:
        buf = torch.empty(
            NC, H_out, W_in, dtype=torch.float32, device=grad_output.device
        )
        _pass1_w_gather_nchw_kernel_plain[(NC * H_out, col_tiles)](
            grad_out_flat,
            buf,
            wx,
            wx_start,
            W_in,
            W_out,
            support_w,
            inv_w_scale,
            BLOCK_IW=BLOCK_IW,
            MAX_OW=MAX_OW,
            MAX_KSIZE_W=MAX_KSIZE_W,
            num_warps=nw,
        )

        _pass2_h_gather_nchw_kernel_plain[(NC * H_in, col_tiles)](
            buf,
            grad_in_flat,
            wy,
            wy_start,
            H_in,
            W_in,
            H_out,
            support_h,
            inv_h_scale,
            H_out * W_in,
            BLOCK_IW=BLOCK_IW,
            MAX_OH=MAX_OH,
            MAX_KSIZE_H=MAX_KSIZE_H,
            num_warps=nw,
        )

    if out_tensor is not None and out_tensor.is_contiguous():
        return out_tensor
    result = grad_in_flat.reshape(N, C, H_in, W_in)
    if out_tensor is not None:
        out_tensor.resize_(result.shape).copy_(result)
        return out_tensor
    return result


def upsample_lanczos2d_aa_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_LANCZOS2D_AA_BACKWARD")
    if grad_output.ndim != 4:
        raise RuntimeError(
            "Expected grad_output to be a tensor of dimension 4 but got: "
            f"dimension {grad_output.ndim}"
        )
    if len(output_size) != 2 or len(input_size) != 4:
        raise RuntimeError("Expected output_size[2] and input_size[4]")
    if not grad_output.is_floating_point():
        raise RuntimeError(
            f'"upsample_lanczos2d_aa_backward" not implemented for {grad_output.dtype}'
        )

    N, C, H_in, W_in = map(int, input_size)
    H_out, W_out = output_size
    if tuple(grad_output.shape) != (N, C, H_out, W_out):
        raise RuntimeError(
            f"Expected grad_output shape ({N}, {C}, {H_out}, {W_out}), "
            f"but got {tuple(grad_output.shape)}"
        )

    if grad_output.dtype == torch.float64 or _should_use_gemm_path(
        grad_output.device.type, N * C, H_in, W_in, H_out, W_out, align_corners
    ):
        return _gemm_backward_path(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w
        )

    return _backward_impl(
        grad_output, output_size, input_size, align_corners, scales_h, scales_w, None
    )


def upsample_lanczos2d_aa_backward_grad_input(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales_h=None,
    scales_w=None,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME UPSAMPLE_LANCZOS2D_AA_BACKWARD_GRAD_INPUT")
    if grad_input.device != grad_output.device:
        raise RuntimeError(
            f"Expected grad_input on {grad_output.device}, but got {grad_input.device}"
        )
    if grad_input.dtype != grad_output.dtype:
        raise RuntimeError(
            f"Expected grad_input dtype {grad_output.dtype}, but got {grad_input.dtype}"
        )

    grad_input.resize_(tuple(map(int, input_size)))
    result = _backward_impl(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w,
        grad_input,
    )
    return result
