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

from flag_gems import runtime
from flag_gems.ops.quantized_max_pool3d import (
    _aten_dtype_name,
    _check_quantized_input,
    _int_view,
    _parse_pool3d_params,
    _quantized_max_pool3d_output_shape,
    _resize_quantized_out,
)
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)

# Quantized integer dtypes natively accepted by ``aten::quantized_max_pool3d``,
# mapped to the dtype of their ``int_repr()``.
_QINT_TO_INT = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("quantized_max_pool3d"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def quantized_max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides (compile-time: the tap addresses stay affine, which
    # is what lets the backend emit per-row block loads instead of a discrete
    # per-lane gather; the layouts that reach this kernel are finite, so the
    # extra specializations are cheap).
    in_stride_n: tl.constexpr,
    in_stride_c: tl.constexpr,
    in_stride_d: tl.constexpr,
    in_stride_h: tl.constexpr,
    in_stride_w: tl.constexpr,
    # Output tensor strides
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_d: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Forward kernel for quantized 3-D max pooling (XPU-safe variant).

    Same tiling as the general kernel, but no ``tl.load(..., other=)``:
    out-of-bounds indices are clamped to a valid address and the value mask is
    applied with ``tl.where`` after the (unconditional) load. Masked stores
    for the tile tail are kept.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    hw_remainder = pid_dhw % (num_h_blocks * num_w_blocks)
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    d_out = d_block_idx

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kd in tl.static_range(0, kernel_d):
        d_in = d_out * stride_d - padding_d + kd * dilation_d
        d_ok = (d_in >= 0) & (d_in < in_d)
        d_safe = tl.where(d_ok, d_in, 0)
        for kh in tl.static_range(0, kernel_h):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            h_ok = (h_in >= 0) & (h_in < in_h)
            h_safe = tl.where(h_ok, h_in, 0)
            for kw in tl.static_range(0, kernel_w):
                w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
                w_ok = (w_in >= 0) & (w_in < in_w)
                w_safe = tl.where(w_ok, w_in, 0)
                valid = d_ok & h_ok & w_ok
                input_offset = (
                    d_safe * in_stride_d + h_safe * in_stride_h + w_safe * in_stride_w
                )
                current_val = tl.load(input_base_ptr + input_offset)
                max_val_acc = tl.maximum(
                    max_val_acc, tl.where(valid, current_val, min_val)
                )

    out_base_ptr = (
        output_ptr + n_idx * out_stride_n + c_idx * out_stride_c + d_out * out_stride_d
    )
    output_block_ptr = (
        out_base_ptr
        + h_out_offsets[:, None] * out_stride_h
        + w_out_offsets[None, :] * out_stride_w
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Fast path for the benchmark-dominant geometry (k=3, s=2, p=1, dilation=1).
#
# The generic tiled kernel above addresses every tap with a 2-D pointer tensor
# whose row stride is only known at runtime, so on XPU3 each tap lowers to a
# per-lane gather, and (BLOCK_H, BLOCK_W) tiles themselves are serialized
# row-by-row by the backend (~0.5-1.3 GB/s).  The two flat kernels below keep
# every access a 1-D, run-contiguous stream, which is the only access shape
# that reaches the block-DMA path on this toolchain (~240-520 GB/s):
#
#   1. ``wpool``: max over the 3 W taps -> an intermediate of shape
#      (N, C, D, H, out_w).  Lanes enumerate the whole intermediate, so both
#      the load runs and the store are contiguous.  For an even ``in_w`` the
#      rows are 2-byte aligned and the tap bytes are extracted from an int16
#      "pair" view, so both taps are plain unit-stride int16 loads.
#   2. ``hdpool``: max over the 3x3 (d, h) window of the intermediate ->
#      (N, C, out_d, out_h, out_w).  Lanes enumerate the final output; the nine
#      taps are scalar offsets from a per-lane base that stays run-contiguous
#      along W.
#
# Shape guidance (measured on XPU3): putting a ``tl.where`` on a *load address*
# turns the run-contiguous gather into a serialized per-lane path (up to 16x
# slower), so padding taps are neutralized by a value-level ``tl.where`` and
# the intermediate is allocated with a neutral-slack halo that keeps every
# computed address in bounds.  Padding the addresses instead is not an option.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def quantized_max_pool3d_wpool_word_kernel(
    x_ptr,
    a_ptr,
    total_words,
    out_w: tl.constexpr,
    signed: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """W-window max, four outputs per lane, packed into one int32 store.

    ``x_ptr`` is the input viewed as int64 (little-endian: byte 0 = lowest
    address) and ``a_ptr`` the intermediate viewed as int32.  The 9 taps of
    the 4 outputs in a lane are bytes ``8j-1 .. 8j+7``, i.e. the previous and
    the current word, so the whole pass is two unit-stride int64 loads plus one
    unit-stride int32 store per 4 outputs.

    Note: a plain int32-word variant (three loads, no 64-bit shifts) is ~2x
    faster but the *masked negative-offset* int32 load silently returns values
    from neighbouring lanes on XPU3, so this masked form keeps int64.  The
    unmasked two-launch split below (``_wpool_head_kernel`` +
    ``_wpool_body_kernel``) avoids the masked negative load entirely and does
    use int32 words.
    """
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < total_words

    wc = tl.load(x_ptr + j, mask=m, other=0)
    wp = tl.load(x_ptr + j - 1, mask=(j > 0) & m, other=0)

    b0 = wc & 0xFF
    b1 = (wc >> 8) & 0xFF
    b2 = (wc >> 16) & 0xFF
    b3 = (wc >> 24) & 0xFF
    b4 = (wc >> 32) & 0xFF
    b5 = (wc >> 40) & 0xFF
    b6 = (wc >> 48) & 0xFF
    b7 = (wc >> 56) & 0xFF
    bp = (wp >> 56) & 0xFF

    if signed:
        b0 = tl.where(b0 >= 128, b0 - 256, b0)
        b1 = tl.where(b1 >= 128, b1 - 256, b1)
        b2 = tl.where(b2 >= 128, b2 - 256, b2)
        b3 = tl.where(b3 >= 128, b3 - 256, b3)
        b4 = tl.where(b4 >= 128, b4 - 256, b4)
        b5 = tl.where(b5 >= 128, b5 - 256, b5)
        b6 = tl.where(b6 >= 128, b6 - 256, b6)
        b7 = tl.where(b7 >= 128, b7 - 256, b7)
        bp = tl.where(bp >= 128, bp - 256, bp)

    # ``r + i == 0`` (only i == 0) or ``r + i == out_w`` marks a row start:
    # its left tap belongs to the padding.
    r = (4 * j) % out_w

    m0 = tl.maximum(tl.maximum(tl.where(r == 0, neutral, bp), b0), b1)
    m1 = tl.maximum(tl.maximum(tl.where(r + 1 == out_w, neutral, b1), b2), b3)
    m2 = tl.maximum(tl.maximum(tl.where(r + 2 == out_w, neutral, b3), b4), b5)
    m3 = tl.maximum(tl.maximum(tl.where(r + 3 == out_w, neutral, b5), b6), b7)

    packed = (
        (m0 & 0xFF) | ((m1 & 0xFF) << 8) | ((m2 & 0xFF) << 16) | ((m3 & 0xFF) << 24)
    )
    tl.store(a_ptr + j, packed.to(tl.int32), mask=m)


@libentry()
@triton.jit
def quantized_max_pool3d_wpool_head_kernel(
    x_ptr,
    a_ptr,
    total_words,
    out_w: tl.constexpr,
    signed: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Word-0 block of the two-launch W pass (masked, int64 view).

    Processes words ``[pid * BLOCK, (pid + 1) * BLOCK)``.  The launch grid
    keeps the head at one block per program, so the residual prefix does not
    serialize on a single program.
    """
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < total_words

    wc = tl.load(x_ptr + j, mask=m, other=0)
    wp = tl.load(x_ptr + j - 1, mask=(j > 0) & m, other=0)

    b0 = wc & 0xFF
    b1 = (wc >> 8) & 0xFF
    b2 = (wc >> 16) & 0xFF
    b3 = (wc >> 24) & 0xFF
    b4 = (wc >> 32) & 0xFF
    b5 = (wc >> 40) & 0xFF
    b6 = (wc >> 48) & 0xFF
    b7 = (wc >> 56) & 0xFF
    bp = (wp >> 56) & 0xFF

    if signed:
        b0 = tl.where(b0 >= 128, b0 - 256, b0)
        b1 = tl.where(b1 >= 128, b1 - 256, b1)
        b2 = tl.where(b2 >= 128, b2 - 256, b2)
        b3 = tl.where(b3 >= 128, b3 - 256, b3)
        b4 = tl.where(b4 >= 128, b4 - 256, b4)
        b5 = tl.where(b5 >= 128, b5 - 256, b5)
        b6 = tl.where(b6 >= 128, b6 - 256, b6)
        b7 = tl.where(b7 >= 128, b7 - 256, b7)
        bp = tl.where(bp >= 128, bp - 256, bp)

    r = (4 * j) % out_w
    m0 = tl.maximum(tl.maximum(tl.where(r == 0, neutral, bp), b0), b1)
    m1 = tl.maximum(tl.maximum(tl.where(r + 1 == out_w, neutral, b1), b2), b3)
    m2 = tl.maximum(tl.maximum(tl.where(r + 2 == out_w, neutral, b3), b4), b5)
    m3 = tl.maximum(tl.maximum(tl.where(r + 3 == out_w, neutral, b5), b6), b7)

    packed = (
        (m0 & 0xFF) | ((m1 & 0xFF) << 8) | ((m2 & 0xFF) << 16) | ((m3 & 0xFF) << 24)
    )
    tl.store(a_ptr + j, packed.to(tl.int32), mask=m)


@libentry()
@triton.jit
def quantized_max_pool3d_wpool_body_kernel(
    x_ptr,
    x8_ptr,
    a_ptr,
    out_w: tl.constexpr,
    signed: tl.constexpr,
    neutral: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unmasked body of the two-launch W pass (no masks at all).

    Program ``pid`` covers words ``[START + pid * BLOCK, START + (pid + 1) *
    BLOCK)``, i.e. everything after the masked head.  ``START`` is a constant
    (always a multiple of 256) so every index keeps its divisibility and stays
    int32; a runtime start disables that and costs ~60% on this backend.

    The two full words come from the int32 view; the single ``bp`` byte is a
    plain u8 load, which trims the read traffic to 9B per 4 outputs.
    """
    pid = tl.program_id(0)
    j = START + pid * BLOCK + tl.arange(0, BLOCK)

    lo = tl.load(x_ptr + 2 * j)
    hi = tl.load(x_ptr + 2 * j + 1)
    bp = tl.load(x8_ptr + 8 * j - 1).to(tl.int32)

    b0 = lo & 0xFF
    b1 = (lo >> 8) & 0xFF
    b2 = (lo >> 16) & 0xFF
    b3 = (lo >> 24) & 0xFF
    b4 = hi & 0xFF
    b5 = (hi >> 8) & 0xFF
    b6 = (hi >> 16) & 0xFF
    b7 = (hi >> 24) & 0xFF

    if signed:
        b0 = tl.where(b0 >= 128, b0 - 256, b0)
        b1 = tl.where(b1 >= 128, b1 - 256, b1)
        b2 = tl.where(b2 >= 128, b2 - 256, b2)
        b3 = tl.where(b3 >= 128, b3 - 256, b3)
        b4 = tl.where(b4 >= 128, b4 - 256, b4)
        b5 = tl.where(b5 >= 128, b5 - 256, b5)
        b6 = tl.where(b6 >= 128, b6 - 256, b6)
        b7 = tl.where(b7 >= 128, b7 - 256, b7)
        bp = tl.where(bp >= 128, bp - 256, bp)

    r = (4 * j) % out_w
    m0 = tl.maximum(tl.maximum(tl.where(r == 0, neutral, bp), b0), b1)
    m1 = tl.maximum(tl.maximum(tl.where(r + 1 == out_w, neutral, b1), b2), b3)
    m2 = tl.maximum(tl.maximum(tl.where(r + 2 == out_w, neutral, b3), b4), b5)
    m3 = tl.maximum(tl.maximum(tl.where(r + 3 == out_w, neutral, b5), b6), b7)

    packed = (
        (m0 & 0xFF) | ((m1 & 0xFF) << 8) | ((m2 & 0xFF) << 16) | ((m3 & 0xFF) << 24)
    )
    tl.store(a_ptr + j, packed.to(tl.int32))


@libentry()
@triton.jit
def quantized_max_pool3d_wpool_row7_kernel(
    x_ptr,
    a_ptr,
    num_rows,
    signed: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """W-window max for ``in_w == 7`` (out_w == 4): one row per lane.

    Odd ``in_w`` keeps rows out of the int16/int32 pair views, and the generic
    computed-address byte kernel is ~6x slower.  With ``in_w == 7`` a row has
    exactly four outputs, so a lane can load the seven row bytes, max the
    windows and emit one packed int32 (four bytes) for the flat intermediate.
    """
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < num_rows
    base = r * 7

    b0 = tl.load(x_ptr + base + 0, mask=m, other=0).to(tl.int32)
    b1 = tl.load(x_ptr + base + 1, mask=m, other=0).to(tl.int32)
    b2 = tl.load(x_ptr + base + 2, mask=m, other=0).to(tl.int32)
    b3 = tl.load(x_ptr + base + 3, mask=m, other=0).to(tl.int32)
    b4 = tl.load(x_ptr + base + 4, mask=m, other=0).to(tl.int32)
    b5 = tl.load(x_ptr + base + 5, mask=m, other=0).to(tl.int32)
    b6 = tl.load(x_ptr + base + 6, mask=m, other=0).to(tl.int32)

    if signed:
        b0 = tl.where(b0 >= 128, b0 - 256, b0)
        b1 = tl.where(b1 >= 128, b1 - 256, b1)
        b2 = tl.where(b2 >= 128, b2 - 256, b2)
        b3 = tl.where(b3 >= 128, b3 - 256, b3)
        b4 = tl.where(b4 >= 128, b4 - 256, b4)
        b5 = tl.where(b5 >= 128, b5 - 256, b5)
        b6 = tl.where(b6 >= 128, b6 - 256, b6)

    # taps of output q are bytes 2q-1, 2q, 2q+1 clipped to the row; the clips
    # fall on the dtype-min neutral, so ``tl.maximum`` over the in-row bytes
    # alone is exact.
    o0 = tl.maximum(b0, b1)
    o1 = tl.maximum(tl.maximum(b1, b2), b3)
    o2 = tl.maximum(tl.maximum(b3, b4), b5)
    o3 = tl.maximum(b5, b6)

    packed = (
        (o0.to(tl.int32) & 0xFF)
        | ((o1.to(tl.int32) & 0xFF) << 8)
        | ((o2.to(tl.int32) & 0xFF) << 16)
        | ((o3.to(tl.int32) & 0xFF) << 24)
    )
    tl.store(a_ptr + r, packed, mask=m)


@libentry()
@triton.jit
def quantized_max_pool3d_hdpool_pair_kernel(
    a_ptr,
    out_ptr,
    total_pairs,
    in_d: tl.constexpr,
    in_h: tl.constexpr,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w2: tl.constexpr,
    signed: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """3x3 (d, h) window max, two outputs per lane on the int16 pair views."""
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < total_pairs

    pj = j % out_w2
    oh = (j // out_w2) % out_h
    od = (j // (out_w2 * out_h)) % out_d
    plane = j // (out_w2 * out_h * out_d)

    hw2 = in_h * out_w2  # int16 elements per intermediate depth slice
    base = plane * (in_d * hw2) + (2 * od) * hw2 + (2 * oh) * out_w2 + pj

    acc_lo = tl.full((BLOCK,), neutral, tl.int32)
    acc_hi = tl.full((BLOCK,), neutral, tl.int32)
    for kd in tl.static_range(0, 3):
        d_in = 2 * od + kd - 1
        d_ok = (d_in >= 0) & (d_in < in_d)
        for kh in tl.static_range(0, 3):
            h_in = 2 * oh + kh - 1
            h_ok = (h_in >= 0) & (h_in < in_h)
            ok = d_ok & h_ok
            pv = tl.load(a_ptr + base + (kd - 1) * hw2 + (kh - 1) * out_w2, mask=m)
            if signed:
                lo = (pv << 8) >> 8
                hi = pv >> 8
            else:
                lo = pv & 0xFF
                hi = (pv >> 8) & 0xFF
            acc_lo = tl.maximum(acc_lo, tl.where(ok, lo, neutral))
            acc_hi = tl.maximum(acc_hi, tl.where(ok, hi, neutral))

    packed = (acc_lo & 0xFF) | ((acc_hi & 0xFF) << 8)
    tl.store(out_ptr + j, packed.to(tl.int16), mask=m)


@libentry()
@triton.jit
def quantized_max_pool3d_wpool_pair_kernel(
    x_ptr,
    a_ptr,
    total,
    out_w: tl.constexpr,
    signed: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """W-window max on the int16 pair view (``in_w`` even, pair index == lane)."""
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total
    q = o % out_w

    p = tl.load(x_ptr + o, mask=m, other=0)
    # ``pair[o - 1]`` supplies the left tap.  Both the ``o > 0`` guard and the
    # block-tail mask go into the *mask*: an equivalent ``tl.where`` on the
    # address silently miscompiles on XPU3 (it returns values from the wrong
    # pair for the first rows), so the address itself must stay plain.
    pl = tl.load(x_ptr + o - 1, mask=(o > 0) & m, other=0)

    if signed:
        lo = (p << 8) >> 8
        hi = p >> 8
        lhi = pl >> 8
    else:
        lo = p & 0xFF
        hi = (p >> 8) & 0xFF
        lhi = (pl >> 8) & 0xFF

    lhi = tl.where(q > 0, lhi, neutral)
    v = tl.maximum(tl.maximum(lo, hi), lhi)
    if signed:
        tl.store(a_ptr + o, v.to(tl.int8), mask=m)
    else:
        tl.store(a_ptr + o, v.to(tl.uint8), mask=m)


@libentry()
@triton.jit
def quantized_max_pool3d_wpool_kernel(
    x_ptr,
    a_ptr,
    total,
    # Geometry (compile-time so the flat index decompositions stay cheap)
    numel: tl.constexpr,
    in_w: tl.constexpr,
    out_w: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """W-window max for an odd ``in_w`` (rows are not pair-aligned)."""
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total

    q = o % out_w
    base = (o // out_w) * in_w + 2 * q
    base = tl.minimum(base, numel - 1)

    # Padding taps are filtered by the load *mask* (never by a ``tl.where`` on
    # the address: that form silently miscompiles on XPU3 and is also ~16x
    # slower), and the value falls back to the neutral element.
    has_l = q > 0
    left = tl.load(x_ptr + base - 1, mask=has_l & m, other=neutral)
    left = tl.where(has_l, left, neutral)

    mid = tl.load(x_ptr + base, mask=m, other=neutral)

    # Right tap ``2q + 1``: a padding slot when the window runs past the row
    # (odd ``in_w`` with ceil_mode, or a widened out_w).
    has_r = 2 * q + 1 < in_w
    raddr = tl.minimum(base + 1, numel - 1)
    right = tl.load(x_ptr + raddr, mask=has_r & m, other=neutral)
    right = tl.where(has_r, right, neutral)

    v = tl.maximum(tl.maximum(left, mid), right)
    tl.store(a_ptr + o, v, mask=m)


@libentry()
@triton.jit
def quantized_max_pool3d_hdpool_kernel(
    a_ptr,
    out_ptr,
    total,
    # Intermediate/final geometry (compile-time)
    in_d: tl.constexpr,
    in_h: tl.constexpr,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    neutral: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < total

    q = o % out_w
    oh = (o // out_w) % out_h
    od = (o // (out_w * out_h)) % out_d
    plane = o // (out_w * out_h * out_d)  # (n, c) index

    hw = in_h * out_w  # one intermediate depth-slice
    base = plane * (in_d * hw) + (2 * od) * hw + (2 * oh) * out_w + q

    acc = tl.full((BLOCK,), neutral, a_ptr.type.element_ty)
    for kd in tl.static_range(0, 3):
        d_in = 2 * od + kd - 1
        d_ok = (d_in >= 0) & (d_in < in_d)
        for kh in tl.static_range(0, 3):
            h_in = 2 * oh + kh - 1
            h_ok = (h_in >= 0) & (h_in < in_h)
            ok = d_ok & h_ok
            v = tl.load(a_ptr + base + (kd - 1) * hw + (kh - 1) * out_w, mask=m)
            acc = tl.maximum(acc, tl.where(ok, v, neutral))
    tl.store(out_ptr + o, acc, mask=m)


def _fast_pool3d_geometry(params, input, out_int):
    """Return the flat-plan parameters, or ``None`` when the fast path cannot run.

    The two flat kernels implement exactly the benchmark-dominant geometry:
    k=3, stride=2, padding=1, dilation=1, on a contiguous 1-byte quantized
    input with a contiguous 1-byte output.  Every other geometry keeps the
    generic tiled kernel.

    The W pass also assumes the flat row layout ``out_w == in_w // 2`` (even
    ``in_w``) or ``out_w == (in_w + 1) // 2`` (odd ``in_w``, right edge clipped
    by the kernel).  ``ceil_mode`` can widen ``out_w`` by one for even widths;
    that geometry needs the unguarded right edge, so it stays on the generic
    kernel.
    """
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params
    if (kd, kh, kw) != (3, 3, 3):
        return None
    if (sd, sh, sw) != (2, 2, 2):
        return None
    if (pd, ph, pw) != (1, 1, 1):
        return None
    if (dd, dh, dw) != (1, 1, 1):
        return None
    if input.element_size() != 1 or out_int.element_size() != 1:
        return None
    if not input.is_contiguous() or not out_int.is_contiguous():
        return None
    in_w = input.shape[-1]
    out_w = out_int.shape[-1]
    if out_w != (in_w // 2 if in_w % 2 == 0 else (in_w + 1) // 2):
        return None
    neutral = -128 if input.dtype == torch.qint8 else 0
    return neutral


_WPOOL_BLOCK = 2048
_WPOOL_BODY_BLOCK = 4096
_WPOOL_HEAD_BLOCK = 256
_WPOOL_ROW7_BLOCK = 1024
# HD pair pass: small outputs keep long windows per program, large outputs are
# better with more programs (measured on the benchmark matrix).
_HDPOOL_PAIR_SMALL = 4096
_HDPOOL_PAIR_LARGE = 1024
_HDPOOL_PAIR_SMALL_MAX = 65536
_HDBYTE_BLOCK = 2048
_PAD_SLACK = 4096


def _wpool_split(words):
    """Split the word range into a masked head ``[0, h)`` and an exact body.

    Returns ``(h, body_blocks)`` with ``h % 256 == 0``, ``(words - h) %
    4096 == 0`` and ``body_blocks == (words - h) // 4096``.  The head absorbs
    the residual prefix so the unmasked body always runs with the large block.
    ``None`` when the input is too small to split that way.
    """
    h = words % _WPOOL_BODY_BLOCK
    if h == 0:
        h = _WPOOL_BODY_BLOCK
    if h % _WPOOL_HEAD_BLOCK != 0 or h >= words:
        return None
    return h, (words - h) // _WPOOL_BODY_BLOCK


def _launch_quantized_max_pool3d_fast(int_repr, out_int, neutral):
    """Two flat passes: W-window max, then the 3x3 (d, h) window max."""
    in_n, in_c, in_d, in_h, in_w = int_repr.shape
    out_n, out_c, out_d, out_h, out_w = out_int.shape

    numel = int_repr.numel()
    total_inter = in_n * in_c * in_d * in_h * out_w
    total_out = out_int.numel()

    hw = in_h * out_w
    # Neutral slack so every computed tap address of both kernels lands inside
    # the allocation (values outside the live rows are neutralized in-kernel).
    # The live region must start on a 128-byte boundary: the W kernel's packed
    # int32 stores lose ~40% of their throughput at 8/16-byte offsets
    # (measured), and 128 bytes covers every word/pair view alignment.
    pad_lo = (((hw + out_w + 1) + 127) // 128) * 128
    pad_hi = (((hw + out_w + 1) + 127) // 128) * 128 + _PAD_SLACK
    inter = torch.empty(
        (pad_lo + total_inter + pad_hi), dtype=int_repr.dtype, device=int_repr.device
    )
    inter_live = inter[pad_lo : pad_lo + total_inter]

    signed = int_repr.dtype == torch.int8
    align8 = int_repr.storage_offset() % 8 == 0
    words = total_inter // 4
    split = _wpool_split(words) if total_inter % 4 == 0 else None

    if in_w % 2 == 0 and out_w >= 4 and numel % 8 == 0 and align8 and split:
        # Two-launch split: a masked head on the int64 view for the residual
        # prefix, then an unmasked int32 body for the rest.  The body's offsets
        # are all non-negative, which sidesteps the XPU3 miscompile of masked
        # negative-offset int32 loads, and its int32 ALU is ~2x the int64 form.
        head_words, body_blocks = split
        quantized_max_pool3d_wpool_head_kernel[(head_words // _WPOOL_HEAD_BLOCK,)](
            int_repr.reshape(-1).view(torch.int64),
            inter_live.view(torch.int32),
            min(head_words, words),
            out_w=out_w,
            signed=signed,
            neutral=neutral,
            BLOCK=_WPOOL_HEAD_BLOCK,
        )
        if body_blocks > 0:
            quantized_max_pool3d_wpool_body_kernel[(body_blocks,)](
                int_repr.reshape(-1).view(torch.int32),
                int_repr.reshape(-1),
                inter_live.view(torch.int32),
                out_w=out_w,
                signed=signed,
                neutral=neutral,
                START=head_words,
                BLOCK=_WPOOL_BODY_BLOCK,
            )
    elif in_w == 7 and out_w == 4 and total_inter % 4 == 0:
        # Odd row width: rows are out of every multi-byte view, so pool one row
        # per lane (four outputs -> one packed int32 store).
        num_rows = in_n * in_c * in_d * in_h
        quantized_max_pool3d_wpool_row7_kernel[
            (triton.cdiv(num_rows, _WPOOL_ROW7_BLOCK),)
        ](
            int_repr,
            inter_live.view(torch.int32),
            num_rows,
            signed=signed,
            neutral=neutral,
            BLOCK=_WPOOL_ROW7_BLOCK,
        )
    elif in_w % 2 == 0 and out_w >= 4 and numel % 8 == 0 and align8:
        # Words not divisible by any exact block: masked int64 word kernel.
        quantized_max_pool3d_wpool_word_kernel[(triton.cdiv(words, _WPOOL_BLOCK),)](
            int_repr.reshape(-1).view(torch.int64),
            inter_live.view(torch.int32),
            words,
            out_w=out_w,
            signed=signed,
            neutral=neutral,
            BLOCK=_WPOOL_BLOCK,
        )
    elif in_w % 2 == 0 and int_repr.storage_offset() % 2 == 0:
        # Rows are 2-byte aligned: both W taps come from unit-stride int16
        # loads of the pair view (pair index == flat intermediate index).
        quantized_max_pool3d_wpool_pair_kernel[
            (triton.cdiv(total_inter, _WPOOL_BLOCK),)
        ](
            int_repr.reshape(-1).view(torch.int16),
            inter_live,
            total_inter,
            out_w=out_w,
            signed=signed,
            neutral=neutral,
            BLOCK=_WPOOL_BLOCK,
        )
    else:
        quantized_max_pool3d_wpool_kernel[(triton.cdiv(total_inter, _WPOOL_BLOCK),)](
            int_repr,
            inter_live,
            total_inter,
            numel=numel,
            in_w=in_w,
            out_w=out_w,
            neutral=neutral,
            BLOCK=_WPOOL_BLOCK,
        )

    out_align = out_int.storage_offset() % 2 == 0
    if out_w % 2 == 0 and total_out % 2 == 0 and out_align:
        total_pairs = total_out // 2
        hd_block = (
            _HDPOOL_PAIR_SMALL
            if total_pairs <= _HDPOOL_PAIR_SMALL_MAX
            else _HDPOOL_PAIR_LARGE
        )
        quantized_max_pool3d_hdpool_pair_kernel[(triton.cdiv(total_pairs, hd_block),)](
            inter_live.view(torch.int16),
            out_int.reshape(-1).view(torch.int16),
            total_pairs,
            in_d=in_d,
            in_h=in_h,
            out_d=out_d,
            out_h=out_h,
            out_w2=out_w // 2,
            signed=signed,
            neutral=neutral,
            BLOCK=hd_block,
        )
    else:
        quantized_max_pool3d_hdpool_kernel[(triton.cdiv(total_out, _HDBYTE_BLOCK),)](
            inter_live,
            out_int,
            total_out,
            in_d=in_d,
            in_h=in_h,
            out_d=out_d,
            out_h=out_h,
            out_w=out_w,
            neutral=neutral,
            BLOCK=_HDBYTE_BLOCK,
        )
    return out_int


def _launch_quantized_max_pool3d(input, params, out_shape, out_int):
    """Run the pooling kernel over ``input``'s integer representation."""
    kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw = params
    in_n, in_c, in_d, in_h, in_w = input.shape
    _, _, out_d, out_h, out_w = out_shape

    # Alias the storage rather than calling ``int_repr()`` (``_int_view``):
    # the vendor ``int_repr`` registration exists, but a direct view also
    # avoids the dispatch entirely and works for strided inputs.
    int_repr = _int_view(input)

    neutral = _fast_pool3d_geometry(params, input, out_int)
    if neutral is not None:
        return _launch_quantized_max_pool3d_fast(int_repr, out_int, neutral)

    grid = lambda meta: (
        in_n * in_c,
        out_d
        * triton.cdiv(out_h, meta["BLOCK_H"])
        * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    quantized_max_pool3d_forward_kernel[grid](
        int_repr,
        out_int,
        *int_repr.stride(),
        *out_int.stride(),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
    )
    return out_int


def _alloc_quantized(out_shape, input, memory_format):
    """Allocate a per-tensor quantized output with the input's qparams."""
    return torch._empty_affine_quantized(
        tuple(out_shape),
        scale=float(input.q_scale()),
        zero_point=int(input.q_zero_point()),
        dtype=input.dtype,
        device=input.device,
        memory_format=memory_format,
    )


def _output_memory_format(input):
    """Match ATen: channels-last in, channels-last out; else contiguous."""
    if input.is_contiguous(memory_format=torch.channels_last_3d) and not (
        input.is_contiguous()
    ):
        return torch.channels_last_3d
    return torch.contiguous_format


def quantized_max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Apply 3-D max pooling to a per-tensor quantized tensor (XPU-safe)."""
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D")

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    _check_quantized_input(input)
    out_shape = _quantized_max_pool3d_output_shape(input, params, ceil_mode)

    out = _alloc_quantized(out_shape, input, _output_memory_format(input))

    if out.numel() != 0:
        _launch_quantized_max_pool3d(input, params, out_shape, _int_view(out))

    return out


def quantized_max_pool3d_out(
    input: torch.Tensor,
    kernel_size,
    stride=(),
    padding=0,
    dilation=1,
    ceil_mode=False,
    *,
    out: torch.Tensor,
):
    """``out`` variant of :func:`quantized_max_pool3d` (XPU-safe).

    Follows the same validation order as the autogenerated
    ``quantized_max_pool3d.out`` kernel. XDNN's quantized ``copy_`` neither
    adopts the source's scale/zero_point (ATen does) nor supports as-strided
    destinations, so:

    * when ``out`` already carries the input's quantization parameters, the
      pooled values are written straight through ``out``'s integer view --
      this keeps the alias contract (``res is out``) and preserves a strided
      destination's strides;
    * otherwise a freshly allocated result with the input's parameters is
      returned, mirroring what ATen leaves behind (``res is out`` does not
      hold in that case, but ``out`` is never partially overwritten).
    """
    logger.debug("GEMS_KUNLUNXIN QUANTIZED_MAX_POOL3D_OUT")

    params = _parse_pool3d_params(kernel_size, stride, padding, dilation)
    _check_quantized_input(input)
    out_shape = _quantized_max_pool3d_output_shape(input, params, ceil_mode)

    # ATen's generated ``resize_out`` checks dtype before device.
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {_aten_dtype_name(input.dtype)}, "
            f"but got {_aten_dtype_name(out.dtype)} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to have device {input.device}, "
            f"but got {out.device} instead"
        )

    if tuple(out.shape) != tuple(out_shape):
        _resize_quantized_out(out, out_shape)

    match_qparams = abs(float(out.q_scale()) - float(input.q_scale())) < 1e-9 and int(
        out.q_zero_point()
    ) == int(input.q_zero_point())

    if match_qparams:
        if out.numel() != 0:
            _launch_quantized_max_pool3d(input, params, out_shape, _int_view(out))
        return out

    return quantized_max_pool3d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def _int_repr_impl(t):
    """Replacement for XDNN's broken ``aten::int_repr`` on ``QuantizedCUDA``.

    Returns an *aliasing* view of the quantized tensor's storage with the
    corresponding plain integer dtype and the tensor's own sizes/strides --
    the semantics of ATen's ``int_repr`` (the integer representation shares
    the quantized storage). XDNN's kernel raises
    ``cudaErrorInvalidDeviceFunction`` for every quantized tensor, which also
    poisons the device context, so ``tensor.int_repr()`` is unusable as a
    reference on this stack without the override.
    """
    if not t.is_quantized:
        raise RuntimeError("int_repr expects a quantized tensor")
    int_dtype = _QINT_TO_INT.get(t.dtype)
    if int_dtype is None:
        raise RuntimeError(
            f"int_repr not implemented for '{t.dtype}' on kunlunxin; "
            "expected one of torch.quint8, torch.qint8, torch.qint32"
        )
    return torch.empty(0, dtype=int_dtype, device=t.device).set_(
        t.untyped_storage(), t.storage_offset(), tuple(t.shape), t.stride()
    )


# Quantized tensors carry ``DispatchKeySet(QuantizedCUDA, ...)``, so a ``CUDA``
# registration is silently ignored; ``QuantizedCUDA`` is the key that is hit.
_int_repr_lib = torch.library.Library("aten", "IMPL")
_int_repr_lib.impl("int_repr", _int_repr_impl, "QuantizedCUDA", allow_override=True)
