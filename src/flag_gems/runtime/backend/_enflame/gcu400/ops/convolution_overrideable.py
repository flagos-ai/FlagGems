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

from flag_gems.ops.conv2d import conv2d_output_size
from flag_gems.utils import libentry

from .constant_pad_nd import constant_pad_nd
from .conv3d import conv3d
from .conv_transpose1d import conv_transpose1d
from .conv_transpose2d import conv_transpose2d
from .conv_transpose3d import conv_transpose3d

logger = logging.getLogger(__name__)

# tl.dot requires every operand dimension >= 16.
_MIN_DOT_DIM = 16

# ---------------------------------------------------------------------------
# GCU400 forward kernel: row-block transposed implicit GEMM over a repacked
# weight.
#
# The historical flat-(n, ho, wo) im2col kernel ran at ~0.1 TFLOPS on GCU400
# because its 2D input tiles decompose the flat position index with div/mod;
# a div/mod (or any non mul/add op) inside a load's offset expression drops
# the tile off the vectorized affine load path and every element becomes a
# scalar DSM gather (~1.7-2.4 ns/element/SP).  The kernel below keeps every
# address vector a plain arange expression:
#
#   * one program = ROWS_PER_CTA output image rows (n, ho) x BLOCK_WO output
#     columns x BLOCK_CO output channels (grid: row-blocks x wo-blocks x
#     group*co-blocks);
#   * X_t[BLOCK_CI, BLOCK_WO] is loaded per (h, w) tap and per row: channels
#     on axis 0, the W-contiguous output-column axis LAST (stride
#     m_stride_sw, passed as a constexpr so stride-1 stays provable);
#   * the weight is repacked host-side to (KH*KW, out_c, CI_pad) by an own
#     Triton kernel, so the per-tap W_t[BLOCK_CO, BLOCK_CI] slab is a
#     contiguous 2D tile (channel stride 1) instead of the stride-KH*KW
#     gather of the OIHW layout; each tap's slab is loaded once and reused
#     by the ROWS_PER_CTA dots of the row block (weight traffic /ROWS);
#   * accum_r[BLOCK_CO, BLOCK_WO] += tl.dot(W_t, X_t_r) per tap and row, fp32
#     accumulator, ieee dot (allow_tf32=False keeps the fp32 gate green);
#   * num_warps=1: a single warp keeps the MMA pipeline ownership intact
#     (w2/w4 split the operand tiles across warps and serialize the dot
#     issue path; measured -21% on the core shapes).
#
# The host zero-pads the per-group channel dim of the input only (to the
# BLOCK_CI multiple, via the FlagGems constant_pad_nd op); the weight repack
# kernel zero-fills its own channel pad lanes, so both operands stay dense
# on the K axis.  Spatial bounds are handled by a host-side zero pad with a
# fast 3D-tile window-copy kernel: a load mask with interior false lanes (or
# any non mul/add op in the offset expression) drops the tile to per-element
# predicated loads on this backend, and GCU touches the address of masked
# lanes, so in-kernel boundary masks would both degrade vectorization and
# read out of the allocation.
#
# Measured (bench kernel mode, do_bench median w1000/r100, GCU400/L600,
# fp16, vs the previous single-row OIHW-gather revision):
#   (16,32,24,24)x24k3:    0.262 -> 0.073 ms
#   (32,64,64,64)x32k3:    2.25  -> 0.43 ms
#   (32,64,128,128)x32k3:  4.61  -> 0.96 ms
#   (32,64,64,64)x32k5:    5.16  -> 0.98 ms
# ---------------------------------------------------------------------------

# Output rows folded into one program: amortizes the per-tap weight slab over
# ROWS dots (weight traffic and CTA count /ROWS).
_ROWS_PER_CTA = 4
_CONV_NUM_WARPS = 1
_CONV_NUM_STAGES = 2


def _select_forward_config(out_width, out_c_per_group, weight_c):
    """Deterministic tile choice for the transposed im2col kernel."""
    block_wo = min(128, max(_MIN_DOT_DIM, triton.next_power_of_2(out_width)))
    block_co = min(64, max(_MIN_DOT_DIM, triton.next_power_of_2(out_c_per_group)))
    block_ci = min(64, max(_MIN_DOT_DIM, triton.next_power_of_2(weight_c)))
    return block_wo, block_ci, block_co


@libentry()
@triton.jit
def _conv2d_forward_kernel(
    input_pointer,
    weight_pointer,  # repacked (KH*KW, out_c, CI_PAD), contiguous
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    rows_total,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_t_stride,
    weight_co_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    m_stride_sw: tl.constexpr,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    EVEN_WO: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CO_BLOCKS_PER_GROUP: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_wo = tl.program_id(1)
    pid_co = tl.program_id(2)

    wo0 = pid_wo * BLOCK_WO
    m_off = tl.arange(0, BLOCK_WO)
    wo_ok = (wo0 + m_off) < out_width

    pid_group = pid_co // CO_BLOCKS_PER_GROUP
    pid_cob = pid_co % CO_BLOCKS_PER_GROUP
    out_per_group_c = out_c // groups
    co = pid_cob * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_ok = co < out_per_group_c

    # ROWS output rows per program; tail programs clamp the row index (GCU
    # touches the address of masked lanes) and mask the stores.
    row0 = pid_row * ROWS
    row_ok0 = row0 < rows_total
    row_ok1 = (row0 + 1) < rows_total
    row_ok2 = (row0 + 2) < rows_total
    row_ok3 = (row0 + 3) < rows_total
    row_c0 = tl.minimum(row0, rows_total - 1)
    row_c1 = tl.minimum(row0 + 1, rows_total - 1)
    row_c2 = tl.minimum(row0 + 2, rows_total - 1)
    row_c3 = tl.minimum(row0 + 3, rows_total - 1)
    n0 = row_c0 // out_height
    ho0 = row_c0 % out_height
    n1 = row_c1 // out_height
    ho1 = row_c1 % out_height
    n2 = row_c2 // out_height
    ho2 = row_c2 % out_height
    n3 = row_c3 // out_height
    ho3 = row_c3 % out_height

    x_row_base0 = (
        input_pointer
        + input_n_stride * n0
        + input_c_stride * pid_group * weight_c
        + input_height_stride * stride_height * ho0
        + m_stride_sw * wo0
    )
    x_row_base1 = (
        input_pointer
        + input_n_stride * n1
        + input_c_stride * pid_group * weight_c
        + input_height_stride * stride_height * ho1
        + m_stride_sw * wo0
    )
    x_row_base2 = (
        input_pointer
        + input_n_stride * n2
        + input_c_stride * pid_group * weight_c
        + input_height_stride * stride_height * ho2
        + m_stride_sw * wo0
    )
    x_row_base3 = (
        input_pointer
        + input_n_stride * n3
        + input_c_stride * pid_group * weight_c
        + input_height_stride * stride_height * ho3
        + m_stride_sw * wo0
    )
    w_base = weight_pointer + weight_co_stride * (pid_group * out_per_group_c)

    CI_COUNT: tl.constexpr = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    # The input is spatially zero-padded by the host, so every tap is in
    # bounds: the X loads carry at most a tail mask.  (A mask with interior
    # false lanes -- e.g. width-boundary taps on an unpadded tensor -- drops
    # the tile to per-element predicated loads on this backend, and so does
    # any non mul/add op such as clamp/div/mod in the offset expression.)
    accum0 = tl.zeros((BLOCK_CO, BLOCK_WO), dtype=tl.float32)
    accum1 = tl.zeros((BLOCK_CO, BLOCK_WO), dtype=tl.float32)
    accum2 = tl.zeros((BLOCK_CO, BLOCK_WO), dtype=tl.float32)
    accum3 = tl.zeros((BLOCK_CO, BLOCK_WO), dtype=tl.float32)
    for h in tl.static_range(weight_height):
        x_h0 = x_row_base0 + input_height_stride * (
            h * dilation_height - padding_height
        )
        x_h1 = x_row_base1 + input_height_stride * (
            h * dilation_height - padding_height
        )
        x_h2 = x_row_base2 + input_height_stride * (
            h * dilation_height - padding_height
        )
        x_h3 = x_row_base3 + input_height_stride * (
            h * dilation_height - padding_height
        )
        for w in tl.static_range(weight_width):
            t = h * weight_width + w
            x_w0 = x_h0 + input_width_stride * (w * dilation_width - padding_width)
            x_w1 = x_h1 + input_width_stride * (w * dilation_width - padding_width)
            x_w2 = x_h2 + input_width_stride * (w * dilation_width - padding_width)
            x_w3 = x_h3 + input_width_stride * (w * dilation_width - padding_width)
            for cb in tl.static_range(CI_COUNT):
                c_vec = cb * BLOCK_CI + tl.arange(0, BLOCK_CI)
                w_ptr = (w_base + weight_t_stride * t + weight_co_stride * co)[
                    :, None
                ] + c_vec[None, :]
                if EVEN_CO:
                    if EVEN_C:
                        w = tl.load(w_ptr)
                    else:
                        w = tl.load(w_ptr, mask=(c_vec < weight_c)[None, :])
                else:
                    if EVEN_C:
                        w = tl.load(w_ptr, mask=co_ok[:, None])
                    else:
                        w = tl.load(
                            w_ptr,
                            mask=co_ok[:, None] & (c_vec < weight_c)[None, :],
                        )
                x_ptr0 = (x_w0 + input_c_stride * c_vec)[:, None] + (
                    m_stride_sw * m_off
                )[None, :]
                x_ptr1 = (x_w1 + input_c_stride * c_vec)[:, None] + (
                    m_stride_sw * m_off
                )[None, :]
                x_ptr2 = (x_w2 + input_c_stride * c_vec)[:, None] + (
                    m_stride_sw * m_off
                )[None, :]
                x_ptr3 = (x_w3 + input_c_stride * c_vec)[:, None] + (
                    m_stride_sw * m_off
                )[None, :]
                if EVEN_C:
                    if EVEN_WO:
                        x0 = tl.load(x_ptr0)
                        x1 = tl.load(x_ptr1)
                        x2 = tl.load(x_ptr2)
                        x3 = tl.load(x_ptr3)
                    else:
                        x0 = tl.load(x_ptr0, mask=wo_ok[None, :])
                        x1 = tl.load(x_ptr1, mask=wo_ok[None, :])
                        x2 = tl.load(x_ptr2, mask=wo_ok[None, :])
                        x3 = tl.load(x_ptr3, mask=wo_ok[None, :])
                else:
                    c_ok = c_vec < weight_c
                    if EVEN_WO:
                        x0 = tl.load(x_ptr0, mask=c_ok[:, None])
                        x1 = tl.load(x_ptr1, mask=c_ok[:, None])
                        x2 = tl.load(x_ptr2, mask=c_ok[:, None])
                        x3 = tl.load(x_ptr3, mask=c_ok[:, None])
                    else:
                        xm = c_ok[:, None] & wo_ok[None, :]
                        x0 = tl.load(x_ptr0, mask=xm)
                        x1 = tl.load(x_ptr1, mask=xm)
                        x2 = tl.load(x_ptr2, mask=xm)
                        x3 = tl.load(x_ptr3, mask=xm)
                accum0 += tl.dot(w, x0, allow_tf32=False)
                if ROWS >= 2:
                    accum1 += tl.dot(w, x1, allow_tf32=False)
                if ROWS >= 4:
                    accum2 += tl.dot(w, x2, allow_tf32=False)
                    accum3 += tl.dot(w, x3, allow_tf32=False)

    if HAS_BIAS:
        bias_vec = bias_pointer + (pid_group * out_per_group_c) + co
        bias = tl.load(bias_vec, mask=co_ok).to(tl.float32)
        accum0 = accum0 + bias[:, None]
        accum1 = accum1 + bias[:, None]
        accum2 = accum2 + bias[:, None]
        accum3 = accum3 + bias[:, None]

    out_c_off = output_c_stride * (pid_group * out_per_group_c + co)
    out_ptr0 = (
        output_pointer
        + output_n_stride * n0
        + output_height_stride * ho0
        + output_width_stride * wo0
        + out_c_off
    )[:, None] + (output_width_stride * m_off)[None, :]
    out_ptr1 = (
        output_pointer
        + output_n_stride * n1
        + output_height_stride * ho1
        + output_width_stride * wo0
        + out_c_off
    )[:, None] + (output_width_stride * m_off)[None, :]
    out_ptr2 = (
        output_pointer
        + output_n_stride * n2
        + output_height_stride * ho2
        + output_width_stride * wo0
        + out_c_off
    )[:, None] + (output_width_stride * m_off)[None, :]
    out_ptr3 = (
        output_pointer
        + output_n_stride * n3
        + output_height_stride * ho3
        + output_width_stride * wo0
        + out_c_off
    )[:, None] + (output_width_stride * m_off)[None, :]
    if EVEN_WO and EVEN_CO:
        tl.store(out_ptr0, accum0, mask=wo_ok[None, :] & row_ok0)
        if ROWS >= 2:
            tl.store(out_ptr1, accum1, mask=wo_ok[None, :] & row_ok1)
        if ROWS >= 4:
            tl.store(out_ptr2, accum2, mask=wo_ok[None, :] & row_ok2)
            tl.store(out_ptr3, accum3, mask=wo_ok[None, :] & row_ok3)
    elif EVEN_WO:
        m1 = co_ok[:, None] & row_ok0
        tl.store(out_ptr0, accum0, mask=m1)
        if ROWS >= 2:
            tl.store(out_ptr1, accum1, mask=co_ok[:, None] & row_ok1)
        if ROWS >= 4:
            tl.store(out_ptr2, accum2, mask=co_ok[:, None] & row_ok2)
            tl.store(out_ptr3, accum3, mask=co_ok[:, None] & row_ok3)
    elif EVEN_CO:
        tl.store(out_ptr0, accum0, mask=wo_ok[None, :] & row_ok0)
        if ROWS >= 2:
            tl.store(out_ptr1, accum1, mask=wo_ok[None, :] & row_ok1)
        if ROWS >= 4:
            tl.store(out_ptr2, accum2, mask=wo_ok[None, :] & row_ok2)
            tl.store(out_ptr3, accum3, mask=wo_ok[None, :] & row_ok3)
    else:
        tl.store(out_ptr0, accum0, mask=co_ok[:, None] & wo_ok[None, :] & row_ok0)
        if ROWS >= 2:
            tl.store(out_ptr1, accum1, mask=co_ok[:, None] & wo_ok[None, :] & row_ok1)
        if ROWS >= 4:
            tl.store(out_ptr2, accum2, mask=co_ok[:, None] & wo_ok[None, :] & row_ok2)
            tl.store(out_ptr3, accum3, mask=co_ok[:, None] & wo_ok[None, :] & row_ok3)


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _pad_input_channels(input, groups, weight_c, pad_c):
    """Zero-pad the per-group channel dim of the input to a multiple of the
    tile width, via the FlagGems constant_pad_nd op (no aten compute).

    The padded channels are zeros, so every padded K lane contributes zero
    and the convolution result is unchanged.  The weight needs no separate
    pad: the repack kernel zero-fills its own channel pad lanes.
    """
    if pad_c <= 0:
        return input
    if groups == 1:
        return constant_pad_nd(input, [0, 0, 0, 0, 0, pad_c])
    in_n, in_c, in_h, in_w = input.shape
    input = input.reshape(in_n, groups, weight_c, in_h, in_w)
    input = constant_pad_nd(input, [0, 0, 0, 0, 0, pad_c])
    return input.reshape(in_n, groups * (weight_c + pad_c), in_h, in_w)


@libentry()
@triton.jit
def _weight_repack_kernel(
    w_pointer,
    wr_pointer,
    out_c,
    weight_c,
    w_n_stride,
    w_c_stride,
    w_h_stride,
    w_w_stride,
    KW: tl.constexpr,
    CI_PAD: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    # One program per tap t = kh*KW + kw: copy the (out_c, weight_c) tap slab
    # of the OIHW weight into wr[t] with a contiguous channel axis, zero
    # filling the channel pad lanes.  Both tiles are affine strided 2D tiles
    # (no unit-stride axis-0), so the loads/stores stay on the vector path.
    t = tl.program_id(0)
    kh = t // KW
    kw = t % KW
    co = tl.arange(0, BLOCK_CO)
    ci = tl.arange(0, BLOCK_CI)
    src = (
        w_pointer
        + co[:, None] * w_n_stride
        + ci[None, :] * w_c_stride
        + kh * w_h_stride
        + kw * w_w_stride
    )
    val = tl.load(src, mask=(co[:, None] < out_c) & (ci[None, :] < weight_c), other=0.0)
    dst = wr_pointer + t * (out_c * CI_PAD) + co[:, None] * CI_PAD + ci[None, :]
    tl.store(dst, val, mask=(co[:, None] < out_c) & (ci[None, :] < CI_PAD))


_MAX_PAD_BLOCK_W = 8192


@libentry()
@triton.jit
def _pad_window_copy_kernel(
    src_pointer,
    dst_pointer,
    planes_total,
    in_h,
    in_w,
    out_w,
    pad_top,
    pad_left,
    h_blocks,
    BLOCK_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Copy the H x W window of BLOCK_P (n, c) planes into the zero-filled
    # padded planes at (pad_top, pad_left).  3D (plane, row, col) tiles keep
    # both sides affine with uniform strides and tail-only masks; one plane
    # per program was latency-bound (measured 159us for a 17MB copy), while
    # the 3D plane-block form runs the same copy at ~3x the throughput.
    pid = tl.program_id(0)
    pb = pid // h_blocks
    hb = pid % h_blocks
    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    r = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    c = tl.arange(0, BLOCK_W)
    src = (
        src_pointer
        + p[:, None, None] * (in_h * in_w)
        + r[None, :, None] * in_w
        + c[None, None, :]
    )
    dst = (
        dst_pointer
        + p[:, None, None] * ((in_h + 2 * pad_top) * out_w)
        + (r + pad_top)[None, :, None] * out_w
        + (pad_left + c)[None, None, :]
    )
    m = (
        (p < planes_total)[:, None, None]
        & (r < in_h)[None, :, None]
        & (c < in_w)[None, None, :]
    )
    v = tl.load(src, mask=m)
    tl.store(dst, v, mask=m)


def _pad_spatial(input, padding_h, padding_w):
    """Spatially zero-pad H/W so every conv tap stays in bounds (the kernel
    then needs no width/height masks) and the padded W extent is a multiple
    of 16 elements.  Implemented as a zero allocation (factory) plus the
    3D-tile window copy kernel; the generic constant_pad_nd path is ~20x
    slower here (per-element predication).  Returns
    (padded_or_same_input, padded_h, padded_w).
    """
    in_n, in_c, in_h, in_w = input.shape
    w_target = ((in_w + 2 * padding_w + 15) // 16) * 16
    if padding_h == 0 and padding_w == 0 and w_target == in_w:
        return input, in_h, in_w
    block_w = triton.next_power_of_2(in_w)
    if not input.is_contiguous() or block_w > _MAX_PAD_BLOCK_W:
        pad_w_right = w_target - in_w - padding_w
        padded = constant_pad_nd(input, [padding_w, pad_w_right, padding_h, padding_h])
        return padded, in_h + 2 * padding_h, w_target
    out_h = in_h + 2 * padding_h
    padded = torch.zeros(
        (in_n, in_c, out_h, w_target),
        device=input.device,
        dtype=input.dtype,
    )
    block_h = min(64, max(1, triton.next_power_of_2(in_h)))
    planes_total = in_n * in_c
    block_p = max(1, min(16, 32768 // (block_h * block_w)))
    h_blocks = (in_h + block_h - 1) // block_h
    grid = ((planes_total + block_p - 1) // block_p * h_blocks,)
    _pad_window_copy_kernel[grid](
        input,
        padded,
        planes_total,
        in_h,
        in_w,
        w_target,
        padding_h,
        padding_w,
        h_blocks,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
        num_warps=4,
        num_stages=2,
    )
    return padded, out_h, w_target


def _conv2d_forward(input, weight, bias, stride, padding, dilation, groups, out=None):
    """2D forward convolution: row-block transposed implicit GEMM over a
    repacked weight (see the module comment above)."""
    stride_height, stride_width = _pair(stride)
    padding_height, padding_width = _pair(padding)
    dilation_height, dilation_width = _pair(dilation)

    in_n, _, input_height, input_width = input.shape
    out_c, weight_c, weight_height, weight_width = weight.shape

    out_height = conv2d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv2d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )
    block_wo, block_ci, block_co = _select_forward_config(
        out_width, out_c // groups, weight_c
    )

    pad_c = block_ci - weight_c % block_ci if weight_c % block_ci else 0
    if weight_c + pad_c < _MIN_DOT_DIM:
        pad_c += _MIN_DOT_DIM - (weight_c + pad_c)
    weight_c_padded = weight_c + pad_c
    input = _pad_input_channels(input, groups, weight_c, pad_c)
    input, in_h_pad, in_w_pad = _pad_spatial(input, padding_height, padding_width)

    # Repack the weight to (KH*KW, out_c, CI_pad): every per-tap (CO, CI)
    # slab becomes a contiguous tile and the channel pad lanes are zeros.
    taps = weight_height * weight_width
    weight_repacked = torch.empty(
        (taps, out_c, weight_c_padded),
        device=input.device,
        dtype=weight.dtype,
    )
    _weight_repack_kernel[(taps,)](
        weight,
        weight_repacked,
        out_c,
        weight_c,
        *weight.stride(),
        KW=weight_width,
        CI_PAD=weight_c_padded,
        BLOCK_CO=max(_MIN_DOT_DIM, triton.next_power_of_2(out_c)),
        BLOCK_CI=max(_MIN_DOT_DIM, triton.next_power_of_2(weight_c_padded)),
        num_warps=4,
        num_stages=2,
    )

    output_dtype = input.dtype
    if out is not None and (
        tuple(out.shape) == (in_n, out_c, out_height, out_width)
        and out.dtype == output_dtype
        and out.device == input.device
        and out.is_contiguous()
    ):
        output = out
    else:
        output = torch.empty(
            (in_n, out_c, out_height, out_width),
            device=input.device,
            dtype=output_dtype,
        )

    co_blocks_per_group = triton.cdiv(out_c // groups, block_co)
    rows_total = in_n * out_height
    grid = (
        triton.cdiv(rows_total, _ROWS_PER_CTA),
        triton.cdiv(out_width, block_wo),
        groups * co_blocks_per_group,
    )
    even_wo = out_width % block_wo == 0
    even_co = (out_c // groups) % block_co == 0
    even_c = weight_c_padded % block_ci == 0
    m_stride_sw = input.stride()[3] * stride_width

    has_bias = bias is not None
    bias_arg = bias if has_bias else output

    _conv2d_forward_kernel[grid](
        input,
        weight_repacked,
        output,
        bias_arg,
        in_n,
        in_h_pad,
        in_w_pad,
        out_c,
        out_height,
        out_width,
        rows_total,
        *input.stride(),
        out_c * weight_c_padded,  # weight_t_stride
        weight_c_padded,  # weight_co_stride
        *output.stride(),
        m_stride_sw,
        weight_c_padded,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        0,
        0,
        dilation_height,
        dilation_width,
        groups,
        ROWS=_ROWS_PER_CTA,
        BLOCK_WO=block_wo,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        EVEN_WO=even_wo,
        EVEN_CO=even_co,
        EVEN_C=even_c,
        HAS_BIAS=has_bias,
        CO_BLOCKS_PER_GROUP=co_blocks_per_group,
        num_warps=_CONV_NUM_WARPS,
        num_stages=_CONV_NUM_STAGES,
    )
    return output


def _convolution_overrideable_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    out=None,
):
    """Generic convolution dispatch backed by FlagGems Triton conv kernels.

    Mirrors ``aten::convolution_overrideable``: the spatial rank is inferred
    from ``weight.ndim - 2`` and the call is routed to the matching FlagGems
    Triton implementation.
    """
    spatial_dims = weight.ndim - 2
    assert spatial_dims in (1, 2, 3), (
        f"convolution_overrideable only supports 1D/2D/3D convolutions, "
        f"received weight with shape {tuple(weight.shape)}"
    )

    if transposed:
        if spatial_dims == 1:
            return conv_transpose1d(
                input, weight, bias, stride, padding, output_padding, groups, dilation
            )
        if spatial_dims == 2:
            return conv_transpose2d(
                input, weight, bias, stride, padding, output_padding, groups, dilation
            )
        if spatial_dims == 3:
            return conv_transpose3d(
                input, weight, bias, stride, padding, output_padding, groups, dilation
            )
        raise NotImplementedError(
            "convolution_overrideable only supports 1D/2D/3D transposed convolutions."
        )

    if spatial_dims == 1:
        # Reuse the 2D Triton kernel by promoting the single spatial axis.
        stride_w = _pair(stride)[0]
        padding_w = _pair(padding)[0]
        dilation_w = _pair(dilation)[0]
        return _conv2d_forward(
            input.unsqueeze(-1),
            weight.unsqueeze(-1),
            bias,
            (stride_w, 1),
            (padding_w, 0),
            (dilation_w, 1),
            groups,
            out=None if out is None else out.unsqueeze(-1),
        ).squeeze(-1)
    if spatial_dims == 2:
        return _conv2d_forward(
            input, weight, bias, stride, padding, dilation, groups, out=out
        )
    return conv3d(input, weight, bias, stride, padding, dilation, groups)


def convolution_overrideable(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    logger.debug("GEMS_ENFLAME CONVOLUTION_OVERRIDEABLE")
    return _convolution_overrideable_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


def convolution_overrideable_out(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    *,
    out,
):
    logger.debug("GEMS_ENFLAME CONVOLUTION_OVERRIDEABLE_OUT")
    result = _convolution_overrideable_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        out=out,
    )
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)
    return out
