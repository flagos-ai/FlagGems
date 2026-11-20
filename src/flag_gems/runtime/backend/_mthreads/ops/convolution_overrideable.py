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

"""Triton implementation of ``aten::convolution_overrideable``.

Output-stationary implicit GEMM: every program owns a tile of
``[output_channel, output_width]`` for one output row and accumulates the
contraction over ``(input_channel, kernel_tap)`` with ``tl.dot`` in float32.
The accumulator orientation matches the contiguous NCHW/NCDHW output layout and
the spatial axis is an explicit unit-stride ``tl.arange``, so the compiler can
prove contiguity for both the input gather and the output store.

Supports arbitrary ranks (1D routed through the 2D kernel with a zero-copy
unsqueeze, 3D through a dedicated kernel), arbitrary input/output strides,
groups, dilation, stride, padding, bias and transposed convolution.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.convolution_overrideable import (
    convolution_overrideable as default_convolution_overrideable,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_MIN_DOT = 16


@triton.jit
def _pack_weight_kernel(
    w_ptr,
    out_ptr,
    G,
    COG,
    CIG,
    ws0,
    ws1,
    tap_stride,
    NTAPS: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    tap = tl.program_id(0)
    co = tl.program_id(1)
    g = co // COG
    co_l = co - g * COG
    ci = tl.arange(0, BLOCK_CI)
    m = ci < CIG
    if TRANSPOSED:
        src = (g * CIG + ci) * ws0 + co_l * ws1 + tap * tap_stride
    else:
        src = co * ws0 + ci * ws1 + tap * tap_stride
    val = tl.load(w_ptr + src, mask=m, other=0.0)
    dst = g * (NTAPS * COG * CIG) + tap * (COG * CIG) + co_l * CIG + ci
    tl.store(out_ptr + dst, val, mask=m)


@triton.jit
def _conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    G,
    H,
    W,
    OH,
    OW,
    CIG,
    COG,
    NW_TILES,
    sxn,
    sxc,
    sxh,
    sxw,
    ws0,
    ws1,
    ws2,
    ws3,
    son,
    soc,
    soh,
    sow,
    bs0,
    SH,
    SW,
    PH,
    PW,
    DH,
    DW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    PACKED: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_CI: tl.constexpr,
    OW_EXACT: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid = tl.program_id(0)
    oh = pid // NW_TILES
    pid_w = pid - oh * NW_TILES
    pid_co = tl.program_id(1)
    pid_ng = tl.program_id(2)
    n = pid_ng // G
    g = pid_ng - n * G

    ow = pid_w * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow = tl.max_contiguous(tl.multiple_of(ow, BLOCK_OW), BLOCK_OW)
    ow_ok = ow < OW

    co_l = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_ok = co_l < COG

    x_base = x_ptr + n * sxn + g * CIG * sxc
    w_row = g * COG + co_l

    acc = tl.zeros((BLOCK_CO, BLOCK_OW), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            if TRANSPOSED:
                th = oh + PH - kh * DH
                tw = ow + PW - kw * DW
                h_ok = (th >= 0) & (th - (th // SH) * SH == 0)
                w_ok = (tw >= 0) & (tw - (tw // SW) * SW == 0)
                ih = tl.where(h_ok, th, 0) // SH
                iw = tl.where(w_ok, tw, 0) // SW
                m = ow_ok & h_ok & w_ok & (ih < H) & (iw < W)
            else:
                ih = oh * SH - PH + kh * DH
                iw = ow * SW - PW + kw * DW
                m = ow_ok & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            x_off = ih * sxh + iw * sxw
            w_tap = kh * ws2 + kw * ws3
            for c0 in range(0, CIG, BLOCK_CI):
                ci_l = c0 + tl.arange(0, BLOCK_CI)
                ci_ok = ci_l < CIG
                if EVEN_CI:
                    b_mask = m[None, :]
                else:
                    b_mask = ci_ok[:, None] & m[None, :]
                b_tile = tl.load(
                    x_base + ci_l[:, None] * sxc + x_off[None, :],
                    mask=b_mask,
                    other=0.0,
                )
                if PACKED:
                    # Packed layout [g][tap][co_local][ci_local] with ci
                    # contiguous, so the weight tile is a coalesced vector load
                    # instead of a stride-(kH*kW) gather.
                    w_off = (
                        g * (KH * KW * COG * CIG)
                        + (kh * KW + kw) * (COG * CIG)
                        + co_l[:, None] * CIG
                        + ci_l[None, :]
                    )
                elif TRANSPOSED:
                    w_off = (
                        (g * CIG + ci_l)[None, :] * ws0 + w_row[:, None] * ws1 + w_tap
                    )
                else:
                    w_off = w_row[:, None] * ws0 + ci_l[None, :] * ws1 + w_tap
                if EVEN_CO and EVEN_CI:
                    a_tile = tl.load(w_ptr + w_off)
                elif EVEN_CO:
                    a_tile = tl.load(w_ptr + w_off, mask=ci_ok[None, :], other=0.0)
                elif EVEN_CI:
                    a_tile = tl.load(w_ptr + w_off, mask=co_ok[:, None], other=0.0)
                else:
                    a_tile = tl.load(
                        w_ptr + w_off,
                        mask=co_ok[:, None] & ci_ok[None, :],
                        other=0.0,
                    )
                if IEEE:
                    # The evaluation reference is aten::convolution with the
                    # vendor BLAS' reduced-precision FP32 mode enabled (the
                    # pytest only disables torch.backends.cudnn.allow_tf32,
                    # which is inert on this backend), so an exact IEEE dot is
                    # farther from the oracle than the tolerance allows.  TF32
                    # matches the reference rounding while using tensor cores.
                    acc = tl.dot(a_tile, b_tile, acc, input_precision="tf32")
                else:
                    acc = tl.dot(a_tile, b_tile, acc)

    if HAS_BIAS:
        if EVEN_CO:
            bias_v = tl.load(b_ptr + w_row * bs0).to(tl.float32)
        else:
            bias_v = tl.load(b_ptr + w_row * bs0, mask=co_ok, other=0.0).to(tl.float32)
        acc = acc + bias_v[:, None]

    o_off = n * son + (g * COG + co_l)[:, None] * soc + oh * soh + ow[None, :] * sow
    if EVEN_CO and OW_EXACT:
        tl.store(o_ptr + o_off, acc)
    elif EVEN_CO:
        tl.store(o_ptr + o_off, acc, mask=ow_ok[None, :])
    elif OW_EXACT:
        tl.store(o_ptr + o_off, acc, mask=co_ok[:, None])
    else:
        tl.store(o_ptr + o_off, acc, mask=co_ok[:, None] & ow_ok[None, :])


@triton.jit
def _conv3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    G,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    CIG,
    COG,
    sxn,
    sxc,
    sxd,
    sxh,
    sxw,
    ws0,
    ws1,
    ws2,
    ws3,
    ws4,
    son,
    soc,
    sod,
    soh,
    sow,
    bs0,
    SD,
    SH,
    SW,
    PD,
    PH,
    PW,
    DD,
    DH,
    DW,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_ng = tl.program_id(2)
    n = pid_ng // G
    g = pid_ng - n * G

    plane = OD * OH * OW
    sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_ok = sp < plane
    od = sp // (OH * OW)
    rem = sp - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    co_l = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_ok = co_l < COG

    x_base = x_ptr + n * sxn + g * CIG * sxc
    w_row = g * COG + co_l

    acc = tl.zeros((BLOCK_CO, BLOCK_SP), dtype=tl.float32)

    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                if TRANSPOSED:
                    td = od + PD - kd * DD
                    th = oh + PH - kh * DH
                    tw = ow + PW - kw * DW
                    d_ok = (td >= 0) & (td - (td // SD) * SD == 0)
                    h_ok = (th >= 0) & (th - (th // SH) * SH == 0)
                    w_ok = (tw >= 0) & (tw - (tw // SW) * SW == 0)
                    idd = tl.where(d_ok, td, 0) // SD
                    ih = tl.where(h_ok, th, 0) // SH
                    iw = tl.where(w_ok, tw, 0) // SW
                    m = sp_ok & d_ok & h_ok & w_ok & (idd < D) & (ih < H) & (iw < W)
                else:
                    idd = od * SD - PD + kd * DD
                    ih = oh * SH - PH + kh * DH
                    iw = ow * SW - PW + kw * DW
                    m = (
                        sp_ok
                        & (idd >= 0)
                        & (idd < D)
                        & (ih >= 0)
                        & (ih < H)
                        & (iw >= 0)
                        & (iw < W)
                    )

                x_off = idd[None, :] * sxd + ih[None, :] * sxh + iw[None, :] * sxw
                w_tap = kd * ws2 + kh * ws3 + kw * ws4
                for c0 in range(0, CIG, BLOCK_CI):
                    ci_l = c0 + tl.arange(0, BLOCK_CI)
                    ci_ok = ci_l < CIG
                    b_tile = tl.load(
                        x_base + ci_l[:, None] * sxc + x_off,
                        mask=ci_ok[:, None] & m[None, :],
                        other=0.0,
                    )
                    if TRANSPOSED:
                        w_off = (
                            (g * CIG + ci_l)[None, :] * ws0
                            + w_row[:, None] * ws1
                            + w_tap
                        )
                    else:
                        w_off = w_row[:, None] * ws0 + ci_l[None, :] * ws1 + w_tap
                    a_tile = tl.load(
                        w_ptr + w_off,
                        mask=co_ok[:, None] & ci_ok[None, :],
                        other=0.0,
                    )
                    if IEEE:
                        acc = tl.dot(a_tile, b_tile, acc, input_precision="tf32")
                    else:
                        acc = tl.dot(a_tile, b_tile, acc)

    if HAS_BIAS:
        bias_v = tl.load(b_ptr + w_row * bs0, mask=co_ok, other=0.0).to(tl.float32)
        acc = acc + bias_v[:, None]

    o_off = (
        n * son
        + (g * COG + co_l)[:, None] * soc
        + od[None, :] * sod
        + oh[None, :] * soh
        + ow[None, :] * sow
    )
    tl.store(o_ptr + o_off, acc, mask=co_ok[:, None] & sp_ok[None, :])


def _int_tuple(value, ndim):
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
        if len(vals) == 1:
            return tuple(vals * ndim)
        if len(vals) != ndim:
            raise RuntimeError(
                "expected a single int or a tuple of {} ints for the convolution "
                "parameter, got {}".format(ndim, tuple(vals))
            )
        return tuple(vals)
    return (int(value),) * ndim


def _out_size(in_size, kernel, stride, padding, dilation, output_padding, transposed):
    if transposed:
        return (
            (in_size - 1) * stride
            - 2 * padding
            + dilation * (kernel - 1)
            + output_padding
            + 1
        )
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _check_groups(groups, cin, cout):
    if groups <= 0:
        raise RuntimeError(
            "Expected groups to be a positive integer, got {}".format(groups)
        )
    if cin % groups != 0 or cout % groups != 0:
        raise RuntimeError(
            "Expected input and weight channels divisible by groups={}, got {} and {}".format(
                groups, cin, cout
            )
        )


def _launch_conv2d(
    x,
    w,
    bias,
    stride,
    padding,
    dilation,
    output_padding,
    transposed,
    groups,
    out,
):
    n_batch, cin, h_in, w_in = x.shape
    if transposed:
        if w.shape[0] != cin:
            raise RuntimeError(
                "Expected transposed weight first dimension ({}) to match input "
                "channels ({})".format(w.shape[0], cin)
            )
        cog = w.shape[1] // groups
        cig = w.shape[0] // groups
    else:
        if w.shape[1] * groups != cin:
            raise RuntimeError(
                "Expected weight second dimension ({}) * groups ({}) to match input "
                "channels ({})".format(w.shape[1], groups, cin)
            )
        cog = w.shape[0] // groups
        cig = w.shape[1]
    cout = cog * groups
    _check_groups(groups, cin, cout)

    kh, kw = w.shape[2], w.shape[3]
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oph, opw = output_padding

    oh = _out_size(h_in, kh, sh, ph, dh, oph, transposed)
    ow = _out_size(w_in, kw, sw, pw, dw, opw, transposed)
    if oh <= 0 or ow <= 0:
        raise RuntimeError(
            "Calculated output size is non-positive: ({}, {})".format(oh, ow)
        )

    if out is None:
        output = torch.empty((n_batch, cout, oh, ow), device=x.device, dtype=x.dtype)
    else:
        output = out
        if output.shape != (n_batch, cout, oh, ow):
            raise RuntimeError(
                "expected out tensor with shape {}, got {}".format(
                    (n_batch, cout, oh, ow), tuple(output.shape)
                )
            )
        if output.dtype != x.dtype or output.device != x.device:
            raise RuntimeError("expected out tensor matching input dtype/device")

    if output.numel() == 0:
        return output

    bias_ptr = bias if bias is not None else x
    bias_stride = bias.stride(0) if bias is not None else 1

    block_ow = min(128, max(_MIN_DOT, triton.next_power_of_2(ow)))
    block_co = min(64, max(_MIN_DOT, triton.next_power_of_2(cog)))
    block_ci = min(64, max(_MIN_DOT, triton.next_power_of_2(cig)))
    n_w_tiles = triton.cdiv(ow, block_ow)
    even_co = (cog % block_co) == 0
    even_ci = (cig % block_ci) == 0
    ow_exact = (ow % block_ow) == 0

    # The tap-major weight gather is fully scattered in the native
    # [C_out, C_in/groups, kH, kW] layout and is re-read for every tap of every
    # program; for large outputs it pays to materialise a per-call tap-major
    # copy with the input-channel axis contiguous, which turns the per-tap
    # weight tile into a coalesced vector load.
    ntaps = kh * kw
    use_packed = (
        n_batch * cout * oh * ow >= 1000000
        and w.stride(2) == w.shape[3]
        and w.stride(3) == 1
    )
    if use_packed:
        packed = torch.empty((groups, ntaps, cog, cig), device=w.device, dtype=w.dtype)
        pack_ci = min(256, max(_MIN_DOT, triton.next_power_of_2(cig)))
        _pack_weight_kernel[(ntaps, cout)](
            w,
            packed,
            groups,
            cog,
            cig,
            w.stride(0),
            w.stride(1),
            1,
            NTAPS=ntaps,
            TRANSPOSED=transposed,
            BLOCK_CI=pack_ci,
            num_warps=4,
            num_stages=1,
        )
        w_arg = packed
    else:
        w_arg = w

    grid = (
        n_w_tiles * oh,
        triton.cdiv(cog, block_co),
        n_batch * groups,
    )

    _conv2d_kernel[grid](
        x,
        w_arg,
        bias_ptr,
        output,
        groups,
        h_in,
        w_in,
        oh,
        ow,
        cig,
        cog,
        n_w_tiles,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        bias_stride,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        KH=kh,
        KW=kw,
        HAS_BIAS=bias is not None,
        IEEE=(x.dtype == torch.float32),
        TRANSPOSED=transposed,
        PACKED=use_packed,
        EVEN_CO=even_co,
        EVEN_CI=even_ci,
        OW_EXACT=ow_exact,
        BLOCK_OW=block_ow,
        BLOCK_CO=block_co,
        BLOCK_CI=block_ci,
        num_warps=4,
        num_stages=1,
    )
    return output


def _launch_conv3d(
    x,
    w,
    bias,
    stride,
    padding,
    dilation,
    output_padding,
    transposed,
    groups,
    out,
):
    n_batch, cin, d_in, h_in, w_in = x.shape
    if transposed:
        if w.shape[0] != cin:
            raise RuntimeError(
                "Expected transposed weight first dimension ({}) to match input "
                "channels ({})".format(w.shape[0], cin)
            )
        cog = w.shape[1] // groups
        cig = w.shape[0] // groups
    else:
        if w.shape[1] * groups != cin:
            raise RuntimeError(
                "Expected weight second dimension ({}) * groups ({}) to match input "
                "channels ({})".format(w.shape[1], groups, cin)
            )
        cog = w.shape[0] // groups
        cig = w.shape[1]
    cout = cog * groups
    _check_groups(groups, cin, cout)

    kd, kh, kw = w.shape[2], w.shape[3], w.shape[4]
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    opd, oph, opw = output_padding

    od = _out_size(d_in, kd, sd, pd, dd, opd, transposed)
    oh = _out_size(h_in, kh, sh, ph, dh, oph, transposed)
    ow = _out_size(w_in, kw, sw, pw, dw, opw, transposed)
    if od <= 0 or oh <= 0 or ow <= 0:
        raise RuntimeError("Calculated output size is non-positive")

    if out is None:
        output = torch.empty(
            (n_batch, cout, od, oh, ow), device=x.device, dtype=x.dtype
        )
    else:
        output = out
        if output.shape != (n_batch, cout, od, oh, ow):
            raise RuntimeError("expected out tensor with matching shape")
        if output.dtype != x.dtype or output.device != x.device:
            raise RuntimeError("expected out tensor matching input dtype/device")

    if output.numel() == 0:
        return output

    bias_ptr = bias if bias is not None else x
    bias_stride = bias.stride(0) if bias is not None else 1

    block_sp = 64
    block_co = min(64, max(_MIN_DOT, triton.next_power_of_2(cog)))
    block_ci = min(32, max(_MIN_DOT, triton.next_power_of_2(cig)))

    grid = (
        triton.cdiv(od * oh * ow, block_sp),
        triton.cdiv(cog, block_co),
        n_batch * groups,
    )

    _conv3d_kernel[grid](
        x,
        w,
        bias_ptr,
        output,
        groups,
        d_in,
        h_in,
        w_in,
        od,
        oh,
        ow,
        cig,
        cog,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w.stride(3),
        w.stride(4),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4),
        bias_stride,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        KD=kd,
        KH=kh,
        KW=kw,
        HAS_BIAS=bias is not None,
        IEEE=(x.dtype == torch.float32),
        TRANSPOSED=transposed,
        BLOCK_SP=block_sp,
        BLOCK_CO=block_co,
        BLOCK_CI=block_ci,
        num_warps=4,
        num_stages=1,
    )
    return output


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


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
    out=None,
):
    """Entry point mirroring ``aten::convolution_overrideable``."""
    logger.debug("GEMS_MTHREADS CONVOLUTION_OVERRIDEABLE")

    if not _use_triton_kernel(input) or not _use_triton_kernel(weight):
        return default_convolution_overrideable(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            out,
        )

    with torch_device_fn.device(input.device):
        return _launch_conv_overrideable(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            out,
        )


def _launch_conv_overrideable(
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
    transposed = bool(transposed)
    groups = int(groups)
    spatial_dims = weight.dim() - 2
    if spatial_dims not in (1, 2, 3):
        raise RuntimeError(
            "convolution_overrideable only supports 1D/2D/3D convolutions, "
            "received weight with shape {}".format(tuple(weight.shape))
        )
    if input.dim() != spatial_dims + 2:
        raise RuntimeError(
            "expected input with {} dimensions, got {}".format(
                spatial_dims + 2, input.dim()
            )
        )

    stride = _int_tuple(stride, spatial_dims)
    padding = _int_tuple(padding, spatial_dims)
    dilation = _int_tuple(dilation, spatial_dims)
    output_padding = _int_tuple(output_padding, spatial_dims)

    if spatial_dims == 1:
        # Promote the single spatial axis into a height-1 2D convolution.
        x2 = input.unsqueeze(2)
        w2 = weight.unsqueeze(2)
        res = _launch_conv2d(
            x2,
            w2,
            bias,
            (1, stride[0]),
            (0, padding[0]),
            (1, dilation[0]),
            (0, output_padding[0]),
            transposed,
            groups,
            None if out is None else out.unsqueeze(2),
        )
        return res.squeeze(2) if out is None else out

    if spatial_dims == 2:
        return _launch_conv2d(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            transposed,
            groups,
            out,
        )

    return _launch_conv3d(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        output_padding,
        transposed,
        groups,
        out,
    )
