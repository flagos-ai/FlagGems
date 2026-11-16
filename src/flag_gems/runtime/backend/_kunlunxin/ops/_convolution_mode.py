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

"""Kunlunxin ``_convolution_mode`` (string-padding convolution) forward.

``torch._convolution_mode`` is the ATen entry point behind
``torch.nn.functional.conv1d/conv2d/conv3d`` whenever ``padding`` is the string
``"valid"`` or ``"same"``.  The generic FlagGems implementation forwards to the
generic ``flag_gems.ops.conv{1,2,3}d`` Triton kernels and materialises the
``"same"`` padding with ``F.pad``.

On XPU3 those generic kernels do not compute the convolution: their masked
``tl.load`` tile degenerates to a no-op (measured: a constant 0.035 ms for every
shape, output all-zero or NaN), so ``_convolution_mode`` fails on every
non-trivial case.  This override routes the 1D/2D cases to the backend's own
(correct) conv kernels and provides a real Triton kernel for the 3D case,
folding the string mode into explicit per-axis pads.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .conv1d import conv1d
from .conv2d import conv2d

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["total", "ci", "ctot", "din", "hin", "win", "cip"])
def _input_pad_kernel(
    src,
    dst,
    total,
    ci,
    ctot,
    din,
    hin,
    win,
    cip,
    dsn,
    dsc,
    dsd,
    dsh,
    dsw,
    pdl,
    phl,
    pwl,
    BLOCK: tl.constexpr,
):
    """Copy contiguous ``src`` (N, ctot, din, hin, win) into a zeroed padded buffer.

    Channels are regrouped from ``ci`` per group to ``cip`` per group and the
    spatial origin is shifted by ``(pdl, phl, pwl)``.  The innermost axis stays
    contiguous on both sides, so every row access is unit-stride.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < total
    w_i = idx % win
    t = idx // win
    h_i = t % hin
    t = t // hin
    d_i = t % din
    t = t // din
    c_g = t % ctot
    ni = t // ctot
    grp = c_g // ci
    c_l = c_g - grp * ci
    dst_off = (
        ni * dsn
        + (grp * cip + c_l) * dsc
        + (d_i + pdl) * dsd
        + (h_i + phl) * dsh
        + (w_i + pwl) * dsw
    )
    tl.store(dst + dst_off, tl.load(src + idx, mask=m, other=0.0), mask=m)


@libentry()
@triton.jit(
    do_not_specialize=[
        "n",
        "din",
        "hin",
        "win",
        "dout",
        "hout",
        "wout",
        "cout",
        "cpg",
        "opg",
        "xsn",
        "xsc",
        "xsd",
        "xsh",
        "xsw",
        "wso",
        "wsc",
        "wsd",
        "wsh",
        "wsw",
        "ysn",
        "ysc",
        "ysd",
        "ysh",
        "ysw",
    ]
)
def _conv3d_kernel(
    x,
    w,
    b,
    y,
    n,
    din,
    hin,
    win,
    dout,
    hout,
    wout,
    cout,
    cpg,
    opg,
    xsn,
    xsc,
    xsd,
    xsh,
    xsw,
    wso,
    wsc,
    wsd,
    wsh,
    wsw,
    ysn,
    ysc,
    ysd,
    ysh,
    ysw,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scalar 3D convolution (index clamp + ``tl.where``, never a masked load).

    XPU3 does not support the ``rect_select`` lowering used for masked loads
    ("rect_select is not supported on arch3"), so the taps are gathered with
    clamped indices and the out-of-range lanes are zeroed explicitly.
    """
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = n * cout * dout * hout * wout
    valid = m < total
    ow = m % wout
    q = m // wout
    oh = q % hout
    q = q // hout
    od = q % dout
    q = q // dout
    oc = q % cout
    ni = q // cout
    grp = oc // opg
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kd in range(0, KD):
        id_ = od * SD - PD + kd * DD
        d_valid = (id_ >= 0) & (id_ < din)
        id_safe = tl.minimum(tl.maximum(id_, 0), din - 1)
        for kh in range(0, KH):
            ih = oh * SH - PH + kh * DH
            h_valid = d_valid & (ih >= 0) & (ih < hin)
            ih_safe = tl.minimum(tl.maximum(ih, 0), hin - 1)
            for kw in range(0, KW):
                iw = ow * SW - PW + kw * DW
                pos_valid = (iw >= 0) & (iw < win)
                iw_safe = tl.minimum(tl.maximum(iw, 0), win - 1)
                x_off = (
                    ni * xsn
                    + (grp * cpg) * xsc
                    + id_safe * xsd
                    + ih_safe * xsh
                    + iw_safe * xsw
                )
                for ci in range(0, cpg):
                    xv = tl.load(x + x_off + ci * xsc, mask=valid, other=0.0)
                    wv = tl.load(
                        w
                        + oc * wso
                        + ci * wsc
                        + kd * wsd
                        + kh * wsh
                        + kw * wsw,
                        mask=valid,
                        other=0.0,
                    )
                    xv = tl.where(h_valid & pos_valid, xv, 0.0)
                    acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b + oc, mask=valid, other=0.0).to(tl.float32)
    tl.store(y + m * ysw, acc, mask=valid)


def _as_tuple(value, nd):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),) * nd


def _out_size(in_size, kernel, stride, pad_low, pad_high, dilation):
    return (in_size + pad_low + pad_high - dilation * (kernel - 1) - 1) // stride + 1


def _pad_input(x, pad_low, pad_high):
    """Zero-pad a contiguous 5D input; returns the padded tensor (5D)."""
    pdl, phl, pwl = pad_low
    pdh, phh, pwh = pad_high
    n, c, din, hin, win = x.shape
    if not (pdl or phl or pwl or pdh or phh or pwh):
        return x
    # The copy kernel walks the source as a flat buffer; only this rare
    # (asymmetric "same") branch needs a dense source.  This is a pure data
    # movement, no numerics are computed here.
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.zeros(
        (n, c, din + pdl + pdh, hin + phl + phh, win + pwl + pwh),
        device=x.device,
        dtype=x.dtype,
    )
    _input_pad_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        out,
        x.numel(),
        c,
        c,
        din,
        hin,
        win,
        c,
        *out.stride(),
        pdl,
        phl,
        pwl,
        BLOCK=1024,
    )
    return out


def _conv3d(x, w, bias, stride, pad_low, pad_high, dilation, groups):
    (sd, sh, sw) = stride
    (pdl, phl, pwl) = pad_low
    (pdh, phh, pwh) = pad_high
    (dd, dh, dw) = dilation
    kd, kh, kw = w.shape[2], w.shape[3], w.shape[4]
    n, c, din, hin, win = x.shape
    cout, cpg = w.shape[0], w.shape[1]
    dout = _out_size(din, kd, sd, pdl, pdh, dd)
    hout = _out_size(hin, kh, sh, phl, phh, dh)
    wout = _out_size(win, kw, sw, pwl, pwh, dw)
    if dout <= 0 or hout <= 0 or wout <= 0:
        raise RuntimeError(
            "Calculated output size is too small: "
            f"({dout}, {hout}, {wout}) for input ({din}, {hin}, {win})"
        )

    out = torch.empty((n, cout, dout, hout, wout), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out
    grid = (triton.cdiv(out.numel(), 64),)
    _conv3d_kernel[grid](
        x,
        w,
        bias if bias is not None else x,
        out,
        n,
        din,
        hin,
        win,
        dout,
        hout,
        wout,
        cout,
        cpg,
        cout // groups,
        *x.stride(),
        *w.stride(),
        *out.stride(),
        KD=kd,
        KH=kh,
        KW=kw,
        SD=sd,
        SH=sh,
        SW=sw,
        PD=pdl,
        PH=phl,
        PW=pwl,
        DD=dd,
        DH=dh,
        DW=dw,
        HAS_BIAS=bias is not None,
        BLOCK=64,
        num_warps=4,
    )
    return out


def _convolution_mode(input, weight, bias, stride, padding, dilation, groups):
    """Computes a convolution using a string padding mode (``valid``/``same``)."""
    logger.debug("GEMS_KUNLUNXIN _CONVOLUTION_MODE")
    if weight.ndim < 3:
        raise ValueError(f"weight must have at least 3 dimensions, got {weight.ndim}")
    nd = weight.ndim - 2

    stride = _as_tuple(stride, nd)
    dilation = _as_tuple(dilation, nd)
    groups = int(groups)

    if padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")
    elif padding != "valid":
        raise ValueError(
            f"Invalid padding string: {padding!r}, should be one of {{'same', 'valid'}}"
        )

    kernel = tuple(int(k) for k in weight.shape[-nd:])
    if padding == "same":
        # PyTorch splits the total pad as left = total // 2, right = total - left.
        low = tuple((d * (k - 1)) // 2 for k, d in zip(kernel, dilation))
        high = tuple(d * (k - 1) - p for k, d, p in zip(kernel, dilation, low))
    else:
        low = (0,) * nd
        high = (0,) * nd

    if input.shape[1] != groups * weight.shape[1]:
        raise ValueError(
            f"Expected input batch {input.shape} and weight {weight.shape} to "
            f"have the same number of channels per group with groups={groups}"
        )

    # Map to 5D (N, C, D, H, W) with the missing spatial axes as size 1.
    x = input
    w = weight
    for _ in range(3 - nd):
        x = x.unsqueeze(2)
        w = w.unsqueeze(2)
    pad3_low = (0,) * (3 - nd) + low
    pad3_high = (0,) * (3 - nd) + high
    stride3 = (1,) * (3 - nd) + stride
    dilation3 = (1,) * (3 - nd) + dilation

    if nd == 3:
        return _conv3d(x, w, bias, stride3, pad3_low, pad3_high, dilation3, groups)

    # 1D/2D: the backend conv kernels take per-axis integer padding, so the only
    # case needing a materialised pad is an asymmetric "same" border.
    if pad3_low != pad3_high:
        x = _pad_input(x, pad3_low, pad3_high)
        pad3_low = (0, 0, 0)
    if nd == 1:
        return conv1d(
            x.squeeze(2).squeeze(2),
            w.squeeze(2).squeeze(2),
            bias,
            stride3[2],
            pad3_low[2],
            dilation3[2],
            groups,
        )
    return conv2d(
        x.squeeze(2),
        w.squeeze(2),
        bias,
        (stride3[1], stride3[2]),
        (pad3_low[1], pad3_low[2]),
        (dilation3[1], dilation3[2]),
        groups,
    )
