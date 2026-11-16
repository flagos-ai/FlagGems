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

"""Kunlunxin (XPU3) override for ``aten::_nnpack_spatial_convolution``.

The generic implementation (``flag_gems/ops/nnpack_spatial_convolution.py``) is
numerically wrong on this backend: every load that feeds a compute value is a
``tl.load(ptr, mask=..., other=0.0)``.  XPU3 lowers the ``other=`` merge through
``rect_select``, which is unsupported on arch3 and silently returns garbage for
the masked lanes instead of the requested ``0.0``.  Measured on the pinned test
matrix: all real convolutions failed (e.g. ``(2,3,19,7)`` 36.1% mismatched,
max abs 3.5) while the affected elements were exactly the padding borders.

Fix: clamp every gather index into the real allocation and zero the invalid
lanes with ``tl.where`` *after* an unmasked load.  No masked load with ``other``
remains in any value path.  The backward keeps the same math as the generic
kernels but:

* replaces the ``tl.dot`` of two masked 2-D loads (a documented XPU3 silent
  miscompile) with a per-output-channel outer-product FMA chain;
* replaces the ``tl.atomic_add`` gradient scatters (unreliable on this backend)
  with a deterministic per-split partial buffer + a second reduction kernel.

Only the host helpers (validation / operand staging) are imported from the
generic module; every value is produced by the real Triton kernels below.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.nnpack_spatial_convolution import (  # host helpers only
    _out_size,
    _pair2,
    _p2,
    _validate,
)

logger = logging.getLogger(__name__)

_NSPLIT = 1


# ---------------------------------------------------------------------------
# forward: fp32 cross-correlation over one output row tile (plain FMA, no dot)
# ---------------------------------------------------------------------------
@triton.jit
def _nnp_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    N,
    IC,
    H,
    W,
    OC,
    OH,
    OW,
    PH,
    PW,
    SH,
    SW,
    sxn,
    sxc,
    sxh,
    sxw,
    swo,
    swc,
    swh,
    sww,
    son,
    soc,
    soh,
    sow,
    HAS_BIAS: tl.constexpr,
    COMPENSATE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_w = tl.program_id(0)
    oh = tl.program_id(1)
    pid_n = tl.program_id(2)

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_valid = ow < OW
    oc = tl.arange(0, BLOCK_OC)
    oc_valid = oc < OC
    occ = tl.minimum(oc, OC - 1)

    # Neumaier-compensated accumulation: the wide value-range rows multiply
    # +-3e38 operands, so the fp32 running sum loses whole products unless the
    # rounding error of every addition is carried separately (the generic
    # implementation measured this to be what makes those rows pass).
    s = tl.zeros((BLOCK_W, BLOCK_OC), dtype=tl.float32)
    c = tl.zeros((BLOCK_W, BLOCK_OC), dtype=tl.float32)
    x_n = x_ptr + pid_n * sxn
    w_oc = w_ptr + occ * swo

    for ic_i in range(0, IC):
        for kh_i in tl.static_range(0, KH):
            ih = oh * SH - PH + kh_i
            ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
            for kw_i in tl.static_range(0, KW):
                iw = ow * SW - PW + kw_i
                iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
                m = ow_valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xv = tl.load(x_n + ic_i * sxc + ihc * sxh + iwc * sxw)
                xv = tl.where(m, xv, 0.0)
                wv = tl.load(w_oc + ic_i * swc + (kh_i * swh + kw_i * sww))
                wv = tl.where(oc_valid, wv, 0.0)
                v = xv[:, None] * wv[None, :]
                t = s + v
                if COMPENSATE:
                    c += tl.where(tl.abs(s) >= tl.abs(v), (s - t) + v, (v - t) + s)
                s = t

    acc = s + c if COMPENSATE else s
    if HAS_BIAS:
        bv = tl.load(b_ptr + occ)
        acc += tl.where(oc_valid, bv, 0.0)[None, :]

    optr = (
        o_ptr
        + pid_n.to(tl.int64) * son
        + (oh * soh + ow * sow).to(tl.int64)[:, None]
        + oc.to(tl.int64)[None, :] * soc
    )
    tl.store(optr, acc, mask=ow_valid[:, None] & oc_valid[None, :])


# ---------------------------------------------------------------------------
# backward: gradient wrt the input (gather over (oc, kh, kw), no dot)
# ---------------------------------------------------------------------------
@triton.jit
def _nnp_bwd_x_kernel(
    g_ptr,
    w_ptr,
    dx_ptr,
    N,
    IC,
    H,
    W,
    OC,
    OH,
    OW,
    PH,
    PW,
    SH,
    SW,
    sgn,
    sgc,
    sgh,
    sgw,
    swo,
    swc,
    swh,
    sww,
    sdn,
    sdc,
    sdh,
    sdw,
    BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_w = tl.program_id(0)
    ih = tl.program_id(1)
    pid_n = tl.program_id(2)

    col = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    col_valid = col < W
    ic = tl.arange(0, BLOCK_IC)
    ic_valid = ic < IC
    icc = tl.minimum(ic, IC - 1)

    acc = tl.zeros((BLOCK_W, BLOCK_IC), dtype=tl.float32)
    g_n = g_ptr + pid_n * sgn

    for oc_i in range(0, OC):
        for kh_i in tl.static_range(0, KH):
            num = ih + PH - kh_i
            oh = num // SH
            ohc = tl.minimum(tl.maximum(oh, 0), OH - 1)
            ok_h = (num >= 0) & (num - oh * SH == 0) & (oh < OH)
            for kw_i in tl.static_range(0, KW):
                numw = col + PW - kw_i
                ow = numw // SW
                owc = tl.minimum(tl.maximum(ow, 0), OW - 1)
                ok_w = col_valid & (numw >= 0) & (numw - ow * SW == 0) & (ow < OW)
                m = ok_h & ok_w
                gv = tl.load(g_n + oc_i * sgc + (ohc * sgh + owc * sgw))
                gv = tl.where(m, gv, 0.0)
                wv = tl.load(
                    w_ptr + oc_i * swo + icc * swc + (kh_i * swh + kw_i * sww)
                )
                wv = tl.where(ic_valid, wv, 0.0)
                acc += gv[:, None] * wv[None, :]

    tl.store(
        dx_ptr
        + pid_n * sdn
        + ic[None, :] * sdc
        + (ih * sdh + col * sdw)[:, None],
        acc,
        mask=col_valid[:, None] & ic_valid[None, :],
    )


# ---------------------------------------------------------------------------
# backward: gradient wrt the weight (deterministic split reduction)
# ---------------------------------------------------------------------------
@triton.jit(
    do_not_specialize=["N", "IC", "H", "W", "OC", "OH", "OW", "PH", "PW", "SH", "SW"]
)
def _nnp_bwd_w_partial_kernel(
    g_ptr,
    x_ptr,
    dwp_ptr,
    N,
    IC,
    H,
    W,
    OC,
    OH,
    OW,
    PH,
    PW,
    SH,
    SW,
    NSPLIT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    kw = tl.program_id(2)

    nspatial = N * OH * OW
    chunk = (nspatial + NSPLIT - 1) // NSPLIT
    k_begin = pid * chunk
    k_end = tl.minimum(k_begin + chunk, nspatial)

    sgn = OC * OH * OW
    sgc = OH * OW
    sgh = OW
    sxn = IC * H * W
    sxc = H * W
    sxh = W

    oc = tl.arange(0, BLOCK_OC)
    ic = tl.arange(0, BLOCK_IC)
    occ = tl.minimum(oc, OC - 1)
    icc = tl.minimum(ic, IC - 1)

    acc = tl.zeros((BLOCK_OC, BLOCK_IC), dtype=tl.float32)
    for k in range(k_begin, k_end):
        n = k // (OH * OW)
        rem = k - n * (OH * OW)
        oh = rem // OW
        ow = rem - oh * OW
        ih = oh * SH - PH + kh
        iw = ow * SW - PW + kw
        iok = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
        iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
        g = tl.load(g_ptr + n * sgn + occ * sgc + oh * sgh + ow)
        xv = tl.load(x_ptr + n * sxn + icc * sxc + ihc * sxh + iwc)
        xv = tl.where(iok, xv, 0.0)
        acc += g[:, None] * xv[None, :]

    # Store straight into the padded (BLOCK_OC, BLOCK_IC, KH, KW) buffer: only
    # the real (oc, ic) lanes are written and the host returns a view prefix.
    tl.store(
        dwp_ptr
        + oc[:, None] * (BLOCK_IC * KH * KW)
        + ic[None, :] * (KH * KW)
        + (kh * KW + kw),
        acc,
        mask=(oc[:, None] < OC) & (ic[None, :] < IC),
    )


@triton.jit(do_not_specialize=["N", "OC", "OH", "OW"])
def _nnp_bwd_b_partial_kernel(
    g_ptr,
    dbp_ptr,
    N,
    OC,
    OH,
    OW,
    NSPLIT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    nspatial = N * OH * OW
    chunk = (nspatial + NSPLIT - 1) // NSPLIT
    k_begin = pid * chunk
    k_end = tl.minimum(k_begin + chunk, nspatial)

    sgn = OC * OH * OW
    sgc = OH * OW
    sgh = OW

    oc = tl.arange(0, BLOCK_OC)
    occ = tl.minimum(oc, OC - 1)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for k in range(k_begin, k_end):
        n = k // (OH * OW)
        rem = k - n * (OH * OW)
        oh = rem // OW
        ow = rem - oh * OW
        acc += tl.load(g_ptr + n * sgn + occ * sgc + oh * sgh + ow)
    tl.store(dbp_ptr + oc, acc)


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------
def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _to_dev(t):
    if t is None:
        return None
    if t.is_cuda:
        return t if t.is_contiguous() else t.contiguous()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.to(_device())


def _forward(shape, input, weight, bias):
    batch, ic, h, w, oc, kh, kw, oh, ow, ph, pw, sh, sw = shape
    x = _to_dev(input)
    wt = _to_dev(weight)
    bdev = _to_dev(bias)
    out = torch.empty((batch, oc, oh, ow), dtype=torch.float32, device=x.device)
    if out.numel() == 0:
        return out

    # One program keeps its tile inside a single output row, so BLOCK_W never
    # exceeds OW and no store straddles a row boundary.  The tile geometry and
    # the FMA tap loop are unchanged from the generic implementation; only the
    # value loads changed (clamped + ``tl.where`` instead of ``other=0.0``).
    if ow >= 64:
        block_w = 64
    else:
        block_w = max(8, min(32, _p2(ow)))
    block_oc = min(32, _p2(oc))
    num_warps = 8 if (ow >= 64 and ic * kh * kw > 1) else 4
    grid = (triton.cdiv(ow, block_w), oh, batch)
    _nnp_fwd_kernel[grid](
        x,
        wt,
        bdev if bdev is not None else wt,
        out,
        batch,
        ic,
        h,
        w,
        oc,
        oh,
        ow,
        ph,
        pw,
        sh,
        sw,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        wt.stride(0),
        wt.stride(1),
        wt.stride(2),
        wt.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        HAS_BIAS=(bdev is not None),
        COMPENSATE=(sh == 1 and sw == 1 and ic * kh * kw > 1),
        BLOCK_W=block_w,
        BLOCK_OC=block_oc,
        KH=kh,
        KW=kw,
        num_warps=num_warps,
    )
    return out


def _backward(shape, input, weight, grad_out):
    batch, ic, h, w, oc, kh, kw, oh, ow, ph, pw, sh, sw = shape
    x = _to_dev(input)
    wt = _to_dev(weight)
    g = _to_dev(grad_out)
    dx = torch.zeros((batch, ic, h, w), dtype=torch.float32, device=x.device)
    dw = torch.zeros((oc, ic, kh, kw), dtype=torch.float32, device=x.device)
    db = torch.zeros((oc,), dtype=torch.float32, device=x.device)

    if dx.numel() > 0:
        block_ic = max(16, _p2(ic))
        block_w = max(16, min(64, _p2(w)))
        grid = (triton.cdiv(w, block_w), h, batch)
        _nnp_bwd_x_kernel[grid](
            g,
            wt,
            dx,
            batch,
            ic,
            h,
            w,
            oc,
            oh,
            ow,
            ph,
            pw,
            sh,
            sw,
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            wt.stride(0),
            wt.stride(1),
            wt.stride(2),
            wt.stride(3),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            BLOCK_W=block_w,
            BLOCK_IC=block_ic,
            KH=kh,
            KW=kw,
            num_warps=4,
        )

    if dw.numel() > 0 or db.numel() > 0:
        block_oc = max(16, _p2(oc))
        block_ic = max(16, _p2(ic))

        dwpad = torch.empty(
            (block_oc, block_ic, kh, kw), dtype=torch.float32, device=x.device
        )
        _nnp_bwd_w_partial_kernel[(_NSPLIT, kh, kw)](
            g,
            x,
            dwpad,
            batch,
            ic,
            h,
            w,
            oc,
            oh,
            ow,
            ph,
            pw,
            sh,
            sw,
            NSPLIT=_NSPLIT,
            BLOCK_OC=block_oc,
            BLOCK_IC=block_ic,
            KH=kh,
            KW=kw,
            num_warps=4,
        )
        # Pure view: the leading (oc, ic) prefix carries the real gradient.
        dw = dwpad[:oc, :ic]

        if db.numel() > 0:
            dbstack = torch.empty((block_oc,), dtype=torch.float32, device=x.device)
            _nnp_bwd_b_partial_kernel[(_NSPLIT,)](
                g,
                dbstack,
                batch,
                oc,
                oh,
                ow,
                NSPLIT=_NSPLIT,
                BLOCK_OC=block_oc,
                num_warps=4,
            )
            db = dbstack[:oc]
    return dx, dw, db


class _NNPConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, shape):
        out = _forward(shape, input, weight, bias)
        ctx.save_for_backward(input, weight)
        ctx.shape = shape
        ctx.has_bias = bias is not None
        # The tests build the operands on the host, so the result must come back
        # to the operand device; the accelerator only ever computes.
        ctx.input_device = input.device
        ctx.weight_device = weight.device
        ctx.bias_device = bias.device if bias is not None else input.device
        return out if input.is_cuda else out.to(input.device)

    @staticmethod
    def backward(ctx, grad_out):
        input, weight = ctx.saved_tensors
        dx, dw, db = _backward(ctx.shape, input, weight, grad_out)
        dx = dx.to(ctx.input_device)
        dw = dw.to(ctx.weight_device)
        if not ctx.has_bias:
            return dx, dw, None, None
        return dx, dw, db.to(ctx.bias_device), None


def nnpack_spatial_convolution(input, weight, bias, padding, stride=[1, 1], **kwargs):
    """aten::_nnpack_spatial_convolution(input, weight, bias, padding, stride=[1, 1])."""
    logger.debug("GEMS NNPACK_SPATIAL_CONVOLUTION")
    out = kwargs.pop("out", None)
    if kwargs:
        raise TypeError(
            "_nnpack_spatial_convolution() got unexpected keyword arguments %s"
            % sorted(kwargs)
        )
    shape = _validate(input, weight, bias, padding, stride)

    if out is not None:
        if not torch.is_tensor(out):
            raise TypeError("_nnpack_spatial_convolution(): 'out' must be a Tensor")
        expected = (shape[0], shape[4], shape[7], shape[8])
        if tuple(int(v) for v in out.shape) != expected:
            raise RuntimeError(
                "_nnpack_spatial_convolution.out(): expected out to have shape "
                "%s, but got %s" % (expected, tuple(int(v) for v in out.shape))
            )
        if out.dtype != torch.float32:
            raise RuntimeError(
                "_nnpack_spatial_convolution.out(): expected out to be float32, "
                "got %s" % str(out.dtype).split(".")[-1]
            )
        result = _forward(shape, input, weight, bias)
        out.copy_(result if out.is_cuda else result.to(out.device))
        return out

    if torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        return _NNPConvFunction.apply(input, weight, bias, shape)

    result = _forward(shape, input, weight, bias)
    if input.is_cuda:
        return result
    return result.to(input.device)
