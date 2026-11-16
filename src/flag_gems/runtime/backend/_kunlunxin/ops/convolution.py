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
"""Kunlunxin (XPU3) override for ``aten::convolution``.

The generic ``flag_gems.ops.convolution`` is an implicit-GEMM ``tl.dot`` kernel
that builds its operands with ``tl.load(..., mask=..., other=0.0)`` and applies
``tl.where`` to the taps.  On XPU3 that is the documented "masked dot operand"
codegen wall: the backend either fails the MLIR pipeline
(``rect_select is not supported on arch3``) or silently emits a kernel that
computes garbage, so every real convolution fails against a CPU reference while
still returning in a shape-independent ~0.03 ms (see the knowledge base entries
``pattern-xpu3-generic-conv-kernels-silent-noop`` and
``language-xpu3-dot-tap-loop-miscompiles``).

This override keeps the exact argument-validation contract of the generic
implementation and routes the *valid* calls to kernels that are known correct
on XPU3:

* 1-D / 2-D forward -> the backend ``conv1d`` / ``conv2d`` kernels (scalar
  gather, no ``tl.dot``; verified exact against a CPU reference).
* 2-D transposed -> the backend ``conv_transpose2d`` kernel (real Triton).
* 3-D forward / transposed -> the simple scalar kernels below (the standalone
  ``conv3d`` backend kernel miscomputes on XPU3, so it is not reused).

No call is forwarded to ``aten``/native/composite: every path runs a real
Triton kernel.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _conv3d_fwd_kernel(
    x,
    w,
    b,
    y,
    total,
    din,
    hin,
    win,
    dout,
    hout,
    wout,
    cout,
    cpg,
    opg,
    sd,
    sh,
    sw,
    pd,
    ph,
    pw,
    dd,
    dh,
    dw,
    xsn,
    xsc,
    xsd,
    xsh,
    xsw,
    wso,
    wsi,
    wsd,
    wsh,
    wsw,
    ysn,
    ysc,
    ysd,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ow = m % wout
    q = m // wout
    oh = q % hout
    q = q // hout
    od = q % dout
    q = q // dout
    oc = q % cout
    ni = q // cout
    group = oc // OPG
    mask = m < total
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(KD):
        iz = od * sd - pd + r * dd
        for t in range(KH):
            iy = oh * sh - ph + t * dh
            for s in range(KW):
                ix = ow * sw - pw + s * dw
                valid = (
                    mask
                    & (iz >= 0)
                    & (iz < din)
                    & (iy >= 0)
                    & (iy < hin)
                    & (ix >= 0)
                    & (ix < win)
                )
                sz = tl.where(valid, iz, 0)
                sy = tl.where(valid, iy, 0)
                sx = tl.where(valid, ix, 0)
                for ci in range(CPG):
                    xv = tl.load(
                        x
                        + ni * xsn
                        + (group * CPG + ci) * xsc
                        + sz * xsd
                        + sy * xsh
                        + sx * xsw,
                        mask=mask,
                        other=0.0,
                    )
                    xv = tl.where(valid, xv, 0.0)
                    wv = tl.load(w + oc * wso + ci * wsi + r * wsd + t * wsh + s * wsw)
                    acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b + oc).to(tl.float32)
    tl.store(
        y
        + ni * ysn
        + oc * ysc
        + od * ysd
        + oh * ysh
        + ow * ysw,
        acc.to(y.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _conv3d_t_fwd_kernel(
    x,
    w,
    b,
    y,
    total,
    din,
    hin,
    win,
    dout,
    hout,
    wout,
    cout,
    cin,
    cipg,
    opg,
    sd,
    sh,
    sw,
    pd,
    ph,
    pw,
    dd,
    dh,
    dw,
    xsn,
    xsc,
    xsd,
    xsh,
    xsw,
    wsi,
    wso,
    wsd,
    wsh,
    wsw,
    ysn,
    ysc,
    ysd,
    ysh,
    ysw,
    HAS_BIAS: tl.constexpr,
    CIPG: tl.constexpr,
    OPG: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ow = m % wout
    q = m // wout
    oh = q % hout
    q = q // hout
    od = q % dout
    q = q // dout
    oc = q % cout
    ni = q // cout
    group = oc // OPG
    ocl = oc - group * OPG
    mask = m < total
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(KD):
        nz = od + pd - r * dd
        iz = nz // sd
        okz = (nz % sd == 0) & (iz >= 0) & (iz < din)
        for t in range(KH):
            ny = oh + ph - t * dh
            iy = ny // sh
            oky = (ny % sh == 0) & (iy >= 0) & (iy < hin)
            for s in range(KW):
                nx = ow + pw - s * dw
                ix = nx // sw
                okx = (nx % sw == 0) & (ix >= 0) & (ix < win)
                valid = mask & okz & oky & okx
                sz = tl.where(valid, iz, 0)
                sy = tl.where(valid, iy, 0)
                sx = tl.where(valid, ix, 0)
                for ci in range(CIPG):
                    xv = tl.load(
                        x
                        + ni * xsn
                        + (group * CIPG + ci) * xsc
                        + sz * xsd
                        + sy * xsh
                        + sx * xsw,
                        mask=mask,
                        other=0.0,
                    )
                    xv = tl.where(valid, xv, 0.0)
                    wv = tl.load(
                        w
                        + (group * CIPG + ci) * wsi
                        + ocl * wso
                        + r * wsd
                        + t * wsh
                        + s * wsw
                    )
                    acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b + oc).to(tl.float32)
    tl.store(
        y
        + ni * ysn
        + oc * ysc
        + od * ysd
        + oh * ysh
        + ow * ysw,
        acc.to(y.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _bias_roundtrip_kernel(y, b, total, plane, cout, BLOCK: tl.constexpr):
    """Add ``bias`` to an NCHW output with ATen's two-rounding order.

    ``aten::convolution`` rounds the convolution result to the output dtype
    before adding the bias.  Adding the bias at fp32 accumulator precision and
    rounding once drifts by ~1 ULP in fp16/bf16, so read the already-rounded
    result back, add the bias in fp32 and round again.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total
    ch = (idx // plane) % cout
    v = tl.load(y + idx, mask=mask, other=0.0)
    bv = tl.load(b + ch, mask=mask, other=0.0)
    out = v.to(tl.float32) + bv.to(tl.float32)
    tl.store(y + idx, out.to(y.dtype.element_ty), mask=mask)


def _apply_bias(y, bias, nd):
    """Add ``bias`` to ``y`` (NCHW, contiguous) in place with ATen rounding."""
    if bias is None or bias.numel() == 0:
        return y
    total = y.numel()
    if total == 0:
        return y
    plane = 1
    for d in range(2, y.dim()):
        plane *= int(y.shape[d])
    cout = int(y.shape[1])
    b = bias
    if not b.is_contiguous():
        b = b.contiguous()
    BLOCK = 1024
    _bias_roundtrip_kernel[(triton.cdiv(total, BLOCK),)](
        y, b, total, plane, cout, BLOCK=BLOCK
    )
    return y


def _gv(v, i, d):
    """Element-wise accessor for stride/padding/dilation-style arguments."""
    if v is None:
        return d
    if isinstance(v, (int, float, bool)):
        return int(v)
    v = list(v)
    if len(v) == 1:
        return int(v[0]) if i == 0 else d
    return int(v[i])


def _conv3d_forward(x, w, bias, S, P, DL, IN, K, OUT, groups, C_in_pg, C_out_pg):
    n = int(x.shape[0])
    cout = int(w.shape[0])
    out = torch.empty(
        (n, cout, OUT[0], OUT[1], OUT[2]), device=x.device, dtype=x.dtype
    )
    total = n * cout * OUT[0] * OUT[1] * OUT[2]
    if total == 0:
        return out
    has_bias = bias is not None and bias.numel() > 0
    b_ptr = bias if has_bias else x
    BLOCK = 64
    _conv3d_fwd_kernel[(triton.cdiv(total, BLOCK),)](
        x,
        w,
        b_ptr,
        out,
        total,
        IN[0],
        IN[1],
        IN[2],
        OUT[0],
        OUT[1],
        OUT[2],
        cout,
        C_in_pg,
        C_out_pg,
        S[0],
        S[1],
        S[2],
        P[0],
        P[1],
        P[2],
        DL[0],
        DL[1],
        DL[2],
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(x.stride(3)),
        int(x.stride(4)),
        int(w.stride(0)),
        int(w.stride(1)),
        int(w.stride(2)),
        int(w.stride(3)),
        int(w.stride(4)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(out.stride(3)),
        int(out.stride(4)),
        HAS_BIAS=has_bias,
        CPG=C_in_pg,
        OPG=C_out_pg,
        KD=K[0],
        KH=K[1],
        KW=K[2],
        BLOCK=BLOCK,
    )
    return out


def _conv3d_transposed(x, w, bias, S, P, DL, OP, IN, K, OUT, groups, C_in_pg, C_out_pg):
    n = int(x.shape[0])
    cout = C_out_pg * groups
    out = torch.empty(
        (n, cout, OUT[0], OUT[1], OUT[2]), device=x.device, dtype=x.dtype
    )
    total = n * cout * OUT[0] * OUT[1] * OUT[2]
    if total == 0:
        return out
    has_bias = bias is not None and bias.numel() > 0
    b_ptr = bias if has_bias else x
    BLOCK = 64
    _conv3d_t_fwd_kernel[(triton.cdiv(total, BLOCK),)](
        x,
        w,
        b_ptr,
        out,
        total,
        IN[0],
        IN[1],
        IN[2],
        OUT[0],
        OUT[1],
        OUT[2],
        cout,
        int(x.shape[1]),
        C_in_pg,
        C_out_pg,
        S[0],
        S[1],
        S[2],
        P[0],
        P[1],
        P[2],
        DL[0],
        DL[1],
        DL[2],
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(x.stride(3)),
        int(x.stride(4)),
        int(w.stride(0)),
        int(w.stride(1)),
        int(w.stride(2)),
        int(w.stride(3)),
        int(w.stride(4)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(out.stride(3)),
        int(out.stride(4)),
        HAS_BIAS=has_bias,
        CIPG=C_in_pg,
        OPG=C_out_pg,
        KD=K[0],
        KH=K[1],
        KW=K[2],
        BLOCK=BLOCK,
    )
    return out


def _route_forward(x, w, bias, stride, padding, dilation, groups, nd):
    if nd == 1:
        from .conv1d import conv1d

        return conv1d(x, w, bias, stride, padding, dilation, groups)
    if nd == 2:
        from .conv2d import conv2d

        return conv2d(x, w, bias, stride, padding, dilation, groups)
    # nd == 3 handled by the caller with pre-computed geometry
    return None


def _route_transposed(x, w, bias, stride, padding, dilation, output_padding, groups, nd):
    if nd == 1:
        from .conv_transpose2d import conv_transpose2d

        s = _gv(stride, 0, 1)
        p = _gv(padding, 0, 0)
        d = _gv(dilation, 0, 1)
        op = _gv(output_padding, 0, 0)
        out = conv_transpose2d(
            x.unsqueeze(2),
            w.unsqueeze(2),
            bias,
            (1, s),
            (0, p),
            (0, op),
            groups,
            (1, d),
        )
        return out.squeeze(2)
    if nd == 2:
        from .conv_transpose2d import conv_transpose2d

        return conv_transpose2d(
            x, w, bias, stride, padding, output_padding, groups, dilation
        )
    return None


def convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    x = input
    w = weight
    nd = x.dim() - 2
    is_transposed = bool(transposed)
    g = int(groups)

    # --- validation, byte-for-byte the generic contract -------------------
    if x.dim() != w.dim():
        raise RuntimeError(
            "Expected %d-dimensional input for %d-dimensional weight %s, but "
            "got %d-dimensional input of size %s instead"
            % (w.dim(), w.dim(), list(w.shape), x.dim(), list(x.shape))
        )
    if nd < 1 or nd > 3:
        raise RuntimeError(
            "Expected 3D, 4D or 5D input to convolution, but got input of size %s"
            % (list(x.shape),)
        )
    if g <= 0:
        raise RuntimeError("non-positive groups is not supported")
    if int(w.shape[0]) < 1:
        raise RuntimeError(
            "Given groups=%d, expected weight to be at least 1 at dimension 0, "
            "but got weight of size %s instead" % (g, list(w.shape))
        )

    n_ = int(x.shape[0])
    C_in = int(x.shape[1])

    if is_transposed:
        expected_in = int(w.shape[0])
    else:
        expected_in = int(w.shape[1]) * g
    if C_in != expected_in:
        raise RuntimeError(
            "Given %sgroups=%d, weight of size %s, expected input%s to have %d "
            "channels, but got %d channels instead"
            % (
                "transposed " if is_transposed else "",
                g,
                list(w.shape),
                list(x.shape),
                expected_in,
                C_in,
            )
        )
    out_channels = int(w.shape[1]) * g if is_transposed else int(w.shape[0])
    if out_channels % g != 0:
        raise RuntimeError(
            "Given groups=%d, expected weight to be divisible by %d, but got "
            "output channels of %d" % (g, g, out_channels)
        )

    if bias is not None and bias.numel() > 0:
        if bias.dim() != 1 or int(bias.shape[0]) != out_channels:
            raise RuntimeError(
                "Given weight of size %s, expected bias to be 1-dimensional "
                "with %d elements, but got bias of size %s instead"
                % (list(w.shape), out_channels, list(bias.shape))
            )

    for i in range(nd):
        if _gv(stride, i, 1) <= 0:
            raise RuntimeError("non-positive stride is not supported")
        if _gv(dilation, i, 1) <= 0:
            raise ValueError("Expected positive dilation in convolution.")
        if _gv(padding, i, 0) < 0:
            raise RuntimeError("negative padding is not supported")
        if _gv(output_padding, i, 0) < 0:
            raise RuntimeError("negative output_padding is not supported")

    if is_transposed:
        for i in range(nd):
            op = _gv(output_padding, i, 0)
            if op >= _gv(stride, i, 1) and op >= _gv(dilation, i, 1):
                raise RuntimeError(
                    "output padding must be smaller than either stride or "
                    "dilation, but got output_padding=%d, stride=%d, dilation=%d"
                    % (op, _gv(stride, i, 1), _gv(dilation, i, 1))
                )

    C_in_pg = C_in // g

    if nd == 1:
        S = [_gv(stride, 0, 1), 1, 1]
        P = [_gv(padding, 0, 0), 0, 0]
        DL = [_gv(dilation, 0, 1), 1, 1]
        OP = [_gv(output_padding, 0, 0), 0, 0]
        IN = [int(x.shape[2]), 1, 1]
        K = [int(w.shape[2]), 1, 1]
    elif nd == 2:
        S = [_gv(stride, 0, 1), _gv(stride, 1, 1), 1]
        P = [_gv(padding, 0, 0), _gv(padding, 1, 0), 0]
        DL = [_gv(dilation, 0, 1), _gv(dilation, 1, 1), 1]
        OP = [_gv(output_padding, 0, 0), _gv(output_padding, 1, 0), 0]
        IN = [int(x.shape[2]), int(x.shape[3]), 1]
        K = [int(w.shape[2]), int(w.shape[3]), 1]
    else:
        S = [_gv(stride, 0, 1), _gv(stride, 1, 1), _gv(stride, 2, 1)]
        P = [_gv(padding, 0, 0), _gv(padding, 1, 0), _gv(padding, 2, 0)]
        DL = [_gv(dilation, 0, 1), _gv(dilation, 1, 1), _gv(dilation, 2, 1)]
        OP = [
            _gv(output_padding, 0, 0),
            _gv(output_padding, 1, 0),
            _gv(output_padding, 2, 0),
        ]
        IN = [int(x.shape[2]), int(x.shape[3]), int(x.shape[4])]
        K = [int(w.shape[2]), int(w.shape[3]), int(w.shape[4])]

    OUT = []
    for i in range(3):
        if is_transposed:
            o = (IN[i] - 1) * S[i] - 2 * P[i] + DL[i] * (K[i] - 1) + 1 + OP[i]
        else:
            o = (IN[i] + 2 * P[i] - DL[i] * (K[i] - 1) - 1) // S[i] + 1
        OUT.append(int(o))

    if not is_transposed and min(OUT[:nd]) <= 0:
        padded = [IN[i] + 2 * P[i] for i in range(nd)]
        eff_k = [DL[i] * (K[i] - 1) + 1 for i in range(nd)]
        raise RuntimeError(
            "Calculated padded input size per channel: (%s). Kernel size: (%s). "
            "Kernel size can't be greater than actual input size"
            % (
                " x ".join(str(v) for v in padded),
                " x ".join(str(v) for v in eff_k),
            )
        )

    if is_transposed:
        C_out_pg = int(w.shape[1])
    else:
        C_out_pg = int(w.shape[0]) // g

    # --- dispatch to real Triton kernels ---------------------------------
    # The generic forward kernel builds no autograd graph, so mirror that
    # contract (a ``requires_grad`` input still yields a detached result).
    with torch.no_grad():
        if not is_transposed:
            if nd == 3:
                out = _conv3d_forward(
                    x, w, None, S, P, DL, IN, K, OUT, g, C_in_pg, C_out_pg
                )
            else:
                out = _route_forward(x, w, None, stride, padding, dilation, g, nd)
            return _apply_bias(out, bias, nd)

        if nd == 3:
            out = _conv3d_transposed(
                x, w, None, S, P, DL, OP, IN, K, OUT, g, C_in_pg, C_out_pg
            )
        else:
            out = _route_transposed(
                x, w, None, stride, padding, dilation, output_padding, g, nd
            )
        return _apply_bias(out, bias, nd)


__all__ = ["convolution"]
