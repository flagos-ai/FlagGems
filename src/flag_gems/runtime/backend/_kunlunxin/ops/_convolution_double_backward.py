# Copyright 2026 The FlagOS Contributors.
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

"""Kunlunxin override for ``aten::_convolution_double_backward``.

The generic FlagGems implementation (``src/flag_gems/ops/
_convolution_double_backward.py``, read-only for this backend) is a thin
composition of generic conv/convt helpers, and both of them are unusable on
TritonXPU:

* the non-transposed contributions route through the generic ``conv2d``
  forward kernel, a masked 2-D ``tl.dot`` kernel that fails to compile on
  XPU3 (see ``language-xpu3-dot-operand-masking-unsupported``);
* ``grad_self`` / the transposed ``grad_ggO`` contributions route through the
  generic ``conv_transpose2d``, whose residue kernel passes
  ``input_precision="tf32x3"`` — TritonXPU only accepts ``('ieee', 'tf32')``
  (see ``pattern-xpu3-generic-conv-transpose2d-tf32x3-wall``);
* the ``grad_weight`` correlation kernel in the same file is another masked
  2-D ``tl.dot`` kernel.

Baseline on XPU3: 210 failed / 42 passed / 84 skipped (the 42 passes are the
``ggb``-only cells, which never reach a conv).

This module keeps the exact same mathematical decomposition and routes every
conv-shaped piece through the already-shipped, XPU-validated Kunlunxin kernels:

* forward convolution              -> ``_kunlunxin/ops/conv2d.py::_forward_spatial_tile``
  (rank-1 tile kernel, fp32 accumulate, single accumulator per program, no
  ``tl.dot``; the multi-channel ``_forward_spatial_channels*`` variants exceed
  the XPU3 unroll budget at the benchmark's reduction sizes);
* input gradient (transposed conv) -> ``_kunlunxin/ops/conv2d.py::_input_grad``
  (rank-1 gather with clamped loads; this *is* the transposed convolution);
* weight gradient (im2col corr.)   -> ``_kunlunxin/ops/conv2d.py::_weight_grad``
  (rank-1 scalar-loop reduction);
* transposed forward convolution   -> ``_kunlunxin/ops/conv_transpose2d.py``
  (phase-decomposed zero-padded staging gather, no vendor binding required);
* the ``ggb`` broadcast + the sums of the ``grad_ggO`` contributions reuse the
  Gems Triton ``add`` pointwise kernel.

No torch/aten compute primitive is invoked here; the only torch calls are
allocations (``torch.empty``/``empty_like``) plus metadata views
(``view``/``expand``/``unsqueeze``/``squeeze``).  ``output_mask`` is ignored,
matching ``aten::_convolution_double_backward`` (which only forwards it to the
underlying cudnn call and always returns all three gradients); a missing
``ggI``/``ggW``/``ggb`` simply drops the corresponding contribution and makes
the dependent output ``None``.
"""

import logging

import torch
import triton

from flag_gems.ops.add import add
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _forward as _conv2d_forward_flat,
)
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _forward_spatial_tile as _conv2d_forward_tile,
)
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _input_grad as _conv2d_input_grad,
)
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _weight_grad as _conv2d_weight_grad,
)
from flag_gems.runtime.backend._kunlunxin.ops.conv_transpose2d import (
    conv_transpose2d,
)

logger = logging.getLogger(__name__)

# Input-gradient spatial tile / weight-gradient reduction tile.  The backend
# conv2d autograd path uses ``BM=32`` / ``BP=64``; on the double-backward
# benchmark geometry those widths are launch/loop bound, and measured on the
# largest 2D cell ``BM=64`` / ``BP=1024`` are 2.0x / 4.1x faster with identical
# values (``BM=128`` already exceeds the XPU3 unroll budget).
_BM = 64
_BP = 1024


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv2d_forward(x, w, sh, sw, ph, pw, dh, dw, groups):
    """Forward NCHW convolution through the Kunlunxin rank-1 tile kernel.

    The multi-channel ``_forward_spatial_channels{4,8,16}`` kernels the
    autograd ``Conv2d`` path prefers exceed the XPU3 unroll budget once the
    reduction (``CPG*KH*KW``) is large (``TritonXPUUnrollControl`` ->
    ``OutOfResources``), which is exactly the regime the double-backward
    benchmark runs in.  ``_forward_spatial_tile`` keeps a single accumulator
    per program and compiles everywhere; ``BLOCK=128`` only fits when the
    trailing spatial axis is degenerate (``kw == 1``, the 1D-degenerate case),
    so use ``BLOCK=64`` otherwise.
    """
    n, cin, hin, win = x.shape
    cout, cpg, kh, kw = w.shape
    hout = (hin + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wout = (win + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = torch.empty((n, cout, hout, wout), device=x.device, dtype=x.dtype)
    opg = cout // groups
    reduction = cpg * kh * kw
    output_elements = n * cout * hout * wout
    # Mirror the autograd conv dispatch: the per-output-channel tile kernel only
    # for the compute-heavy regime, the flat rank-1 kernel otherwise (tiny
    # channel counts / outputs, where the tile kernel hits the unroll budget).
    if reduction >= 64 and output_elements >= 65536:
        block = 128 if kw == 1 else 64
        _conv2d_forward_tile[(triton.cdiv(n * hout * wout, block), cout)](
            x,
            w,
            None,
            out,
            n,
            hin,
            win,
            cout,
            hout,
            wout,
            cpg,
            opg,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            *x.stride(),
            *w.stride(),
            *out.stride(),
            HAS_BIAS=False,
            CPG=cpg,
            OPG=opg,
            KH=kh,
            KW=kw,
            BLOCK=block,
            num_warps=4,
        )
    else:
        block = 64
        _conv2d_forward_flat[(triton.cdiv(output_elements, block),)](
            x,
            w,
            None,
            out,
            n,
            hin,
            win,
            cout,
            hout,
            wout,
            cpg,
            opg,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            *x.stride(),
            *w.stride(),
            *out.stride(),
            HAS_BIAS=False,
            CPG=cpg,
            OPG=opg,
            KH=kh,
            KW=kw,
            BLOCK=block,
            num_warps=4,
        )
    return out


def _grad_input(dy, w, ref_input, sh, sw, ph, pw, dh, dw, groups):
    """d(ref_input)/d(...) of the forward conv== transposed conv of ``dy``.

    ``ref_input`` only fixes the output shape (channels + spatial extent); its
    data is never read.
    """
    n, cin, hin, win = ref_input.shape
    cout, cpg, kh, kw = w.shape
    hout, wout = dy.shape[2], dy.shape[3]
    dx = torch.empty_like(ref_input)
    _conv2d_input_grad[(triton.cdiv(n * hin * win, _BM), cin)](
        dy,
        w,
        dx,
        n,
        cin,
        hin,
        win,
        hout,
        wout,
        cpg,
        cout // groups,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        *dy.stride(),
        *w.stride(),
        *dx.stride(),
        BM=_BM,
        KH=kh,
        KW=kw,
        OPG=cout // groups,
        CPG=cpg,
    )
    return dx


def _grad_weight(x, dy, weight_shape, sh, sw, ph, pw, dh, dw, groups):
    """Weight-shaped im2col correlation of ``x`` with ``dy``.

    ``grad_w[o, i, r, s] = sum_{n, oh, ow} x[n, gm*cpg+i, oh*sh-ph+r*dh, ...]
    * dy[n, o, oh, ow]`` — exactly the forward conv2d weight gradient.  For a
    transposed forward conv the roles of the two operands swap (the reduction
    then walks the *input* grid and reads ``x`` at the output position
    ``oh*sh-ph+r*dh``), which counts the same tap set.
    """
    cout, cpg, kh, kw = weight_shape
    n = x.shape[0]
    hin, win = x.shape[2], x.shape[3]
    hout, wout = dy.shape[2], dy.shape[3]
    grad_w = torch.empty(weight_shape, device=x.device, dtype=x.dtype)
    _conv2d_weight_grad[(cout * cpg * kh * kw,)](
        x,
        dy,
        grad_w,
        n,
        hin,
        win,
        cout,
        hout,
        wout,
        cpg,
        cout // groups,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        *x.stride(),
        *dy.stride(),
        *grad_w.stride(),
        BP=_BP,
        BK=_BP,
    )
    return grad_w


def _broadcast_bias(ggb, n, gO_h, gO_w, base):
    """``base + ggb[c]`` (or just the broadcast ``ggb`` when ``base`` is None)."""
    bias_part = ggb.view(1, ggb.shape[0], 1, 1).expand(n, ggb.shape[0], gO_h, gO_w)
    if base is None:
        base = torch.zeros(
            (n, ggb.shape[0], gO_h, gO_w), device=ggb.device, dtype=ggb.dtype
        )
    return add(base, bias_part)


def _convolution_double_backward_2d(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    """Second-order backward of a 2D convolution (Kunlunxin kernels)."""
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oph, opw = _pair(output_padding)
    groups = int(groups)

    gO_h, gO_w = gO.shape[2], gO.shape[3]

    grad_ggO = None
    grad_self = None
    grad_weight = None

    # grad_ggO = d<gX,ggI>/dgO + d<gW,ggW>/dgO + d<gB,ggb>/dgO
    parts = []
    if ggI is not None:
        if not transposed:
            parts.append(_conv2d_forward(ggI, weight, sh, sw, ph, pw, dh, dw, groups))
        else:
            parts.append(
                conv_transpose2d(
                    ggI,
                    weight,
                    None,
                    (sh, sw),
                    (ph, pw),
                    (oph, opw),
                    groups,
                    (dh, dw),
                )
            )
    if ggW is not None:
        if not transposed:
            parts.append(_conv2d_forward(self, ggW, sh, sw, ph, pw, dh, dw, groups))
        else:
            parts.append(
                conv_transpose2d(
                    self,
                    ggW,
                    None,
                    (sh, sw),
                    (ph, pw),
                    (oph, opw),
                    groups,
                    (dh, dw),
                )
            )
    if parts:
        grad_ggO = parts[0]
        for p in parts[1:]:
            grad_ggO = add(grad_ggO, p)
    if ggb is not None:
        grad_ggO = _broadcast_bias(ggb, gO.shape[0], gO_h, gO_w, grad_ggO)

    # grad_self only depends on ggW.
    if ggW is not None:
        if not transposed:
            grad_self = _grad_input(
                gO, ggW, self, sh, sw, ph, pw, dh, dw, groups
            )
        else:
            grad_self = _conv2d_forward(gO, ggW, sh, sw, ph, pw, dh, dw, groups)

    # grad_weight only depends on ggI.
    if ggI is not None:
        if not transposed:
            grad_weight = _grad_weight(
                ggI, gO, weight.shape, sh, sw, ph, pw, dh, dw, groups
            )
        else:
            grad_weight = _grad_weight(
                gO, ggI, weight.shape, sh, sw, ph, pw, dh, dw, groups
            )

    return grad_ggO, grad_self, grad_weight


def _convolution_double_backward_1d(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    """Second-order backward of a 1D convolution.

    A 1D conv is a 2D conv with a degenerate trailing spatial axis; route it
    through the 2D path exactly like ``conv1d`` delegates to ``conv2d``.
    """

    def _widen(param, fill):
        if param is None:
            return None
        if isinstance(param, (list, tuple)):
            return [param[0], fill]
        return [param, fill]

    def _unsqueeze(t):
        return t.unsqueeze(-1) if t is not None else None

    def _squeeze(t):
        return t.squeeze(-1) if t is not None else None

    res_2d = _convolution_double_backward_2d(
        _unsqueeze(ggI),
        _unsqueeze(ggW),
        ggb,
        _unsqueeze(gO),
        _unsqueeze(weight),
        _unsqueeze(self),
        _widen(stride, 1),
        _widen(padding, 0),
        _widen(dilation, 1),
        transposed,
        _widen(output_padding, 0),
        groups,
        output_mask,
    )
    return tuple(_squeeze(r) for r in res_2d)


def _convolution_double_backward(
    ggI,
    ggW,
    ggb,
    gO,
    weight,
    self,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS KUNLUNXIN _CONVOLUTION_DOUBLE_BACKWARD")

    spatial_ndim = weight.ndim - 2
    if spatial_ndim == 2:
        return _convolution_double_backward_2d(
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            self,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
    if spatial_ndim == 1:
        return _convolution_double_backward_1d(
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            self,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )

    raise NotImplementedError(
        "_convolution_double_backward: FlagGems supports 1D (3-D operands) and "
        "2D (4-D operands) convolutions, but 3D (5-D operands) is not yet "
        "implemented. got ndim "
        f"self={self.ndim}, weight={weight.ndim}, gO={gO.ndim}."
    )
