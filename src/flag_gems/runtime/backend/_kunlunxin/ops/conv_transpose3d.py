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
#
# kunlunxin (XPU) re-implementation of conv_transpose3d.
#
# The generic Triton kernel (flag_gems/ops/conv_transpose3d.py) miscomputes on
# this backend (max abs error 7.0e+02 against the CPU reference over the
# official test matrix): its implicit-GEMM `tl.dot` operands are built from
# masked 2-D loads, the documented "masked dot operand" codegen wall on XPU3.
#
# A transposed convolution is an output-centric gather: output position `o`
# receives the taps whose input index `i = (o + pad - k * dilation) / stride`
# is an integer in range.  The backend `convolution` override already carries a
# proven scalar gather kernel for exactly this relation
# (`_conv3d_t_fwd_kernel`, no `tl.dot`, no masked loads, clamped indices,
# groups/bias aware), so this override reuses it -- the same approach as the
# `slow_conv_transpose3d` override.  Every path below runs a real Triton
# kernel; nothing is forwarded to aten/native/composite.
import logging

import torch
import triton

from .convolution import _conv3d_t_fwd_kernel

logger = logging.getLogger(__name__)

__all__ = ["conv_transpose3d"]


def _triple(value, name, default):
    if value is None:
        value = default
    if torch.is_tensor(value):
        if value.dim() == 0:
            value = [int(value.item())]
        else:
            value = [int(v) for v in value.tolist()]
    elif isinstance(value, (int, float)):
        value = [int(value)]
    else:
        value = [int(v) for v in value]
    if len(value) == 1:
        value = value * 3
    if len(value) != 3:
        raise RuntimeError("%s must be a single integer or a sequence of 3 integers" % name)
    return value


def conv_transpose3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Applies a 3D transposed convolution (aten::conv_transpose3d.input)."""
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE3D")

    if not torch.is_tensor(input) or not torch.is_tensor(weight):
        raise TypeError("conv_transpose3d expects tensor inputs")
    if weight.dim() != 5:
        raise RuntimeError(
            "conv_transpose3d: expected 5D weight, got %dD" % weight.dim()
        )
    if not input.is_floating_point() or not weight.is_floating_point():
        raise RuntimeError(
            "conv_transpose3d not implemented for non-floating point dtypes"
        )
    if input.dtype != weight.dtype:
        raise RuntimeError(
            "conv_transpose3d: expected input and weight to have the same dtype, "
            "but got %s and %s" % (input.dtype, weight.dtype)
        )

    st = _triple(stride, "stride", 1)
    pa = _triple(padding, "padding", 0)
    op = _triple(output_padding, "output_padding", 0)
    dl = _triple(dilation, "dilation", 1)
    if any(s <= 0 for s in st):
        raise RuntimeError("conv_transpose3d: stride must be greater than zero")
    if any(d <= 0 for d in dl):
        raise RuntimeError("conv_transpose3d: dilation must be greater than zero")
    if any(p < 0 for p in pa):
        raise RuntimeError("conv_transpose3d: padding must be non-negative")
    for i in range(3):
        if op[i] >= max(st[i], dl[i]):
            raise RuntimeError(
                "conv_transpose3d: output padding must be smaller than either "
                "stride or dilation"
            )

    unbatched = input.dim() == 4
    if unbatched:
        x = input.unsqueeze(0)
    elif input.dim() == 5:
        x = input
    else:
        raise RuntimeError(
            "conv_transpose3d: expected 4D or 5D input, got %dD" % input.dim()
        )

    n, cin, din, hin, win = x.shape
    cin_w, cout_pg, kd, kh, kw = weight.shape
    groups = int(groups)
    if groups <= 0 or cin % groups != 0:
        raise RuntimeError(
            "conv_transpose3d: input channels (%d) must be divisible by "
            "groups (%d)" % (cin, groups)
        )
    if cin != cin_w:
        raise RuntimeError(
            "conv_transpose3d: input channels (%d) must match weight in "
            "channels (%d)" % (cin, cin_w)
        )
    cout = int(cout_pg) * groups
    if bias is not None:
        if not torch.is_tensor(bias):
            raise TypeError("conv_transpose3d: bias must be a Tensor or None")
        if bias.numel() != cout:
            raise RuntimeError(
                "conv_transpose3d: bias size (%d) must match out channels (%d)"
                % (bias.numel(), cout)
            )

    dout = (din - 1) * st[0] - 2 * pa[0] + dl[0] * (kd - 1) + op[0] + 1
    hout = (hin - 1) * st[1] - 2 * pa[1] + dl[1] * (kh - 1) + op[1] + 1
    wout = (win - 1) * st[2] - 2 * pa[2] + dl[2] * (kw - 1) + op[2] + 1
    if dout <= 0 or hout <= 0 or wout <= 0:
        raise RuntimeError(
            "conv_transpose3d: calculated output size is too small: "
            "(%d, %d, %d)" % (dout, hout, wout)
        )

    out = torch.empty((n, cout, dout, hout, wout), device=x.device, dtype=x.dtype)
    total = n * cout * dout * hout * wout
    if total == 0:
        return out.squeeze(0) if unbatched else out

    has_bias = bias is not None and bias.numel() > 0
    b_ptr = bias if has_bias else x
    BLOCK = 64
    _conv3d_t_fwd_kernel[(triton.cdiv(total, BLOCK),)](
        x,
        weight,
        b_ptr,
        out,
        total,
        int(din),
        int(hin),
        int(win),
        int(dout),
        int(hout),
        int(wout),
        cout,
        int(cin),
        int(cin) // groups,  # CIPG
        int(cout_pg),  # OPG
        st[0],
        st[1],
        st[2],
        pa[0],
        pa[1],
        pa[2],
        dl[0],
        dl[1],
        dl[2],
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(x.stride(3)),
        int(x.stride(4)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(weight.stride(2)),
        int(weight.stride(3)),
        int(weight.stride(4)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(out.stride(3)),
        int(out.stride(4)),
        HAS_BIAS=has_bias,
        CIPG=int(cin) // groups,
        OPG=int(cout_pg),
        KD=int(kd),
        KH=int(kh),
        KW=int(kw),
        BLOCK=BLOCK,
    )
    return out.squeeze(0) if unbatched else out
