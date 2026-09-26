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

"""Kunlunxin XPU override for ``aten::convolution_overrideable``.

The generic FlagGems implementation
(``flag_gems.ops.convolution_overrideable``) drives an im2col ``tl.dot``
matmul kernel.  On the P800 TritonXPU backend that kernel produces NaN /
garbage (masked ``tl.load`` ``other`` values are unreliable and ``tl.dot``
reads past the padded contraction boundary), so 99/111 accuracy cases fail.

Instead of the ``tl.dot`` path, this override reuses the device-resident
Kunlunxin forward conv kernels (scalar fp32 accumulation, no ``tl.dot``),
which are already validated on this backend:

  * Forward 1D/2D/3D convolution -> the vendor ``conv1d``/``conv2d``/``conv3d``
    kernels.
  * Transposed convolution -> expressed as a forward ``conv2d`` on a
    stride-dilated input with the spatially-flipped, channel-transposed
    weight (the standard "transposed conv == fractionally-strided conv"
    identity).  The vendor ``conv_transpose2d`` xpudnn-fusion binding and the
    ``conv2d`` input-gradient kernel both mis-compile / mis-compute on this
    stack, whereas the forward ``conv2d`` kernel is exact; only tensor
    rearrangement (zero-insertion, weight flip/permute) is done outside the
    kernel, never the convolution arithmetic itself.
"""

import logging

import torch

from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d

logger = logging.getLogger(__name__)


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv_transpose2d(
    input, weight, bias, stride, padding, output_padding, groups, dilation
):
    """Transposed 2D convolution via the forward ``conv2d`` kernel.

    Uses the identity ``convT(x, w) == conv(dilate(x), flip(swap(w)))`` so all
    multiply-accumulate work runs through the validated forward conv2d Triton
    kernel; only zero-insertion and weight reshaping happen on the host.
    """
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oph, opw = _pair(output_padding)
    dh, dw = _pair(dilation)

    n, cin, hin, win = input.shape
    _, cout_pg, kh, kw = weight.shape
    cin_pg = cin // groups

    pf_h = dh * (kh - 1) - ph
    pf_w = dw * (kw - 1) - pw

    hd = (hin - 1) * sh + 1 + oph
    wd = (win - 1) * sw + 1 + opw
    x_dil = torch.zeros((n, cin, hd, wd), device=input.device, dtype=input.dtype)
    x_dil[:, :, 0 : (hin - 1) * sh + 1 : sh, 0 : (win - 1) * sw + 1 : sw] = input

    w_flip = (
        weight.reshape(groups, cin_pg, cout_pg, kh, kw)
        .permute(0, 2, 1, 3, 4)
        .flip(-1, -2)
        .reshape(groups * cout_pg, cin_pg, kh, kw)
        .contiguous()
    )

    return conv2d(x_dil, w_flip, bias, (1, 1), (pf_h, pf_w), (dh, dw), groups)


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
):
    """Route to the matching Kunlunxin Triton conv kernel by spatial rank.

    Mirrors ``aten::convolution_overrideable``: the spatial rank is inferred
    from ``weight.ndim - 2``.
    """
    spatial_dims = weight.ndim - 2
    assert spatial_dims in (1, 2, 3), (
        f"convolution_overrideable only supports 1D/2D/3D convolutions, "
        f"received weight with shape {tuple(weight.shape)}"
    )

    if transposed:
        if spatial_dims == 1:
            stride_w = _pair(stride)[0]
            padding_w = _pair(padding)[0]
            output_padding_w = _pair(output_padding)[0]
            dilation_w = _pair(dilation)[0]
            out = _conv_transpose2d(
                input.unsqueeze(-2),
                weight.unsqueeze(-2),
                bias,
                (1, stride_w),
                (0, padding_w),
                (0, output_padding_w),
                groups,
                (1, dilation_w),
            )
            return out.squeeze(-2)
        if spatial_dims == 2:
            return _conv_transpose2d(
                input, weight, bias, stride, padding, output_padding, groups, dilation
            )
        raise NotImplementedError(
            "convolution_overrideable does not support 3D transposed convolution."
        )

    if spatial_dims == 1:
        return conv1d(input, weight, bias, stride, padding, dilation, groups)
    if spatial_dims == 2:
        return conv2d(input, weight, bias, stride, padding, dilation, groups)
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
    logger.debug("GEMS_KUNLUNXIN CONVOLUTION_OVERRIDEABLE")
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
    logger.debug("GEMS_KUNLUNXIN CONVOLUTION_OVERRIDEABLE_OUT")
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
    )
    out.copy_(result)
    return out
