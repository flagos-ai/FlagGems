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

"""Kunlunxin override for ``aten::convolution_backward_overrideable``.

The generic FlagGems implementation (``src/flag_gems/ops/
convolution_backward_overrideable.py``, read-only for this backend) funnels the
three gradients into generic helpers that cannot run on TritonXPU:

* ``_conv2d_grad_input`` dispatches to the generic ``conv_transpose2d``, whose
  residue kernel issues ``tl.dot(..., input_precision="tf32x3")``.  TritonXPU
  only accepts ``('ieee', 'tf32')``, so every grad_input case dies with
  ``AssertionError: input_precision must be one of ('ieee', 'tf32'). Got
  tf32x3``.
* ``_conv2d_grad_weight`` reuses the generic ``conv2d_backward_kernel_weight``,
  a masked 2-D ``tl.dot`` kernel; on XPU that path returns wrong values (the
  whole grad_weight tensor mismatches with relative error 1.0).

This module keeps the exact same mathematical decomposition (grad_input =
transposed convolution of grad_output, grad_weight = im2col contraction of
input with grad_output, grad_bias = spatial reduction of grad_output) but
routes each of the three pieces through the already-shipped, XPU-validated
Kunlunxin Triton kernels:

* grad_input  -> ``_kunlunxin/ops/conv2d.py::_input_grad``
                 (rank-1 gather kernel, clamped loads, no ``tl.dot``);
* grad_weight -> ``_kunlunxin/ops/conv2d.py::_weight_grad``
                 (rank-1 scalar-loop kernel, no ``tl.dot``, clamped loads);
* grad_bias   -> ``_kunlunxin/ops/conv2d.py::_bias_grad``
                 (rank-1 reduction, no ``tl.dot``).

The generic decomposition is preserved mathematically: grad_input is the
transposed convolution of grad_output with the same weight, which is exactly
what ``_input_grad`` computes for the forward conv2d (and it already handles
stride/padding/dilation/group bookkeeping plus the stride-truncation edges that
``output_padding`` would otherwise patch up).

No torch/aten compute primitive is invoked here; the only torch calls are
allocations (``torch.empty``/``empty_like``) plus the ``copy_`` used by the
``.out`` variant to hand results back into caller-provided buffers.
"""

import logging

import torch
import triton

from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _bias_grad as _conv2d_bias_grad,
)
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _input_grad as _conv2d_input_grad,
)
from flag_gems.runtime.backend._kunlunxin.ops.conv2d import (
    _weight_grad as _conv2d_weight_grad,
)

logger = logging.getLogger(__name__)

# Reduction tile used by the two rank-1 backward kernels (same width the vendor
# conv2d backward launches today; 64 lanes is the smallest tile that keeps the
# XPU `tl.sum` codegen on the oracle path).
_BP = 64
# Input-gradient spatial tile (matches the vendor conv2d backward launch).
_BM = 32


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _grad_input(grad_output, weight, input, stride, padding, dilation, groups):
    in_n, in_c, in_h, in_w = input.shape
    out_c, cpg, kh, kw = weight.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]
    grad_input = torch.empty_like(input, memory_format=torch.contiguous_format)
    _conv2d_input_grad[(triton.cdiv(in_n * in_h * in_w, _BM), in_c)](
        grad_output,
        weight,
        grad_input,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        cpg,
        out_c // groups,
        kh,
        kw,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        *grad_output.stride(),
        *weight.stride(),
        *grad_input.stride(),
        BM=_BM,
        KH=kh,
        KW=kw,
        OPG=out_c // groups,
        CPG=cpg,
    )
    return grad_input


def _grad_weight(grad_output, input, weight, stride, padding, dilation, groups):
    out_channels, weight_channels, weight_height, weight_width = weight.shape
    in_n, _, input_height, input_width = input.shape
    out_height, out_width = grad_output.shape[2], grad_output.shape[3]
    out_channels_per_group = out_channels // groups

    grad_weight = torch.empty_like(weight, memory_format=torch.contiguous_format)
    _conv2d_weight_grad[
        (out_channels * weight_channels * weight_height * weight_width,)
    ](
        input,
        grad_output,
        grad_weight,
        in_n,
        input_height,
        input_width,
        out_channels,
        out_height,
        out_width,
        weight_channels,
        out_channels_per_group,
        weight_height,
        weight_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        *input.stride(),
        *grad_output.stride(),
        *grad_weight.stride(),
        BP=_BP,
        BK=_BP,
    )
    return grad_weight


def convolution_backward_overrideable(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS KUNLUNXIN CONVOLUTION_BACKWARD_OVERRIDEABLE")
    if transposed:
        raise RuntimeError(
            "convolution_backward_overrideable: transposed not supported"
        )
    if input.ndim != 4 or weight.ndim != 4:
        raise RuntimeError(
            "convolution_backward_overrideable currently supports 2D convolution only"
        )

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    groups = int(groups)

    grad_input = None
    grad_weight = None
    grad_bias = None
    if output_mask[0]:
        grad_input = _grad_input(
            grad_output, weight, input, stride, padding, dilation, groups
        )
    if output_mask[1]:
        grad_weight = _grad_weight(
            grad_output, input, weight, stride, padding, dilation, groups
        ).to(weight.dtype)
    if output_mask[2]:
        out_channels = grad_output.shape[1]
        _, _, out_height, out_width = grad_output.shape
        grad_bias = torch.empty(
            (out_channels,), dtype=grad_output.dtype, device=grad_output.device
        )
        _conv2d_bias_grad[(out_channels,)](
            grad_output,
            grad_bias,
            grad_output.shape[0],
            out_channels,
            out_height,
            out_width,
            *grad_output.stride(),
            BP=_BP,
            BO=_BP,
        )
    return grad_input, grad_weight, grad_bias


def convolution_backward_overrideable_out(
    grad_output,
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    out0,
    out1,
    out2,
):
    logger.debug("GEMS KUNLUNXIN CONVOLUTION_BACKWARD_OVERRIDEABLE_OUT")
    grad_input, grad_weight, grad_bias = convolution_backward_overrideable(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    )
    if grad_input is not None:
        out0.copy_(grad_input)
    if grad_weight is not None:
        out1.copy_(grad_weight)
    if grad_bias is not None:
        out2.copy_(grad_bias)
    return out0, out1, out2
