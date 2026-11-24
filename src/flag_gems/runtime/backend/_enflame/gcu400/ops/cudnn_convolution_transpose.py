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

"""GCU400 implementation of ``aten::cudnn_convolution_transpose``.

GCU400-specific derivation (additive delivery, generic file untouched):

The generic (KernelGen) implementation carries a gather-dot kernel in this
file: one program per (BLOCK_NI_HO_WO, BLOCK_CO) output tile, reducing over
``in_c_per_group * kH * kW`` taps one ``BLOCK_CI`` slice at a time. On GCU400
that kernel measured 451-1498us on the four core benchmark shapes -- the
gather addressing (per-output-pixel ``ih = (ho + pad - kh*dil) // stride``
divisibility test plus two 2D masked loads per tap) defeats the matrix units
and leaves the small conv shapes launch/scan bound.

Instead this file composes the same directory's FlagGems ``conv_transpose2d``
operator (the "other FlagGems operator" host rule): its semantic dispatch
(direct-tiled / scatter no-overlap / pointwise-1x1 / general residue kernels,
plus a fast launch cache) is already tuned for this backend and covers the
full 4D PyTorch conv_transpose2d parameter surface this wrapper admits
(fp32/fp16/bf16, groups, dilation, output_padding). The local gather-dot
kernel is deleted; the autograd Function and its backward (adjoint conv2d +
``torch.nn.grad.conv2d_weight``) keep the generic semantics verbatim, with
the plain-conv2d leg routed through ``flag_gems.conv2d`` so backward also
lands on FlagGems kernels on this backend.
"""

import logging

import torch
import torch.nn.functional  # noqa: F401
import torch.nn.grad  # noqa: F401

import flag_gems

from .conv_transpose2d import conv_transpose2d as _conv_transpose2d_gcu400

logger = logging.getLogger(__name__)


def _pair(value, default=0):
    """Normalize a scalar / (h, w) / None spatial parameter to an (h, w) tuple."""
    if value is None:
        return (default, default)
    if isinstance(value, (list, tuple)):
        first = value[0]
        second = value[1] if len(value) > 1 else value[0]
        return (first, second)
    return (value, value)


class CudnnConvolutionTranspose(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32,
    ):
        logger.debug("GEMS_ENFLAME CUDNN_CONV_TRANSPOSE")
        assert weight.ndim == 4, "Weights must be 4D for conv_transpose2d"
        assert input.ndim == 4, "Input must be 4D for conv_transpose2d"

        stride_h, stride_w = _pair(stride)
        padding_h, padding_w = _pair(padding)
        output_padding_h, output_padding_w = _pair(output_padding)
        dilation_h, dilation_w = _pair(dilation)

        in_n, in_c, input_height, input_width = input.shape
        # Transpose-conv weight layout is (in_c, out_c/groups, kH, kW).
        w_in_c, out_c_per_group, weight_height, weight_width = weight.shape
        assert (
            w_in_c == in_c
        ), f"Incompatible input channels {in_c} and weight in-channels {w_in_c}"
        assert (
            in_c % groups == 0
        ), f"Input channels {in_c} not divisible by groups {groups}"
        out_c = out_c_per_group * groups

        # Composed reuse of this backend's conv_transpose2d: it computes the
        # output geometry, allocates, and launches the semantic-dispatch
        # kernels itself (allocation plus another FlagGems operator only).
        output = _conv_transpose2d_gcu400(
            input,
            weight,
            bias=None,
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            output_padding=(output_padding_h, output_padding_w),
            groups=groups,
            dilation=(dilation_h, dilation_w),
        )

        weight_c = weight.contiguous()

        ctx.save_for_backward(input, weight)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (padding_h, padding_w)
        ctx.output_padding = (output_padding_h, output_padding_w)
        ctx.dilation = (dilation_h, dilation_w)
        ctx.groups = groups
        ctx.weight_info = (out_c, weight_c, weight_height, weight_width)
        ctx.input_info = (in_n, in_c, input_height, input_width)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ENFLAME CUDNN_CONV_TRANSPOSE_VJP")
        input, weight = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        dilation = ctx.dilation
        groups = ctx.groups

        # Transpose convolution is the adjoint of a plain convolution that uses
        # the same weight W (layout (in_c, out_c/groups, kH, kW)):
        #     conv_transpose2d(x, W) == conv2d^T(x, W)
        # so its input-grad is a plain conv2d and its weight-grad is torch's own
        # conv2d weight-grad with input and grad_output swapped.
        #
        # output_padding extends the output by a strip on the bottom/right. When
        # output_padding < stride the conv2d below already absorbs it (its stride
        # floor maps the extended output back to the input size), and the strip
        # carries real gradient, so it must NOT be cropped. When
        # output_padding >= stride (only reachable when dilation > stride, e.g.
        # stride=1/dilation=2/output_padding=1) the strip falls outside every
        # input position, carries no gradient, and would otherwise cause a size
        # mismatch, so crop it off grad_output before the adjoint.
        op_h, op_w = output_padding
        stride_h, stride_w = stride
        crop_h = op_h if op_h >= stride_h else 0
        crop_w = op_w if op_w >= stride_w else 0
        if crop_h:
            grad_output = grad_output[..., : grad_output.shape[-2] - crop_h, :]
        if crop_w:
            grad_output = grad_output[..., : grad_output.shape[-1] - crop_w]

        grad_input = flag_gems.conv2d(
            grad_output,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        grad_weight = torch.nn.grad.conv2d_weight(
            grad_output,
            weight.shape,
            input,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        return (
            grad_input,
            grad_weight,
            None,  # padding
            None,  # output_padding
            None,  # stride
            None,  # dilation
            None,  # groups
            None,  # benchmark
            None,  # deterministic
            None,  # allow_tf32
        )


def cudnn_convolution_transpose(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding=None,
    output_padding=None,
    stride=1,
    dilation=1,
    groups=1,
    benchmark=False,
    deterministic=False,
    allow_tf32=True,
):
    # Handle default values
    logger.debug("GEMS_ENFLAME CUDNN_CONVOLUTION_TRANSPOSE")
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    dilation = _pair(dilation)

    # Inference fast path: when no gradient can flow, skip the autograd
    # Function bookkeeping and dispatch straight to the composed FlagGems
    # operator. The apply() setup/teardown overhead is measurable at the
    # small conv sizes this op runs at.
    if not (torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad)):
        return _conv_transpose2d_gcu400(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

    return CudnnConvolutionTranspose.apply(
        input,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32,
    )
