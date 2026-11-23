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

from flag_gems.utils import libentry

# GCU400-local dependencies: the generic conv2d miscompiles on this backend
# (PAD_LANES=12 workaround yields ~50% wrong elements in fp32, see
# conv2d__opt.py header), and the generic conv_transpose2d rejects non-cuda
# devices while the aten fallback aborts on non-zero output_padding
# (topsaten "output_padding must be zero now", TOPSATEN_STATUS_BAD_PARAM).
from .conv2d__opt import conv2d
from .conv_transpose2d import conv_transpose2d

logger = logging.getLogger(__name__)

# Fixed grid size matching the GCU400 SIP topology (4 DIEs x 6 SIPs x 8
# sub-threads); the accumulate kernel strides over (n, c) planes so that the
# whole tensor is covered by 48 resident programs.
ACCUMULATE_GRID_SIZE = 48
# Elements handled per vector iteration of the accumulate kernel. Kept well
# below the generic pointwise tile (16384 elements) because the GCU400 DSM
# budget is 896 KiB and the block-ptr pointwise kernel doubles its staging
# buffers when a trailing unit axis is widened to 2.
ACCUMULATE_BLOCK_SIZE = 4096
# Launch geometry of the weight-correlation kernel. 8 warps == one full SIP
# sub-thread group; 4 pipeline stages keep the two operand tiles in flight.
WEIGHT_KERNEL_NUM_WARPS = 8
WEIGHT_KERNEL_NUM_STAGES = 4
# Bounds of the dot K axis of the weight-correlation reducer.
WEIGHT_KERNEL_MIN_BLOCK_K = 16
WEIGHT_KERNEL_MAX_BLOCK_K = 64


@libentry()
@triton.jit
def conv_double_backward_accumulate_kernel(
    out_pointer,
    part0_pointer,
    part1_pointer,
    bias_pointer,
    num_planes,
    plane_size,
    out_channels,
    HAS_PART1: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    # grad_ggO = part0 + part1 + broadcast(ggb) evaluated in one pass.
    #
    # The mapping is plane based: plane_index enumerates the (n, c) pairs of the
    # gO-shaped output, so the per-channel ggb value is a single scalar load per
    # plane and every load/store inside the inner loop stays contiguous. This
    # replaces the chained generic ``add`` calls, whose 4-D tile heuristic picks
    # a tile that exceeds the DSM budget for degenerate shapes like the widened
    # 1-D case [N, C, L, 1].
    pid = tl.program_id(0)
    for plane_index in tl.range(pid, num_planes, GRID_SIZE):
        plane_offset = plane_index * plane_size
        if HAS_BIAS:
            plane_bias = tl.load(bias_pointer + plane_index % out_channels)
        else:
            plane_bias = 0.0
        for block_start in tl.range(0, plane_size, BLOCK_SIZE):
            plane_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = plane_offsets < plane_size
            element_offsets = plane_offset + plane_offsets
            accum = tl.load(part0_pointer + element_offsets, mask=mask, other=0.0)
            if HAS_PART1:
                accum += tl.load(
                    part1_pointer + element_offsets, mask=mask, other=0.0
                )
            if HAS_BIAS:
                accum += plane_bias
            tl.store(out_pointer + element_offsets, accum, mask=mask)


def _accumulate_grad_ggo(conv_parts, ggb, out_shape, block_size, grid_size):
    """Fuse the grad_ggO contributions into a single Triton pass.

    ``conv_parts`` holds one or two already-materialised convolution results of
    shape ``out_shape``; ``ggb`` is the optional per-channel bias contribution.
    Returns a freshly allocated, contiguous gO-shaped tensor.
    """
    part0 = conv_parts[0]
    part1 = conv_parts[1] if len(conv_parts) > 1 else part0
    has_part1 = len(conv_parts) > 1
    has_bias = ggb is not None
    bias_tensor = ggb if has_bias else part0
    out = torch.empty(out_shape, device=part0.device, dtype=part0.dtype)
    num_planes = out_shape[0] * out_shape[1]
    plane_size = out_shape[2] * out_shape[3]
    grid = (min(grid_size, num_planes),)
    conv_double_backward_accumulate_kernel[grid](
        out,
        part0,
        part1,
        bias_tensor,
        num_planes,
        plane_size,
        out_shape[1],
        HAS_PART1=has_part1,
        HAS_BIAS=has_bias,
        BLOCK_SIZE=block_size,
        GRID_SIZE=grid_size,
        num_warps=4,
    )
    return out


def _pair(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return int(x[0]), int(x[0])
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _conv_out_size(in_size, kernel_size, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _conv_transpose_out_size(
    in_size, kernel_size, stride, padding, dilation, output_padding
):
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + 1
        + output_padding
    )


@libentry()
@triton.jit
def conv_double_backward_weight_kernel(
    input_pointer,  # tensor whose channels are the "input" channels of the weight
    out_grad_pointer,  # tensor whose channels are the "output" channels of the weight
    weight_pointer,  # output gradient w.r.t. weight
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    input_height,
    input_width,
    weight_height,
    weight_width,
    input_c,
    in_n,
    stride_height,
    stride_width,
    out_height,
    out_width,
    out_c,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_CI_HK_WK: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    REDUCE_ALONG_WIDTH: tl.constexpr,
):
    # Computes the correlation
    #   weight[c_out, c_in, kh, kw] =
    #       sum_{n, h, w} input[n, c_in, kh*dil - pad + stride*h, kw*dil - pad + stride*w]
    #                  * out_grad[n, c_out, h, w]
    # where input/out_grad are the two operands of the double-backward whose channel
    # axes become the (c_in, c_out) axes of the resulting weight gradient.
    #
    # The dot's K axis is the *contiguous* spatial axis of the operands (w when it
    # is wide enough, else h for the widened 1-D layout). Reducing along the batch
    # axis instead made both operand tiles fully strided -- every element needed
    # its own transaction -- which dominated the runtime.
    pid_ci_hk_wk = tl.program_id(0)
    pid_groups = tl.program_id(1)
    pid_co = tl.program_id(2)

    ci_hk_wk_offset = pid_ci_hk_wk * BLOCK_CI_HK_WK + tl.arange(0, BLOCK_CI_HK_WK)
    ci_hk_offset = ci_hk_wk_offset // weight_width
    ci_point_value = ci_hk_offset // weight_height
    weight_height_point_value = ci_hk_offset % weight_height
    weight_width_point_value = ci_hk_wk_offset % weight_width

    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    output_c_valid = output_c_offset < out_c

    out_grad_base = out_grad_pointer + (
        pid_groups * out_c + output_c_offset
    ) * output_c_stride

    weight_pointer += (
        (pid_groups * out_c + output_c_offset) * weight_n_stride
    )[None, :] + (
        ci_point_value * weight_c_stride
        + weight_height_point_value * weight_height_stride
        + weight_width_point_value * weight_width_stride
    )[
        :, None
    ]

    input_base = input_pointer + (
        pid_groups * input_c + ci_point_value
    ) * input_c_stride

    # Per-filter-tap offset of the input window; constant over the reduction.
    input_height_origin = (
        weight_height_point_value * dilation_height - padding_height
    )
    input_width_origin = weight_width_point_value * dilation_width - padding_width
    area_valid = (
        (ci_point_value < input_c)
        & (weight_height_point_value < weight_height)
        & (weight_width_point_value < weight_width)
    )

    accum = tl.zeros((BLOCK_CI_HK_WK, BLOCK_CO), dtype=tl.float32)
    if REDUCE_ALONG_WIDTH:
        # K axis == the w axis: unit stride in both operands, so the tile loads
        # stay contiguous and vectorisable.
        w_lane = tl.arange(0, BLOCK_W)
        for n in range(0, in_n):
            out_grad_batch = out_grad_base[None, :] + n * output_n_stride
            input_batch = input_base[:, None] + n * input_n_stride
            for h in range(0, out_height):
                curr_input_height = (
                    input_height_origin + h * stride_height
                )[:, None]
                for w_base in range(0, out_width, BLOCK_W):
                    curr_w = w_base + w_lane
                    w_valid = curr_w < out_width
                    curr_input_width = (
                        input_width_origin[:, None]
                        + (curr_w * stride_width)[None, :]
                    )
                    curr_input = tl.load(
                        input_batch
                        + curr_input_height * input_height_stride
                        + curr_input_width * input_width_stride,
                        mask=area_valid[:, None]
                        & w_valid[None, :]
                        & (curr_input_height >= 0)
                        & (curr_input_height < input_height)
                        & (curr_input_width >= 0)
                        & (curr_input_width < input_width),
                        other=0.0,
                    )
                    curr_out_grad = tl.load(
                        out_grad_batch
                        + h * output_height_stride
                        + (curr_w * output_width_stride)[:, None],
                        mask=w_valid[:, None] & output_c_valid[None, :],
                        other=0.0,
                    )
                    accum += tl.dot(curr_input, curr_out_grad, allow_tf32=False)
    else:
        # Degenerate width (the widened 1-D case): reduce along h instead, which
        # is the axis that becomes contiguous after the unsqueeze.
        h_lane = tl.arange(0, BLOCK_H)
        out_grad_mask = output_c_valid[None, :]
        for n in range(0, in_n):
            out_grad_batch = out_grad_base[None, :] + n * output_n_stride
            input_batch = input_base[:, None] + n * input_n_stride
            for w in range(0, out_width):
                for h_base in range(0, out_height, BLOCK_H):
                    curr_h = h_base + h_lane
                    h_valid = curr_h < out_height
                    curr_input_height = (
                        input_height_origin[:, None]
                        + (curr_h * stride_height)[None, :]
                    )
                    curr_input_width = (input_width_origin + w * stride_width)[
                        :, None
                    ]
                    curr_input = tl.load(
                        input_batch
                        + curr_input_height * input_height_stride
                        + curr_input_width * input_width_stride,
                        mask=area_valid[:, None]
                        & h_valid[None, :]
                        & (curr_input_height >= 0)
                        & (curr_input_height < input_height)
                        & (curr_input_width >= 0)
                        & (curr_input_width < input_width),
                        other=0.0,
                    )
                    curr_out_grad = tl.load(
                        out_grad_batch
                        + w * output_width_stride
                        + (curr_h * output_height_stride)[:, None],
                        mask=out_grad_mask & h_valid[:, None],
                        other=0.0,
                    )
                    accum += tl.dot(curr_input, curr_out_grad, allow_tf32=False)

    tl.store(
        weight_pointer,
        accum,
        area_valid[:, None] & output_c_valid[None, :],
    )


def _pick_weight_tile(per_group_out_c):
    """Deterministic tile choice for the weight-correlation kernel.

    (BLOCK_CI_HK_WK, BLOCK_CO) follows the number of output channels a group
    owns: a group with >= 128 channels keeps the tensor-core friendly 64x64
    tile, while narrow outputs prefer 16x16/32x32 because the smaller tile
    buys more resident programs on the 48-SIP GCU400 grid.
    """
    if per_group_out_c >= 128:
        block_area = 64
    elif per_group_out_c <= 16:
        block_area = 16
    else:
        block_area = 32
    return block_area, block_area


def _pick_reduction_axis(out_width, out_height):
    """Choose the dot K axis: the spatial axis that is contiguous in memory.

    ``w`` is contiguous for both operands whenever the output is at least one
    vector wide; the widened 1-D layout (``out_width == 1``) has to reduce along
    ``h`` instead so that the dot still gets a vector-sized K axis.
    """
    if out_width >= WEIGHT_KERNEL_MIN_BLOCK_K:
        block_w = min(
            max(triton.next_power_of_2(out_width), WEIGHT_KERNEL_MIN_BLOCK_K),
            WEIGHT_KERNEL_MAX_BLOCK_K,
        )
        return True, block_w, 1
    block_h = min(
        max(triton.next_power_of_2(out_height), WEIGHT_KERNEL_MIN_BLOCK_K),
        WEIGHT_KERNEL_MAX_BLOCK_K,
    )
    return False, 1, block_h


def _grad_weight_correlation(
    input, out_grad, weight_shape, stride, padding, dilation, groups
):
    """Compute the weight-shaped gradient via the correlation kernel above.

    weight[c_out, c_in, kh, kw] = sum_{n, h, w} input[n, c_in, ...] * out_grad[n, c_out, ...]
    """
    out_c_total, weight_c, weight_height, weight_width = weight_shape
    out_c = out_c_total // groups  # per-group output channels
    in_n = input.shape[0]
    input_height, input_width = input.shape[2], input.shape[3]
    out_height, out_width = out_grad.shape[2], out_grad.shape[3]
    stride_height, stride_width = _pair(stride)
    padding_height, padding_width = _pair(padding)
    dilation_height, dilation_width = _pair(dilation)

    grad_weight = torch.empty(
        weight_shape,
        device=input.device,
        dtype=input.dtype,
    )

    block_ci_hk_wk, block_co = _pick_weight_tile(out_c)
    reduce_along_width, block_w, block_h = _pick_reduction_axis(
        out_width, out_height
    )

    grid = (
        triton.cdiv(weight_c * weight_height * weight_width, block_ci_hk_wk),
        groups,
        triton.cdiv(out_c, block_co),
    )
    conv_double_backward_weight_kernel[grid](
        input,
        out_grad,
        grad_weight,
        *input.stride(),
        *grad_weight.stride(),
        *out_grad.stride(),
        input_height,
        input_width,
        weight_height,
        weight_width,
        weight_c,
        in_n,
        stride_height,
        stride_width,
        out_height,
        out_width,
        out_c,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        BLOCK_W=block_w,
        BLOCK_H=block_h,
        BLOCK_CI_HK_WK=block_ci_hk_wk,
        BLOCK_CO=block_co,
        REDUCE_ALONG_WIDTH=reduce_along_width,
        num_warps=WEIGHT_KERNEL_NUM_WARPS,
        num_stages=WEIGHT_KERNEL_NUM_STAGES,
    )
    return grad_weight


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
    """Second-order backward of a 2D convolution.

    Let ``y = conv(x, w, b)`` with ``(gX, gW, gB) = backward(gO)``. Given the
    second-order grad-outputs ``(ggI, ggW, ggb)`` for ``(gX, gW, gB)``, this returns
    the gradients of ``<gX, ggI> + <gW, ggW> + <gB, ggb>`` with respect to
    ``(gO, x, w)``, i.e. ``(grad_ggO, grad_self, grad_weight)``.
    """
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    # spatial dims of the forward output (== gO spatial dims)
    gO_h, gO_w = gO.shape[2], gO.shape[3]
    # spatial dims of the forward input (== self spatial dims)
    self_h, self_w = self.shape[2], self.shape[3]
    kH, kW = weight.shape[2], weight.shape[3]

    grad_ggO = None
    grad_self = None
    grad_weight = None

    # Note: aten::_convolution_double_backward ignores ``output_mask`` for its
    # outputs and always returns all three gradients (it only ever forwards the
    # mask down to the underlying cudnn call). To match that behaviour we compute
    # every requested gradient whenever its grad-output input is available; a
    # missing ggI / ggW / ggb simply drops the corresponding contribution.

    # grad_ggO = sum of contributions from ggI, ggW and ggb.
    conv_parts = []
    if ggI is not None:
        if not transposed:
            # contribution of ggI: this is the adjoint of the input-gradient op,
            # which is a plain forward convolution of ggI with the weight using
            # the *forward* stride/padding/dilation (output spatial size == gO's).
            conv_parts.append(
                conv2d(
                    ggI,
                    weight,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            )
        else:
            # contribution of ggI: transposed convolution of ggI with the weight.
            conv_parts.append(
                conv_transpose2d(
                    ggI,
                    weight,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    dilation=dilation,
                )
            )
    if ggW is not None:
        if not transposed:
            # contribution of ggW: forward convolution of x with ggW.
            conv_parts.append(
                conv2d(
                    self,
                    ggW,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            )
        else:
            # contribution of ggW: transposed convolution of x with ggW.
            conv_parts.append(
                conv_transpose2d(
                    self,
                    ggW,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    dilation=dilation,
                )
            )

    if len(conv_parts) > 1 or (conv_parts and ggb is not None):
        # Two convolution contributions and/or the per-channel bias broadcast:
        # fuse them into one Triton pass over the gO-shaped output.
        grad_ggO = _accumulate_grad_ggo(
            conv_parts,
            ggb,
            conv_parts[0].shape,
            ACCUMULATE_BLOCK_SIZE,
            ACCUMULATE_GRID_SIZE,
        )
    elif conv_parts:
        # A single contribution with no bias needs no accumulation at all.
        grad_ggO = conv_parts[0]
    else:
        # Only the ggb contribution: broadcast the per-channel value over the
        # output without any arithmetic.
        bias_part = ggb.view(1, -1, 1, 1).expand(1, ggb.shape[0], gO_h, gO_w)
        # match batch dim of gO for broadcasting.
        grad_ggO = bias_part.expand(gO.shape[0], -1, -1, -1)

    if ggW is not None:
        # grad_self only depends on ggW.
        if not transposed:
            # input-gradient form: transposed convolution of gO with ggW.
            # Recover the self spatial size via output_padding.
            opad_self_h = self_h - (
                (gO_h - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + 1
            )
            opad_self_w = self_w - (
                (gO_w - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + 1
            )
            grad_self = conv_transpose2d(
                gO,
                ggW,
                stride=stride,
                padding=padding,
                output_padding=(opad_self_h, opad_self_w),
                groups=groups,
                dilation=dilation,
            )
        else:
            # transposed forward: grad_self is a forward convolution of gO with ggW.
            grad_self = conv2d(
                gO,
                ggW,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

    if ggI is not None:
        # grad_weight only depends on ggI.
        if not transposed:
            # weight[c_out, c_in, kh, kw] = sum ggI[c_in] * gO[c_out]
            grad_weight = _grad_weight_correlation(
                ggI, gO, weight.shape, stride, padding, dilation, groups
            )
        else:
            # transposed forward: weight[c_in, c_out, kh, kw] = sum gO[c_out] * ggI[c_in]
            grad_weight = _grad_weight_correlation(
                gO, ggI, weight.shape, stride, padding, dilation, groups
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

    1D convolutions (3-D operands ``[N, C, L]``) are handled by routing through
    the 2D implementation: a trailing spatial axis is unsqueezed
    (``[N, C, L]`` -> ``[N, C, L, 1]``), the 2D path runs, and the extra axis is
    squeezed back on every output. This mirrors how ``conv1d`` delegates to
    ``conv2d``. ``ggb`` is a 1-D channel vector and is passed through unchanged.
    """

    # Widen the lone spatial axis to a (degenerate) 2D one and pass it to the
    # 2D path. Scalar/int 1D params become length-2 lists with the second axis
    # set to identity (stride 1, no padding, dilation 1, no output padding).
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
    logger.debug("GEMS_ENFLAME _CONVOLUTION_DOUBLE_BACKWARD")

    # Dispatch by spatial arity (matches the convention used by
    # ``_convolution_mode``: ``spatial_ndim = weight.ndim - 2``). 1D convolutions
    # are routed through the 2D path by unsqueezing the lone spatial axis
    # (mirroring ``conv1d``, which delegates to ``conv2d`` the same way).
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
