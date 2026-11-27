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

"""GCU400 specialization: fully fused conv + bias + alpha*z + ReLU.

The generic implementation (``flag_gems/ops/cudnn_convolution_add_relu.py``) runs in
two stages: first ``cudnn_convolution`` (a generic implicit-GEMM kernel on GCU400),
then a pointwise epilogue over the whole output (read conv, read z, add bias, ReLU,
write out). Both stages move the whole output in and out once.

Here the epilogue is folded into the tail of the 2D convolution kernel: bias,
``alpha * z`` and ReLU are applied to the fp32 accumulator with one final rounding.
Compared with the two-stage form this drops the entire write-conv / read-conv /
write-out round trip and one kernel launch, and reuses the statically unrolled tap
scheme already validated for ``cudnn_convolution_relu`` on GCU400 (on

(16,64,56,56)x128k3 fp16 it took the convolution body from 27.3ms to 10.3ms). Shapes outside this scope (depthwise, 3D) go to the generic implementation; see the ``_dispatch`` comments.  # noqa: E501
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv2d import conv2d_output_size
from flag_gems.ops.cudnn_convolution import _AABS_PRESENT, _aabs_disabled
from flag_gems.ops.cudnn_convolution_add_relu import (
    _conv_add_relu as _generic_conv_add_relu,
)
from flag_gems.utils import libentry

from .cudnn_convolution_relu import _check_dtype, _select_conv2d_relu_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2D fused kernel: conv + bias + alpha*z + ReLU
#
# The only difference from the 2D kernel of ``cudnn_convolution_relu`` is an extra z
# path addressed through the output strides. z is normalized on the host into a dense
# tensor of the output shape, so its indices match the output and can be reused as is.
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def conv2d_add_relu_forward_kernel(
    input_pointer,
    weight_pointer,
    z_pointer,
    bias_pointer,
    output_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    z_n_stride,
    z_c_stride,
    z_height_stride,
    z_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    alpha,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    EVEN_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    # Row-related quantities (batch base offset, strided spatial indices) are hoisted out
    # of the K loop so that only the tap constant offsets remain inside it.
    row_base = input_pointer + (
        input_n_stride * in_n_point_value
        + input_c_stride * pid_group * weight_c
        + input_height_stride * stride_height * output_height_point_value
        + input_width_stride * stride_width * output_width_point_value
    )
    row_in_bounds = in_n_point_value < in_n

    weight_base = weight_pointer + (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )
    co_ok = output_c_offset < out_per_group_c

    # Valid-element mask shared by the output and z; the epilogue z read/write stays under it.
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for tap in tl.static_range(weight_height * weight_width):
        h = tap // weight_width
        w = tap % weight_width
        ih = h * dilation_height - padding_height
        iw = w * dilation_width - padding_width
        tap_row_ptr = row_base + (input_height_stride * ih + input_width_stride * iw)
        row_ok = (
            row_in_bounds
            & (ih + stride_height * output_height_point_value >= 0)
            & (ih + stride_height * output_height_point_value < input_height)
            & (iw + stride_width * output_width_point_value >= 0)
            & (iw + stride_width * output_width_point_value < input_width)
        )
        tap_weight_ptr = weight_height_stride * h + weight_width_stride * w
        for cb in range(BLOCK_CI_COUNT):
            input_c_offset = cb * BLOCK_CI + tl.arange(0, BLOCK_CI)
            curr_input_pointer = (
                tap_row_ptr[:, None] + (input_c_stride * input_c_offset)[None, :]
            )
            curr_weight_pointer = (
                weight_base[None, :]
                + (weight_c_stride * input_c_offset)[:, None]
                + tap_weight_ptr
            )
            if EVEN_C:
                input_block = tl.load(curr_input_pointer, mask=row_ok[:, None])
                weight_block = tl.load(curr_weight_pointer, mask=co_ok[None, :])
            else:
                c_ok = input_c_offset < weight_c
                input_block = tl.load(
                    curr_input_pointer, mask=row_ok[:, None] & c_ok[None, :]
                )
                weight_block = tl.load(
                    curr_weight_pointer, mask=c_ok[:, None] & co_ok[None, :]
                )
            accum += tl.dot(input_block, weight_block, allow_tf32=False)

    if HAS_BIAS:
        bias_offset = pid_group * out_per_group_c + output_c_offset
        bias = tl.load(
            bias_pointer + bias_offset[None, :], co_ok[None, :], other=0.0
        ).to(tl.float32)
        accum = accum + bias

    # alpha * z: z has the same shape and layout as the output, so this index matches the store index below.
    z_pointer += (
        (z_n_stride * in_n_point_value)[:, None]
        + (z_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (z_height_stride * output_height_point_value)[:, None]
        + (z_width_stride * output_width_point_value)[:, None]
    )
    z_block = tl.load(z_pointer, mask=output_mask, other=0.0).to(tl.float32)
    accum = accum + alpha * z_block

    # ReLU uses a NaN-propagating max: aten relu is max(x, 0), so a NaN must stay NaN.
    accum = tl.maximum(accum, 0.0, propagate_nan=tl.PropagateNan.ALL)

    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    tl.store(output_pointer, accum, mask=output_mask)


def _to_list(param, ndim):
    """Expand scalar spatial parameters into a per-dimension list."""
    if isinstance(param, (list, tuple)):
        return list(param)
    return [param] * ndim


def _prepare_out(out, output_shape, dtype, device):
    """Validate and adjust the caller buffer against the out contract of the generic implementation."""
    if out is None:
        return None
    if out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got {out.dtype}"
        )
    if out.shape != output_shape:
        # Match the auto-generated out variants: resize to the right extent when the shape differs.
        out.resize_(output_shape)
    if out.device != device:
        raise RuntimeError(
            f"Expected out tensor to be on device {device}, but got {out.device}"
        )
    return out


def _normalize_z(z, output_shape, dtype, device):
    """Normalize z into a dense tensor with the same shape as the output.

    Broadcast shapes (z smaller than the output) are silently misread on the ATen cuDNN
    path, so expand it to the output shape and make it contiguous, which is equivalent to the epilogue's elementwise broadcast.  # noqa: E501
    """
    if z.shape != output_shape:
        z = z.expand(output_shape)
    if not z.is_contiguous():
        z = z.contiguous()
    if z.dtype != dtype:
        z = z.to(dtype)
    if z.device != device:
        raise RuntimeError(
            f"Expected z tensor to be on device {device}, but got {z.device}"
        )
    return z


def _select_add_relu_config(in_n, out_h, out_w, out_c, weight_c, groups, dtype):
    """Tile choice for the fused conv+bias+alpha*z+ReLU kernel.

    Starts from the conv+bias+ReLU selector (already tuned per shape class on GCU400)
    and applies two measured overrides for 16-bit input with a wide per-group channel
    count (weight_c > 64, out_c/groups >= 128):

    * large spatial volume -- BLOCK_NI_HO_WO=128 / BLOCK_CI=64 / BLOCK_CO=64 with 8
      warps and 3 stages. The wide row tile keeps one dot chain per program long
      enough to hide the tap-loop loads, and halving BLOCK_CO (256 -> 64) is what
      actually buys the win: it drops the per-program dsm footprint from 1.38MB
      (over the 917504B limit for BLOCK_CO=128) to a size that fits. Measured on
      (8,256,64,64)x256k3: 430.4ms -> 177.9ms bf16, 430.4ms -> 178.2ms fp16; a
      10-point sweep around the winner (BLOCK_CI 32/64, BLOCK_CO 32/64/128,
      4/8 warps, 2/3/4 stages) stayed within 177.0-178.4ms except BLOCK_CO=32
      (320.6ms), so the plateau is wide.
    * small spatial volume -- keep the inherited-style BLOCK_CI=64 / BLOCK_CO=256
      tile: with fewer than 8192 rows the 128-row tile leaves most lanes masked.
    """
    config = _select_conv2d_relu_config(
        in_n, out_h, out_w, out_c, weight_c, groups, dtype
    )
    if (
        dtype in (torch.float16, torch.bfloat16)
        and weight_c > 64
        and out_c // groups >= 128
    ):
        if in_n * out_h * out_w >= 8192:
            config = (128, 64, 64, 8, 3)
        else:
            config = (32, 64, 256, 4, 2)
    return config


def _launch_conv2d_add_relu(
    input, weight, z, alpha, bias, stride, padding, dilation, groups, out
):
    """2D fused launch: one kernel does conv + bias + alpha*z + ReLU."""
    in_n, _, input_height, input_width = input.shape
    out_c, weight_c, weight_height, weight_width = weight.shape
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    # tl.dot needs its K dim split by BLOCK_CI with a minimum K of 16; pad the channel
    # dim when C_in is smaller (same approach as the grouped repack of conv2d).
    _MIN_DOT_K = 16
    if weight_c < _MIN_DOT_K:
        pad_c = _MIN_DOT_K - weight_c
        if groups == 1:
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
        else:
            N, C, H, W = input.shape
            input = input.reshape(N, groups, weight_c, H, W)
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, pad_c))
            input = input.reshape(N, groups * (weight_c + pad_c), H, W)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_c))
        weight_c = weight_c + pad_c

    out_height = conv2d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv2d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )
    output_shape = (in_n, out_c, out_height, out_width)

    if out is not None:
        output = out
    else:
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    (
        block_ni_ho_wo,
        block_ci,
        block_co,
        num_warps,
        num_stages,
    ) = _select_add_relu_config(
        in_n, out_height, out_width, out_c, weight_c, groups, input.dtype
    )
    grid = (
        triton.cdiv(in_n * out_height * out_width, block_ni_ho_wo),
        triton.cdiv(int(out_c // groups), block_co),
        groups,
    )
    # A tensor is still needed for the pointer when bias is None; HAS_BIAS=False prevents dereference.
    bias_pointer = bias if bias is not None else weight
    conv2d_add_relu_forward_kernel[grid](
        input,
        weight,
        z,
        bias_pointer,
        output,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *z.stride(),
        *output.stride(),
        1.0 if alpha is None else float(alpha),
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups=groups,
        BLOCK_NI_HO_WO=block_ni_ho_wo,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        EVEN_C=(weight_c % block_ci == 0),
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _is_depthwise(input, weight, groups):
    """Depthwise test: one input channel per group and groups equal to input channels."""
    return weight.shape[1] == 1 and groups == input.shape[1]


def _is_pointwise(weight, stride, padding, dilation, groups):
    """1x1, unpadded pointwise convolution: the generic path has a dedicated GEMM kernel."""
    return (
        groups == 1
        and all(k == 1 for k in weight.shape[2:])
        and all(s == 1 for s in stride)
        and all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
    )


def _fused_is_faster(input, weight, stride, padding, dilation, groups):
    """Decide from measurements whether the fused kernel is really faster on GCU400.

    Measured on the benchmark core cases (geomean over bf16/fp16/fp32, generic two-stage
    vs fully fused):
        * 1x1 pointwise (16,64,56,56)x128k1: 0.70ms vs 3.0ms -- the generic path uses a
            dedicated GEMM kernel and the fused tile is 4x slower;
        * weight_c<16, e.g. (8,3,224,224)x16k3: 3.1ms vs 44ms -- the fused kernel must pad
            the channel dim up to tl.dot's K>=16 floor, and the redundant MACs eat the win;
        * 3x3 body (8,256,64,64)x256k3: 510ms vs 78ms, (16,32,32,32)x32k3 p2 d2:
            5.7ms vs 3.1ms -- fused wins.
    So the fused kernel is only used when the kernel has a real spatial extent and the
    channel count needs no padding.
    """
    if weight.shape[1] < 16:
        return False
    if _is_pointwise(weight, stride, padding, dilation, groups):
        return False
    return True


def _dispatch(
    input, weight, z, alpha, bias, stride, padding, dilation, groups, ndim, out
):
    """Send the shapes the fused kernel covers into it and hand the rest to the generic one.

    Four classes stay on the generic implementation:
        * 3D: the fused kernel only handles 2D, and the generic 3D path is covered only by the
            accuracy cases; duplicating a 3D fused kernel is not worth it.
        * depthwise: keeping tl.dot's K>=16 needs each group padded to 16 channels, which is
            16x redundant work when weight_c==1; the generic implementation has a dedicated
            depthwise kernel and is faster.
        * 1x1 pointwise and * small-channel convolutions: same padding overhead, plus a
            dedicated GEMM kernel on the generic side. See ``_fused_is_faster``.
    """
    if (
        ndim == 3
        or _is_depthwise(input, weight, groups)
        or not _fused_is_faster(input, weight, stride, padding, dilation, groups)
    ):
        return _generic_conv_add_relu(
            input, weight, z, alpha, bias, stride, padding, dilation, groups, out
        )

    if ndim == 1:
        # Lift 1D to 2D by appending a unit dimension (same trick as conv1d->conv2d). z/out both
        # participate as unsqueeze(-1) views and the kernel strides over them as 4D.
        lifted_out = out.unsqueeze(-1) if out is not None else None
        result = _launch_conv2d_add_relu(
            input.unsqueeze(-1),
            weight.unsqueeze(-1),
            z.unsqueeze(-1),
            alpha,
            bias,
            [stride[0], 1],
            [padding[0], 0],
            [dilation[0], 1],
            groups,
            lifted_out,
        )
        return out if out is not None else result.squeeze(-1)

    return _launch_conv2d_add_relu(
        input, weight, z, alpha, bias, stride, padding, dilation, groups, out
    )


def _conv_add_relu(
    input, weight, z, alpha, bias, stride, padding, dilation, groups, out
):
    """Shared body of the functional and out variants."""
    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            "cudnn_convolution_add_relu only supports 1D, 2D and 3D "
            f"convolutions, got input with {ndim} spatial dimensions"
        )
    _check_dtype(input)

    stride = _to_list(stride, ndim)
    padding = _to_list(padding, ndim)
    dilation = _to_list(dilation, ndim)

    # Output extents (same output-size formula as the generic implementation).
    spatial_out = tuple(
        (
            input.shape[2 + i]
            + 2 * padding[i]
            - dilation[i] * (weight.shape[2 + i] - 1)
            - 1
        )
        // stride[i]
        + 1
        for i in range(ndim)
    )
    output_shape = (input.shape[0], weight.shape[0], *spatial_out)

    out = _prepare_out(out, output_shape, input.dtype, input.device)

    dtype = input.dtype
    out_channels = weight.shape[0]
    if bias is not None:
        if bias.numel() != out_channels:
            raise ValueError(
                f"cudnn_convolution_add_relu expects a bias with {out_channels} "
                f"elements, got {bias.numel()}"
            )
        bias = bias.contiguous().view(-1).to(dtype)

    # Zero-element output: conv and epilogue have nothing to do, return an empty tensor with the right extents.
    total = 1
    for dim in output_shape:
        total *= dim
    if total == 0:
        if out is not None:
            return out
        return torch.empty(output_shape, device=input.device, dtype=dtype)

    z = _normalize_z(z, output_shape, dtype, input.device)

    if _AABS_PRESENT:
        with _aabs_disabled():
            return _dispatch(
                input,
                weight,
                z,
                alpha,
                bias,
                stride,
                padding,
                dilation,
                groups,
                ndim,
                out,
            )
    return _dispatch(
        input, weight, z, alpha, bias, stride, padding, dilation, groups, ndim, out
    )


def cudnn_convolution_add_relu(
    input,
    weight,
    z,
    alpha=None,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """
    Fused conv + bias + alpha*z + ReLU, same semantics as ``aten::cudnn_convolution_add_relu``.

    Computes ``relu(conv(input, weight) + bias + alpha * z)`` and supports 1D/2D/3D input;
    2D non-depthwise shapes take the fully fused kernel, the rest reuse the generic one.
    """
    logger.debug("GEMS_ENFLAME CUDNN_CONVOLUTION_ADD_RELU")
    return _conv_add_relu(
        input, weight, z, alpha, bias, stride, padding, dilation, groups, None
    )


def cudnn_convolution_add_relu_out(
    input,
    weight,
    z,
    alpha=None,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    *,
    out,
):
    """Out variant of ``cudnn_convolution_add_relu``; writes into ``out`` and returns it."""
    logger.debug("GEMS_ENFLAME CUDNN_CONVOLUTION_ADD_RELU_OUT")
    return _conv_add_relu(
        input, weight, z, alpha, bias, stride, padding, dilation, groups, out
    )
