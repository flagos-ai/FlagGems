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

"""Triton implementation of ``torch.nn.functional.conv_transpose2d``.

The implementation uses semantic, parameter-regime dispatch only: a direct
tiled path for common dense group=1 cases, a pointwise 1x1 path, a scatter path
for no-overlap sparse-output cases, and a full residue path for the supported
PyTorch API surface.  There are no shape-specific dispatch constants.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_TRITON_DIRECT_LOWP_DTYPES = (torch.float16, torch.bfloat16)

_GENERAL_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_DIRECT_TILED_FAMILY_MAX_CHANNELS = 256
_DIRECT_TILED_FAMILY_MAX_KERNEL = 5
_DIRECT_TILED_FAMILY_MAX_STRIDE = 4
_DIRECT_TILED_OUTPUT_PADDING_MIN_INPUT_ELEMENTS = 1024
_DIRECT_TILED_DEFAULT_SCHEDULE = (64, 32, 32, 4)
_DIRECT_STRIDE2_PAD1_3X3_MAX_CHANNELS = 256


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise RuntimeError("expected a single int or a pair of ints")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _direct_tiled_family_params(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    if bias is not None or groups != 1:
        return None
    if (dilation_h, dilation_w) != (1, 1):
        return None
    if input.dtype not in _GENERAL_TRITON_DTYPES or weight.dtype != input.dtype:
        return None
    if input.device.type != "cuda" or weight.device != input.device:
        return None
    if input.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return None
    if input.dim() != 4 or weight.dim() != 4:
        return None
    if not input.is_contiguous() or not weight.is_contiguous():
        return None
    if stride_h != stride_w or padding_h != padding_w:
        return None
    if output_padding_h != output_padding_w:
        return None
    if stride_h <= 0 or stride_h > _DIRECT_TILED_FAMILY_MAX_STRIDE:
        return None
    if padding_h < 0 or output_padding_h < 0:
        return None

    batch, input_channels, input_height, input_width = input.shape
    weight_input_channels, output_channels, weight_height, weight_width = weight.shape
    if batch <= 0 or input_height <= 0 or input_width <= 0:
        return None
    if input_channels != weight_input_channels:
        return None
    if input_channels < 16 or output_channels < 16:
        return None
    if (
        input_channels > _DIRECT_TILED_FAMILY_MAX_CHANNELS
        or output_channels > _DIRECT_TILED_FAMILY_MAX_CHANNELS
    ):
        return None
    if (
        weight_height <= 0
        or weight_width <= 0
        or weight_height > _DIRECT_TILED_FAMILY_MAX_KERNEL
        or weight_width > _DIRECT_TILED_FAMILY_MAX_KERNEL
    ):
        return None
    output_height = (
        (input_height - 1) * stride_h - 2 * padding_h + weight_height + output_padding_h
    )
    output_width = (
        (input_width - 1) * stride_w - 2 * padding_w + weight_width + output_padding_w
    )
    if output_height <= 0 or output_width <= 0:
        return None
    return (
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        padding_h,
    )


def _can_use_direct_tiled_family(
    input,
    direct_tiled_family_params,
    output_padding_h,
):
    if direct_tiled_family_params is None:
        return False
    (
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        _padding_h,
    ) = direct_tiled_family_params

    if output_padding_h == 0 and stride_h <= 2:
        return True
    input_elements = batch * input_height * input_width
    if (
        input.dtype in _GENERAL_TRITON_DTYPES
        and stride_h == 2
        and output_padding_h == 1
        and weight_height == 3
        and weight_width == 3
        and input_channels >= 64
        and output_channels <= 64
        and input_elements >= _DIRECT_TILED_OUTPUT_PADDING_MIN_INPUT_ELEMENTS
    ):
        return True
    if stride_h >= 3 and output_padding_h == 0:
        if weight_height >= 5 or weight_width >= 5:
            return True
        if input.dtype in _TRITON_DIRECT_LOWP_DTYPES:
            return True
    return False


def _unsupported_conv_transpose2d(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    bias_dtype = None if bias is None else bias.dtype
    raise NotImplementedError(
        "flag_gems.conv_transpose2d supports 3D or 4D CUDA input tensors "
        "and 4D CUDA weight tensors with float32, float16, or bfloat16 dtype; got "
        f"input_shape={tuple(input.shape)}, weight_shape={tuple(weight.shape)}, "
        f"input_dtype={input.dtype}, weight_dtype={weight.dtype}, bias_dtype={bias_dtype}, "
        f"input_device={input.device}, weight_device={weight.device}, "
        f"stride=({stride_h}, {stride_w}), padding=({padding_h}, {padding_w}), "
        f"output_padding=({output_padding_h}, {output_padding_w}), groups={groups}, "
        f"dilation=({dilation_h}, {dilation_w})"
    )


def _validate_conv_transpose2d_args(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    if input.device.type != "cuda" or weight.device != input.device:
        return False
    if input.dim() != 4 or weight.dim() != 4:
        return False
    if not input.is_contiguous() or not weight.is_contiguous():
        return False
    if input.dtype not in _GENERAL_TRITON_DTYPES or weight.dtype != input.dtype:
        return False
    if input.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return False
    if bias is not None:
        if bias.device != input.device or bias.dtype != input.dtype:
            return False
        if bias.dim() != 1 or not bias.is_contiguous():
            return False
    if groups <= 0:
        raise RuntimeError("groups must be a positive integer")
    if stride_h <= 0 or stride_w <= 0:
        raise RuntimeError("non-positive stride is not supported")
    if dilation_h <= 0 or dilation_w <= 0:
        raise RuntimeError("dilation should be greater than zero")
    if padding_h < 0 or padding_w < 0:
        raise RuntimeError("negative padding is not supported")
    if output_padding_h < 0 or output_padding_w < 0:
        raise RuntimeError("negative output_padding is not supported")
    if output_padding_h >= stride_h and output_padding_h >= dilation_h:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )
    if output_padding_w >= stride_w and output_padding_w >= dilation_w:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )

    input_channels = input.shape[1]
    weight_input_channels = weight.shape[0]
    output_channels_per_group = weight.shape[1]
    weight_height = weight.shape[2]
    weight_width = weight.shape[3]
    if (
        input_channels <= 0
        or output_channels_per_group <= 0
        or weight_height <= 0
        or weight_width <= 0
    ):
        raise RuntimeError(
            "non-empty input channels and weight dimensions are required"
        )
    if input_channels != weight_input_channels:
        raise RuntimeError(
            "expected input channel dimension to match weight input channels"
        )
    if input_channels % groups != 0:
        raise RuntimeError("input channels must be divisible by groups")
    output_channels = output_channels_per_group * groups
    if bias is not None and bias.numel() != output_channels:
        raise RuntimeError("expected bias to have one element per output channel")

    input_height = input.shape[2]
    input_width = input.shape[3]
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (weight_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (weight_width - 1)
        + output_padding_w
        + 1
    )
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError("calculated output size is too small")
    return True


def _can_use_scatter_no_overlap(
    input,
    weight,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    groups,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, weight_height, weight_width = weight.shape
    if batch <= 0 or input_height <= 0 or input_width <= 0:
        return False
    effective_kernel_h = (weight_height - 1) * dilation_h + 1
    effective_kernel_w = (weight_width - 1) * dilation_w + 1
    if stride_h < effective_kernel_h or stride_w < effective_kernel_w:
        return False

    input_channels_per_group = input_channels // groups
    if input_channels_per_group > 128 or output_channels_per_group > 128:
        return False
    return weight_height * weight_width <= 25


def _can_use_stride2_pad1_3x3_direct(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    if bias is not None or groups != 1:
        return False
    if (dilation_h, dilation_w) != (1, 1):
        return False
    if (output_padding_h, output_padding_w) != (0, 0):
        return False
    if (stride_h, stride_w) != (2, 2) or (padding_h, padding_w) != (1, 1):
        return False
    if input.dim() != 4 or weight.dim() != 4:
        return False
    if input.device.type != "cuda" or weight.device != input.device:
        return False
    if input.dtype not in _GENERAL_TRITON_DTYPES or weight.dtype != input.dtype:
        return False
    if input.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return False
    if not input.is_contiguous() or not weight.is_contiguous():
        return False

    batch, input_channels, input_height, input_width = input.shape
    weight_input_channels, output_channels, weight_height, weight_width = weight.shape
    if batch <= 0 or input_height <= 0 or input_width <= 0:
        return False
    if input_channels != weight_input_channels:
        return False
    if (weight_height, weight_width) != (3, 3):
        return False
    if input_channels < 16 or output_channels < 16:
        return False
    if (
        input_channels > _DIRECT_STRIDE2_PAD1_3X3_MAX_CHANNELS
        or output_channels > _DIRECT_STRIDE2_PAD1_3X3_MAX_CHANNELS
    ):
        return False
    if input.dtype is torch.float32:
        return True
    if (
        input.dtype is torch.float16
        and input_channels <= 32
        and output_channels >= 64
        and input_height <= 16
    ):
        return True
    return (
        input.dtype is torch.bfloat16
        and input_channels >= 64
        and output_channels <= 32
        and input_height <= 16
    )


@libentry()
@triton.jit
def _conv_transpose2d_direct_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_ci_stride: tl.constexpr,
    weight_co_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    input_channels: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_subgrid = tl.program_id(2)

    output_residue_h = pid_subgrid // stride_width
    output_residue_w = pid_subgrid % stride_width
    compact_height: tl.constexpr = (output_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_plane: tl.constexpr = compact_height * compact_width
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * stride_height + output_residue_h
    ow = compact_w * stride_width + output_residue_w
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels, BLOCK_CI)
    height_residue = (output_residue_h + padding_height) % stride_height
    width_residue = (output_residue_w + padding_width) % stride_width
    for kh in range(weight_height):
        if kh % stride_height == height_residue:
            ih_unstrided = oh + padding_height - kh
            ih = ih_unstrided // stride_height
            valid_h = (ih_unstrided >= 0) & (ih < input_height)
            for kw in range(weight_width):
                if kw % stride_width == width_residue:
                    iw_unstrided = ow + padding_width - kw
                    iw = iw_unstrided // stride_width
                    valid_hw = (
                        (n < batch_size)
                        & valid_h
                        & (iw_unstrided >= 0)
                        & (iw < input_width)
                        & (oh < output_height)
                        & (ow < output_width)
                    )
                    for ci_base in range(ci_blocks):
                        ci_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        input_offsets = (
                            n[:, None] * input_n_stride
                            + ci_offsets[None, :] * input_c_stride
                            + ih[:, None] * input_height_stride
                            + iw[:, None] * input_width_stride
                        )
                        weight_offsets = (
                            ci_offsets[:, None] * weight_ci_stride
                            + co_offsets[None, :] * weight_co_stride
                            + kh * weight_height_stride
                            + kw * weight_width_stride
                        )
                        input_mask = valid_hw[:, None] & (
                            ci_offsets[None, :] < input_channels
                        )
                        weight_mask = (ci_offsets[:, None] < input_channels) & (
                            co_offsets[None, :] < output_channels
                        )
                        input_block = tl.load(
                            input_pointer + input_offsets, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = (
        n[:, None] * output_n_stride
        + co_offsets[None, :] * output_c_stride
        + oh[:, None] * output_height_stride
        + ow[:, None] * output_width_stride
    )
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_stride2_pad1_3x3_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    compact_height: tl.constexpr,
    compact_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_ci_stride: tl.constexpr,
    weight_co_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    input_channels: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_raw = tl.program_id(0)
    phase = pid_raw % 4
    pid_nhw = pid_raw // 4
    pid_co = tl.program_id(1)

    residue_h = phase // 2
    residue_w = phase % 2
    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_plane: tl.constexpr = compact_height * compact_width
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * 2 + residue_h
    ow = compact_w * 2 + residue_w
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels, BLOCK_CI)
    height_residue = (residue_h + 1) % 2
    width_residue = (residue_w + 1) % 2
    for kh_slot in range(2):
        kh = height_residue + kh_slot * 2
        valid_kh = kh < 3
        ih_unstrided = oh + 1 - kh
        ih = ih_unstrided // 2
        valid_h = valid_kh & (ih_unstrided >= 0) & (ih < input_height)
        for kw_slot in range(2):
            kw = width_residue + kw_slot * 2
            valid_kw = kw < 3
            iw_unstrided = ow + 1 - kw
            iw = iw_unstrided // 2
            valid_hw = (
                (n < batch_size)
                & valid_h
                & valid_kw
                & (iw_unstrided >= 0)
                & (iw < input_width)
                & (oh < output_height)
                & (ow < output_width)
            )
            for ci_base in range(ci_blocks):
                ci_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                input_offsets = (
                    n[:, None] * input_n_stride
                    + ci_offsets[None, :] * input_c_stride
                    + ih[:, None] * input_height_stride
                    + iw[:, None] * input_width_stride
                )
                weight_offsets = (
                    ci_offsets[:, None] * weight_ci_stride
                    + co_offsets[None, :] * weight_co_stride
                    + kh * weight_height_stride
                    + kw * weight_width_stride
                )
                input_mask = valid_hw[:, None] & (ci_offsets[None, :] < input_channels)
                weight_mask = (
                    (ci_offsets[:, None] < input_channels)
                    & (co_offsets[None, :] < output_channels)
                    & valid_kh
                    & valid_kw
                )
                input_block = tl.load(
                    input_pointer + input_offsets, mask=input_mask, other=0.0
                )
                weight_block = tl.load(
                    weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                )
                accum += tl.dot(
                    input_block,
                    weight_block,
                    input_precision="tf32x3",
                )

    output_offsets = (
        n[:, None] * output_n_stride
        + co_offsets[None, :] * output_c_stride
        + oh[:, None] * output_height_stride
        + ow[:, None] * output_width_stride
    )
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_residue_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    n_subgrids: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co_in_group = tl.program_id(1)
    pid_phase_group = tl.program_id(2)

    pid_subgrid = pid_phase_group % n_subgrids
    group = pid_phase_group // n_subgrids
    output_residue_h = pid_subgrid // stride_width
    output_residue_w = pid_subgrid % stride_width
    compact_height: tl.constexpr = (output_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width
    compact_plane: tl.constexpr = compact_height * compact_width

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * stride_height + output_residue_h
    ow = compact_w * stride_width + output_residue_w

    co_in_offsets = pid_co_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * output_channels_per_group + co_in_offsets

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    height_residue = (output_residue_h + padding_height) % stride_height
    width_residue = (output_residue_w + padding_width) % stride_width
    for kh in range(weight_height):
        kh_residue: tl.constexpr = (kh * dilation_height) % stride_height
        if kh_residue == height_residue:
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (n < batch_size) & (ih_unstrided >= 0) & (ih < input_height)
            for kw in range(weight_width):
                kw_residue: tl.constexpr = (kw * dilation_width) % stride_width
                if kw_residue == width_residue:
                    iw_unstrided = ow + padding_width - kw * dilation_width
                    iw = iw_unstrided // stride_width
                    valid_hw = (
                        valid_h
                        & (iw_unstrided >= 0)
                        & (iw < input_width)
                        & (oh < output_height)
                        & (ow < output_width)
                    )
                    for ci_base in range(ci_blocks):
                        ci_in_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        ci_offsets = group * input_channels_per_group + ci_in_offsets
                        input_offsets = (
                            n[:, None] * input_channels + ci_offsets[None, :]
                        ) * input_height
                        input_offsets = (
                            input_offsets + ih[:, None]
                        ) * input_width + iw[:, None]
                        weight_offsets = (
                            ci_offsets[:, None] * output_channels_per_group
                            + co_in_offsets[None, :]
                        ) * weight_height
                        weight_offsets = (weight_offsets + kh) * weight_width + kw
                        input_mask = valid_hw[:, None] & (
                            ci_in_offsets[None, :] < input_channels_per_group
                        )
                        weight_mask = (
                            ci_in_offsets[:, None] < input_channels_per_group
                        ) & (co_in_offsets[None, :] < output_channels_per_group)
                        input_block = tl.load(
                            input_pointer + input_offsets, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = n[:, None] * output_channels + co_offsets[None, :]
    output_offsets = (output_offsets * output_height + oh[:, None]) * output_width
    output_offsets = output_offsets + ow[:, None]
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_in_offsets[None, :] < output_channels_per_group)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_general_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    total_elements: tl.constexpr,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    tmp = offsets // output_width
    ow = offsets - tmp * output_width
    tmp2 = tmp // output_height
    oh = tmp - tmp2 * output_height
    n = tmp2 // output_channels
    co = tmp2 - n * output_channels

    group = co // output_channels_per_group
    co_in_group = co - group * output_channels_per_group
    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if has_bias:
        bias = tl.load(bias_pointer + co, mask=mask, other=0.0).to(tl.float32)
        accum += bias

    for ci_in_group in tl.range(0, input_channels_per_group):
        ci = group * input_channels_per_group + ci_in_group
        for kh in tl.static_range(0, weight_height):
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (ih_unstrided % stride_height == 0) & (ih >= 0)
            valid_h = valid_h & (ih < input_height)
            for kw in tl.static_range(0, weight_width):
                iw_unstrided = ow + padding_width - kw * dilation_width
                iw = iw_unstrided // stride_width
                valid = mask & valid_h
                valid = valid & (iw_unstrided % stride_width == 0)
                valid = valid & (iw >= 0) & (iw < input_width)

                input_offsets = (n * input_channels + ci) * input_height + ih
                input_offsets = input_offsets * input_width + iw
                weight_offsets = (
                    ci * output_channels_per_group + co_in_group
                ) * weight_height
                weight_offsets = (weight_offsets + kh) * weight_width + kw
                input_values = tl.load(
                    input_pointer + input_offsets, mask=valid, other=0.0
                ).to(tl.float32)
                weight_values = tl.load(
                    weight_pointer + weight_offsets, mask=valid, other=0.0
                ).to(tl.float32)
                accum += input_values * weight_values

    tl.store(output_pointer + offsets, accum, mask=mask)


@libentry()
@triton.jit
def _conv_transpose2d_residue_static_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    compact_height: tl.constexpr,
    compact_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    output_residue_h: tl.constexpr,
    output_residue_w: tl.constexpr,
    co_blocks_per_group: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_gco = tl.program_id(1)

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_plane: tl.constexpr = compact_height * compact_width
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane
    oh = compact_h * stride_height + output_residue_h
    ow = compact_w * stride_width + output_residue_w

    group = pid_gco // co_blocks_per_group
    pid_co_in_group = pid_gco - group * co_blocks_per_group
    co_in_offsets = pid_co_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * output_channels_per_group + co_in_offsets

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    height_residue: tl.constexpr = (output_residue_h + padding_height) % stride_height
    width_residue: tl.constexpr = (output_residue_w + padding_width) % stride_width
    for kh in tl.static_range(0, weight_height):
        if (kh * dilation_height) % stride_height == height_residue:
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (n < batch_size) & (ih_unstrided >= 0) & (ih < input_height)
            for kw in tl.static_range(0, weight_width):
                if (kw * dilation_width) % stride_width == width_residue:
                    iw_unstrided = ow + padding_width - kw * dilation_width
                    iw = iw_unstrided // stride_width
                    valid_hw = (
                        valid_h
                        & (iw_unstrided >= 0)
                        & (iw < input_width)
                        & (oh < output_height)
                        & (ow < output_width)
                    )
                    for ci_base in range(ci_blocks):
                        ci_in_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        ci_offsets = group * input_channels_per_group + ci_in_offsets
                        input_offsets = (
                            n[:, None] * input_channels + ci_offsets[None, :]
                        ) * input_height
                        input_offsets = (
                            input_offsets + ih[:, None]
                        ) * input_width + iw[:, None]
                        weight_offsets = (
                            ci_offsets[:, None] * output_channels_per_group
                            + co_in_offsets[None, :]
                        ) * weight_height
                        weight_offsets = (weight_offsets + kh) * weight_width + kw
                        input_mask = valid_hw[:, None] & (
                            ci_in_offsets[None, :] < input_channels_per_group
                        )
                        weight_mask = (
                            ci_in_offsets[:, None] < input_channels_per_group
                        ) & (co_in_offsets[None, :] < output_channels_per_group)
                        input_block = tl.load(
                            input_pointer + input_offsets, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision="tf32x3",
                        )

    output_offsets = n[:, None] * output_channels + co_offsets[None, :]
    output_offsets = (output_offsets * output_height + oh[:, None]) * output_width
    output_offsets = output_offsets + ow[:, None]
    output_mask = (
        (n[:, None] < batch_size)
        & (oh[:, None] < output_height)
        & (ow[:, None] < output_width)
        & (co_in_offsets[None, :] < output_channels_per_group)
        & (co_offsets[None, :] < output_channels)
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_scatter_init_kernel(
    bias_pointer,
    output_pointer,
    total_elements: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if has_bias:
        spatial_size: tl.constexpr = output_height * output_width
        co = (offsets // spatial_size) % output_channels
        values = tl.load(bias_pointer + co, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_pointer + offsets, values, mask=mask)


@libentry()
@triton.jit
def _conv_transpose2d_scatter_no_overlap_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_gkk = tl.program_id(2)

    kw = pid_gkk % weight_width
    tmp = pid_gkk // weight_width
    kh = tmp % weight_height
    group = tmp // weight_height

    nhw_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    iw = nhw_offsets % input_width
    tmp = nhw_offsets // input_width
    ih = tmp % input_height
    n = tmp // input_height

    oh = ih * stride_height - padding_height + kh * dilation_height
    ow = iw * stride_width - padding_width + kw * dilation_width
    valid_nhw = (nhw_offsets < batch_size * input_height * input_width) & (
        n < batch_size
    )
    valid_nhw = valid_nhw & (oh >= 0) & (oh < output_height)
    valid_nhw = valid_nhw & (ow >= 0) & (ow < output_width)

    co_in_group = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co = group * output_channels_per_group + co_in_group
    ci_in_group_base = tl.arange(0, BLOCK_CI)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    for ci_block in range(ci_blocks):
        ci_in_group = ci_block * BLOCK_CI + ci_in_group_base
        ci = group * input_channels_per_group + ci_in_group
        input_offsets = (n[:, None] * input_channels + ci[None, :]) * input_height
        input_offsets = (input_offsets + ih[:, None]) * input_width + iw[:, None]
        weight_offsets = (
            ci[:, None] * output_channels_per_group + co_in_group[None, :]
        ) * weight_height
        weight_offsets = (weight_offsets + kh) * weight_width + kw

        ci_mask = ci_in_group < input_channels_per_group
        co_mask = co_in_group < output_channels_per_group
        input_block = tl.load(
            input_pointer + input_offsets,
            mask=valid_nhw[:, None] & ci_mask[None, :],
            other=0.0,
        )
        weight_block = tl.load(
            weight_pointer + weight_offsets,
            mask=ci_mask[:, None] & co_mask[None, :],
            other=0.0,
        )
        accum += tl.dot(
            input_block,
            weight_block,
            input_precision="tf32x3",
        )

    if has_bias:
        bias = tl.load(
            bias_pointer + co,
            mask=co_in_group < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias[None, :]

    output_offsets = (n[:, None] * output_channels + co[None, :]) * output_height
    output_offsets = (output_offsets + oh[:, None]) * output_width + ow[:, None]
    output_mask = valid_nhw[:, None] & (
        co_in_group[None, :] < output_channels_per_group
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_1x1_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    has_bias: tl.constexpr,
    co_blocks_per_group: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_gco = tl.program_id(1)

    group = pid_gco // co_blocks_per_group
    pid_co_in_group = pid_gco - group * co_blocks_per_group
    co_in_offsets = pid_co_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = group * output_channels_per_group + co_in_offsets

    nhw_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    iw = nhw_offsets % input_width
    tmp = nhw_offsets // input_width
    ih = tmp % input_height
    n = tmp // input_height
    valid_nhw = (nhw_offsets < batch_size * input_height * input_width) & (
        n < batch_size
    )

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    if has_bias:
        bias_values = tl.load(
            bias_pointer + co_offsets,
            mask=co_in_offsets < output_channels_per_group,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    for ci_base in range(ci_blocks):
        ci_in_offsets = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
        ci_offsets = group * input_channels_per_group + ci_in_offsets
        input_offsets = n[:, None] * input_channels + ci_offsets[None, :]
        input_offsets = (input_offsets * input_height + ih[:, None]) * input_width
        input_offsets = input_offsets + iw[:, None]
        weight_offsets = (
            ci_offsets[:, None] * output_channels_per_group + co_in_offsets[None, :]
        )
        ci_mask = ci_in_offsets < input_channels_per_group
        co_mask = co_in_offsets < output_channels_per_group
        input_block = tl.load(
            input_pointer + input_offsets,
            mask=valid_nhw[:, None] & ci_mask[None, :],
            other=0.0,
        )
        weight_block = tl.load(
            weight_pointer + weight_offsets,
            mask=ci_mask[:, None] & co_mask[None, :],
            other=0.0,
        )
        accum += tl.dot(input_block, weight_block, input_precision="tf32x3")

    output_offsets = n[:, None] * output_channels + co_offsets[None, :]
    output_offsets = (output_offsets * input_height + ih[:, None]) * input_width
    output_offsets = output_offsets + iw[:, None]
    output_mask = valid_nhw[:, None] & (
        co_in_offsets[None, :] < output_channels_per_group
    )
    tl.store(output_pointer + output_offsets, accum, mask=output_mask)


def _can_use_pointwise_1x1(
    weight,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
):
    return (
        weight.shape[2] == 1
        and weight.shape[3] == 1
        and stride_h == 1
        and stride_w == 1
        and padding_h == 0
        and padding_w == 0
        and output_padding_h == 0
        and output_padding_w == 0
    )


def _conv_transpose2d_pointwise_1x1(input, weight, bias, groups):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, _weight_height, _weight_width = weight.shape
    output_channels = output_channels_per_group * groups
    output = torch.empty(
        (batch, output_channels, input_height, input_width),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output

    input_channels_per_group = input_channels // groups
    block_nhw = 128 if input.dtype is not torch.float32 else 64
    block_ci = 16 if input.dtype is torch.float32 else 32
    if input_channels_per_group <= 16:
        block_ci = 16
    block_co = 16 if output_channels_per_group <= 16 else 32
    co_blocks_per_group = triton.cdiv(output_channels_per_group, block_co)
    grid = (
        triton.cdiv(batch * input_height * input_width, block_nhw),
        groups * co_blocks_per_group,
    )
    bias_pointer = bias if bias is not None else input
    _conv_transpose2d_1x1_kernel[grid](
        input,
        weight,
        bias_pointer,
        output,
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_channels_per_group,
        input_channels_per_group,
        bias is not None,
        co_blocks_per_group,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=4,
    )
    return output


def _conv_transpose2d_scatter_no_overlap(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, weight_height, weight_width = weight.shape
    output_channels = output_channels_per_group * groups
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (weight_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (weight_width - 1)
        + output_padding_w
        + 1
    )
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    total_elements = output.numel()
    if total_elements == 0:
        return output

    init_block = 1024
    bias_pointer = bias if bias is not None else input
    _conv_transpose2d_scatter_init_kernel[(triton.cdiv(total_elements, init_block),)](
        bias_pointer,
        output,
        total_elements,
        output_channels,
        output_height,
        output_width,
        bias is not None,
        BLOCK_SIZE=init_block,
        num_warps=4,
    )

    input_channels_per_group = input_channels // groups
    if input_channels_per_group <= 16:
        block_ci = 16
    elif input_channels_per_group <= 64:
        block_ci = 64 if input.dtype is not torch.float32 else 32
    else:
        block_ci = 64
    block_co = 16 if output_channels_per_group <= 16 else 32
    block_nhw = 32 if input.dtype is torch.float32 else 64
    if output_channels_per_group >= 64:
        block_nhw = 32

    input_nhw = batch * input_height * input_width
    grid = (
        triton.cdiv(input_nhw, block_nhw),
        triton.cdiv(output_channels_per_group, block_co),
        groups * weight_height * weight_width,
    )
    _conv_transpose2d_scatter_no_overlap_kernel[grid](
        input,
        weight,
        bias_pointer,
        output,
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        weight_height,
        weight_width,
        output_channels_per_group,
        input_channels_per_group,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        bias is not None,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=4,
        num_stages=3,
    )
    return output


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS CONV_TRANSPOSE2D")

    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    output_padding_h, output_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)

    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    output = _conv_transpose2d_4d_dispatch(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    )
    if input_was_unbatched:
        return output.squeeze(0)
    return output


def _conv_transpose2d_4d_dispatch(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    if _can_use_stride2_pad1_3x3_direct(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    ):
        return _conv_transpose2d_stride2_pad1_3x3(input, weight)

    direct_tiled_family_params = _direct_tiled_family_params(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    )
    if _can_use_direct_tiled_family(
        input, direct_tiled_family_params, output_padding_h
    ):
        return _conv_transpose2d_direct(
            input,
            weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_padding_h,
            output_padding_w,
        )

    if _validate_conv_transpose2d_args(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    ):
        if _can_use_pointwise_1x1(
            weight,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
        ):
            return _conv_transpose2d_pointwise_1x1(input, weight, bias, groups)
        if _can_use_scatter_no_overlap(
            input,
            weight,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            groups,
        ):
            return _conv_transpose2d_scatter_no_overlap(
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                output_padding_h,
                output_padding_w,
                groups,
            )
        return _conv_transpose2d_general(
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_padding_h,
            output_padding_w,
            groups,
        )

    return _unsupported_conv_transpose2d(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    )


def _select_stride2_pad1_3x3_schedule(input_dtype, input_channels, output_channels):
    block_nhw = 64
    block_ci = 32
    block_co = 32
    num_warps = 4

    if input_dtype is torch.float32:
        block_ci = 16
        block_co = 16
    elif input_channels <= 32 and output_channels >= 64:
        block_nhw = 128
        block_ci = 16
        block_co = 64
    elif input_channels >= 64 and output_channels <= 32:
        block_nhw = 64
        block_ci = 32
        block_co = 32
        num_warps = 8
    elif input_dtype is torch.bfloat16 and input_channels >= 128:
        block_nhw = 128
        block_ci = 16
        block_co = 16
        num_warps = 8

    return block_nhw, block_ci, block_co, num_warps


def _conv_transpose2d_stride2_pad1_3x3(input, weight):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels, _weight_height, _weight_width = weight.shape
    output_height = input_height * 2 - 1
    output_width = input_width * 2 - 1
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output

    block_nhw, block_ci, block_co, num_warps = _select_stride2_pad1_3x3_schedule(
        input.dtype,
        input_channels,
        output_channels,
    )
    compact_height = (output_height + 1) // 2
    compact_width = (output_width + 1) // 2
    grid = (
        triton.cdiv(batch * compact_height * compact_width, block_nhw) * 4,
        triton.cdiv(output_channels, block_co),
    )
    _conv_transpose2d_stride2_pad1_3x3_kernel[grid](
        input,
        weight,
        output,
        batch,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        compact_height,
        compact_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        input_channels,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
    )
    return output


def _select_conv_transpose2d_direct_schedule(
    input_dtype,
    input_channels,
    output_channels,
    weight_height,
    weight_width,
    stride_h,
    output_padding_h,
):
    block_nhw, block_ci, block_co, num_warps = _DIRECT_TILED_DEFAULT_SCHEDULE

    if input_dtype is torch.bfloat16:
        if stride_h >= 3:
            block_nhw = 128
            block_ci = 16
            block_co = 16
            num_warps = 8
        elif input_channels >= 128:
            block_nhw = 256
            block_ci = 16
            block_co = 16
            num_warps = 8
        elif weight_height >= 5 or weight_width >= 5:
            block_nhw = 128
            block_ci = 16
        elif input_channels >= 64 and output_channels <= 32:
            block_ci = 64
            if stride_h == 1:
                num_warps = 8
    elif input_dtype is torch.float16:
        if stride_h >= 3:
            block_nhw = 128
            block_ci = 16
            block_co = 16
            num_warps = 8
        elif weight_height >= 5 or weight_width >= 5:
            block_nhw = 128
            block_ci = 16
        elif input_channels >= 64 and output_channels <= 32:
            block_ci = 64
            if stride_h == 1:
                num_warps = 8
    elif input_dtype is torch.float32 and (weight_height >= 5 or weight_width >= 5):
        block_ci = 16
    elif input_channels >= 64 and output_channels <= 32:
        block_ci = 64
        if stride_h == 1:
            num_warps = 8
    if (
        stride_h == 1
        and weight_height <= 3
        and weight_width <= 3
        and input_channels >= 64
        and output_channels <= 64
    ):
        block_nhw = 256
        block_ci = 16
        block_co = 32
        num_warps = 8
    elif (
        stride_h == 2
        and weight_height <= 3
        and weight_width <= 3
        and input_channels <= 32
        and output_channels >= 64
    ):
        block_nhw = 128
        block_ci = 16
        block_co = 64
        num_warps = 4
    elif (
        stride_h == 2
        and weight_height <= 3
        and weight_width <= 3
        and input_channels >= 64
        and output_channels <= 32
    ):
        block_nhw = 32
        block_ci = 16
        block_co = 32
        num_warps = 8
    if output_padding_h:
        block_nhw = min(block_nhw, 128)
        block_ci = min(block_ci, 32)

    return block_nhw, block_ci, block_co, num_warps


def _conv_transpose2d_direct(
    input,
    weight,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels, weight_height, weight_width = weight.shape
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (weight_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (weight_width - 1)
        + output_padding_w
        + 1
    )
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    compact_height = triton.cdiv(output_height, stride_h)
    compact_width = triton.cdiv(output_width, stride_w)
    max_sub_spatial = batch * compact_height * compact_width
    n_subgrids = stride_h * stride_w

    block_nhw, block_ci, block_co, num_warps = _select_conv_transpose2d_direct_schedule(
        input.dtype,
        input_channels,
        output_channels,
        weight_height,
        weight_width,
        stride_h,
        output_padding_h,
    )

    grid = (
        triton.cdiv(max_sub_spatial, block_nhw),
        triton.cdiv(output_channels, block_co),
        n_subgrids,
    )
    _conv_transpose2d_direct_kernel[grid](
        input,
        weight,
        output,
        batch,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        input_channels,
        weight_height,
        weight_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
    )
    return output


def _conv_transpose2d_general(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
):
    return _conv_transpose2d_residue(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_padding_h,
        output_padding_w,
        groups,
    )


def _conv_transpose2d_residue(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, weight_height, weight_width = weight.shape
    output_channels = output_channels_per_group * groups
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (weight_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (weight_width - 1)
        + output_padding_w
        + 1
    )
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    total_elements = output.numel()
    if total_elements == 0:
        return output

    input_channels_per_group = input_channels // groups
    if (
        input.dtype in _TRITON_DIRECT_LOWP_DTYPES
        and weight_height >= 5
        and weight_width >= 5
        and stride_h == 2
        and stride_w == 2
        and dilation_h == 1
        and dilation_w == 1
        and input_channels_per_group >= 64
        and output_channels_per_group <= 32
    ):
        block_nhw = 256
        block_ci = 16
        block_co = 32
        co_blocks_per_group = triton.cdiv(output_channels_per_group, block_co)
        bias_pointer = bias if bias is not None else input
        for residue_h in range(stride_h):
            compact_height = (output_height + stride_h - 1 - residue_h) // stride_h
            for residue_w in range(stride_w):
                compact_width = (output_width + stride_w - 1 - residue_w) // stride_w
                grid = (
                    triton.cdiv(batch * compact_height * compact_width, block_nhw),
                    groups * co_blocks_per_group,
                )
                _conv_transpose2d_residue_static_kernel[grid](
                    input,
                    weight,
                    bias_pointer,
                    output,
                    batch,
                    input_channels,
                    input_height,
                    input_width,
                    output_channels,
                    output_height,
                    output_width,
                    compact_height,
                    compact_width,
                    weight_height,
                    weight_width,
                    output_channels_per_group,
                    input_channels_per_group,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    dilation_h,
                    dilation_w,
                    bias is not None,
                    residue_h,
                    residue_w,
                    co_blocks_per_group,
                    BLOCK_NHW=block_nhw,
                    BLOCK_CI=block_ci,
                    BLOCK_CO=block_co,
                    num_warps=4,
                    num_stages=2,
                )
        return output

    block_nhw = 64
    block_ci = 32
    block_co = 32
    num_warps = 4
    if input.dtype is torch.float32:
        block_ci = 16
        block_co = 16
    elif input_channels_per_group <= 16:
        block_ci = 16
    if output_channels_per_group <= 16:
        block_co = 16
    if (
        weight_height >= 5
        and weight_width >= 5
        and stride_h == 2
        and stride_w == 2
        and input_channels_per_group >= 64
        and output_channels_per_group <= 32
    ):
        block_nhw = 128
        block_ci = 64 if input.dtype is not torch.float32 else 32
        block_co = 16
        num_warps = 8
    if stride_h * stride_w >= 4 and input.dtype is not torch.float32:
        block_nhw = 128
        num_warps = 8

    compact_height = triton.cdiv(output_height, stride_h)
    compact_width = triton.cdiv(output_width, stride_w)
    max_sub_spatial = batch * compact_height * compact_width
    n_subgrids = stride_h * stride_w
    co_blocks_per_group = triton.cdiv(output_channels_per_group, block_co)
    grid = (
        triton.cdiv(max_sub_spatial, block_nhw),
        co_blocks_per_group,
        groups * n_subgrids,
    )
    bias_pointer = bias if bias is not None else input
    _conv_transpose2d_residue_kernel[grid](
        input,
        weight,
        bias_pointer,
        output,
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        weight_height,
        weight_width,
        output_channels_per_group,
        input_channels // groups,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        bias is not None,
        n_subgrids,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
    )
    return output
