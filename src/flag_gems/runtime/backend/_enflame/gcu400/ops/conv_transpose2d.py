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

# tf32x3 is an fp32-emulation hint: low-precision inputs never needed it, and
# on this backend it miscompiles fp16/bf16 dots into inf/garbage.  Measured on
# a 8x64x32x32 fp32 conv the two precisions are indistinguishable (5.789 ms vs
# 5.790 ms), so keep plain ieee here.
_DOT_PRECISION = tl.constexpr("ieee")

_TRITON_DIRECT_LOWP_DTYPES = (torch.float16, torch.bfloat16)

# Upcasting the 16-bit operands to fp32 before tl.dot turns every dot into an
# fp32 MMA and doubles the DSM staging footprint.  tl.dot already accumulates
# bf16/fp16 inputs in fp32, so the upcast buys precision nothing and costs
# throughput; it is kept switchable so the two forms can be A/B measured.
_TILED_UPCAST_LOWP = True

_GENERAL_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_DIRECT_TILED_FAMILY_MAX_CHANNELS = 256
_DIRECT_TILED_FAMILY_MAX_KERNEL = 5
_DIRECT_TILED_FAMILY_MAX_STRIDE = 4
_DIRECT_TILED_OUTPUT_PADDING_MIN_INPUT_ELEMENTS = 1024


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
    if input.device.type not in ("cuda", "gcu") or weight.device != input.device:
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
    if input.device.type not in ("cuda", "gcu") or weight.device != input.device:
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


@libentry()
@triton.jit
def _conv_transpose2d_tiled_dot(
    input_pointer,
    weight_pointer,
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
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    upcast_fp32: tl.constexpr,
    ci_blocks: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # Residue-phase tiled dot for the direct family (groups=1, bias=None,
    # dilation=1, contiguous): output positions are partitioned by their
    # (oh % sh, ow % sw) phase so each program only visits the taps whose
    # back-projection lands on an input sample.  Taps are unrolled with
    # static_range (a runtime loop around tl.dot costs ~44us per iteration
    # on this backend), and the back-projection is folded into one scalar
    # offset per tap so the tap body needs no vector division.
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_subgrid = tl.program_id(2)
    residue_h = pid_subgrid // stride_width
    residue_w = pid_subgrid % stride_width

    compact_height: tl.constexpr = (output_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width
    compact_plane: tl.constexpr = compact_height * compact_width

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    valid_co = co < output_channels

    acc = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)

    height_residue = (residue_h + padding_height) % stride_height
    width_residue = (residue_w + padding_width) % stride_width
    height_limit = (output_height - residue_h + stride_height - 1) // stride_height
    width_limit = (output_width - residue_w + stride_width - 1) // stride_width
    base_valid = (
        (n < batch_size) & (compact_h < height_limit) & (compact_w < width_limit)
    )

    for kh in tl.static_range(0, weight_height):
        if kh % stride_height == height_residue:
            ih_offset = (residue_h + padding_height - kh) // stride_height
            valid_h = (
                base_valid
                & (compact_h >= -ih_offset)
                & (compact_h < input_height - ih_offset)
            )
            ih = compact_h + ih_offset
            for kw in tl.static_range(0, weight_width):
                if kw % stride_width == width_residue:
                    iw_offset = (residue_w + padding_width - kw) // stride_width
                    valid = (
                        valid_h
                        & (compact_w >= -iw_offset)
                        & (compact_w < input_width - iw_offset)
                    )
                    iw = compact_w + iw_offset
                    # ci loop stays a runtime loop on purpose: a fully
                    # unrolled tap x ci chain allocates one DSM staging
                    # buffer per dot instance and overflows the 917504B
                    # budget (measured on CI=256), while the runtime loop
                    # reuses a single buffer across iterations.
                    for ci_base in range(ci_blocks):
                        ci = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        input_offsets = (
                            (n[:, None] * input_channels + ci[None, :]) * input_height
                            + ih[:, None]
                        ) * input_width + iw[:, None]
                        weight_offsets = (
                            (ci[:, None] * output_channels + co[None, :])
                            * weight_height
                            + kh
                        ) * weight_width + kw
                        input_mask = valid[:, None] & (ci[None, :] < input_channels)
                        weight_mask = (ci[:, None] < input_channels) & valid_co[None, :]
                        input_block = tl.load(
                            input_pointer + input_offsets,
                            mask=input_mask,
                            other=0.0,
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets,
                            mask=weight_mask,
                            other=0.0,
                        )
                        if upcast_fp32:
                            acc += tl.dot(
                                input_block.to(tl.float32),
                                weight_block.to(tl.float32),
                                input_precision=_DOT_PRECISION,
                            )
                        else:
                            acc += tl.dot(
                                input_block,
                                weight_block,
                                input_precision=_DOT_PRECISION,
                            )

    oh = compact_h * stride_height + residue_h
    ow = compact_w * stride_width + residue_w
    output_offsets = (n[:, None] * output_channels + co[None, :]) * (
        output_height * output_width
    )
    output_offsets = output_offsets + oh[:, None] * output_width + ow[:, None]
    output_mask = base_valid[:, None] & valid_co[None, :]
    tl.store(
        output_pointer + output_offsets,
        acc.to(output_pointer.dtype.element_ty),
        mask=output_mask,
    )


@libentry()
@triton.jit
def _conv_transpose2d_tiled_dot_transposed(
    input_pointer,
    weight_pointer,
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
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    upcast_fp32: tl.constexpr,
    ci_blocks: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # Same residue-phase tiling as _conv_transpose2d_tiled_dot, but the dot is
    # computed transposed: acc[co, nhw] = dot(W[co, ci], X[ci, nhw]).
    #
    # NCHW memory makes the channel axis a strided axis (stride H*W) while the
    # compact spatial axis is unit-stride.  Owning BLOCK_NHW in the *fast* axis
    # of both the loaded tile and the accumulator turns the input tile into a
    # coalesced (BLOCK_CI, BLOCK_NHW) read instead of a (BLOCK_NHW, BLOCK_CI)
    # strided gather, which is what the non-transposed variant pays for on every
    # tap.  Measured on the benchmark matrix this is 1.2x-2.7x faster on
    # fp16/bf16 and is never slower than the non-transposed variant on fp32.
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_subgrid = tl.program_id(2)
    residue_h = pid_subgrid // stride_width
    residue_w = pid_subgrid % stride_width

    compact_height: tl.constexpr = (output_height + stride_height - 1) // stride_height
    compact_width: tl.constexpr = (output_width + stride_width - 1) // stride_width
    compact_plane: tl.constexpr = compact_height * compact_width

    compact_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_nh = compact_offsets // compact_width
    compact_h = compact_nh % compact_height
    compact_w = compact_offsets % compact_width
    n = compact_offsets // compact_plane

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    valid_co = co < output_channels

    acc = tl.zeros((BLOCK_CO, BLOCK_NHW), dtype=tl.float32)

    height_residue = (residue_h + padding_height) % stride_height
    width_residue = (residue_w + padding_width) % stride_width
    height_limit = (output_height - residue_h + stride_height - 1) // stride_height
    width_limit = (output_width - residue_w + stride_width - 1) // stride_width
    base_valid = (
        (n < batch_size) & (compact_h < height_limit) & (compact_w < width_limit)
    )

    for kh in tl.static_range(0, weight_height):
        if kh % stride_height == height_residue:
            ih_offset = (residue_h + padding_height - kh) // stride_height
            valid_h = (
                base_valid
                & (compact_h >= -ih_offset)
                & (compact_h < input_height - ih_offset)
            )
            ih = compact_h + ih_offset
            for kw in tl.static_range(0, weight_width):
                if kw % stride_width == width_residue:
                    iw_offset = (residue_w + padding_width - kw) // stride_width
                    valid = (
                        valid_h
                        & (compact_w >= -iw_offset)
                        & (compact_w < input_width - iw_offset)
                    )
                    iw = compact_w + iw_offset
                    for ci_base in range(ci_blocks):
                        ci = ci_base * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        input_offsets = (
                            (n[None, :] * input_channels + ci[:, None]) * input_height
                            + ih[None, :]
                        ) * input_width + iw[None, :]
                        weight_offsets = (
                            (ci[None, :] * output_channels + co[:, None])
                            * weight_height
                            + kh
                        ) * weight_width + kw
                        input_mask = valid[None, :] & (ci[:, None] < input_channels)
                        weight_mask = (ci[None, :] < input_channels) & valid_co[:, None]
                        input_block = tl.load(
                            input_pointer + input_offsets,
                            mask=input_mask,
                            other=0.0,
                        )
                        weight_block = tl.load(
                            weight_pointer + weight_offsets,
                            mask=weight_mask,
                            other=0.0,
                        )
                        if upcast_fp32:
                            acc += tl.dot(
                                weight_block.to(tl.float32),
                                input_block.to(tl.float32),
                                input_precision=_DOT_PRECISION,
                            )
                        else:
                            acc += tl.dot(
                                weight_block,
                                input_block,
                                input_precision=_DOT_PRECISION,
                            )

    oh = compact_h * stride_height + residue_h
    ow = compact_w * stride_width + residue_w
    output_offsets = (n[None, :] * output_channels + co[:, None]) * (
        output_height * output_width
    )
    output_offsets = output_offsets + oh[None, :] * output_width + ow[None, :]
    output_mask = valid_co[:, None] & base_valid[None, :]
    tl.store(
        output_pointer + output_offsets,
        acc.to(output_pointer.dtype.element_ty),
        mask=output_mask,
    )


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
    for kh in tl.static_range(0, weight_height):
        kh_residue = (kh * dilation_height) % stride_height
        if kh_residue == height_residue:
            ih_unstrided = oh + padding_height - kh * dilation_height
            ih = ih_unstrided // stride_height
            valid_h = (n < batch_size) & (ih_unstrided >= 0) & (ih < input_height)
            for kw in tl.static_range(0, weight_width):
                kw_residue = (kw * dilation_width) % stride_width
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
                            input_precision=_DOT_PRECISION,
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
                            input_precision=_DOT_PRECISION,
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
            input_precision=_DOT_PRECISION,
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
        accum += tl.dot(input_block, weight_block, input_precision=_DOT_PRECISION)

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
    logger.debug("GEMS_ENFLAME CONV_TRANSPOSE2D")

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


def _next_pow2(value):
    out = 1
    while out < value:
        out *= 2
    return out


def _select_conv_transpose2d_tiled_schedule(
    input_dtype,
    input_channels,
    output_channels,
    weight_height,
    weight_width,
    stride_h,
):
    """Pick (BLOCK_NHW, BLOCK_CI, BLOCK_CO, num_warps) for the tiled dot kernel.

    Arms are measured on GCU400 over the benchmark shape matrix (12-case
    sweep x 3 dtypes plus domain-edge DSM probes); the sibling
    slow_conv_transpose2d measurements bootstrap the starting point.  The
    static_range tap unroll allocates one DSM staging set per tap, so
    heavy-tap kernels (>= 17 taps, e.g. 5x5) must use narrow (BLOCK_CI,
    BLOCK_CO) tiles to stay under the 917504B DSM limit (measured: lowp
    ~36KB/tap at (.,32,32); k25 at (64,16,64) fails at 1339392B while
    (128,16,16) compiles).  BLOCK_CI widening (ci_blocks <= 4) is only
    applied for light-tap kernels where the per-tap budget has room.
    """
    taps = weight_height * weight_width
    if input_dtype is torch.float32:
        if taps >= 17:
            block_nhw, block_ci, block_co, num_warps = 128, 16, 16, 4
        elif stride_h == 1:
            block_nhw, block_ci, block_co, num_warps = 32, 32, 32, 4
        elif output_channels >= 64:
            block_nhw, block_ci, block_co, num_warps = 64, 16, 64, 4
        else:
            block_nhw, block_ci, block_co, num_warps = 128, 16, 32, 4
    else:
        if taps >= 17:
            block_nhw, block_ci, block_co, num_warps = 128, 16, 16, 4
        elif stride_h == 1:
            # Re-measured 2026-11-22 on (1,64,32,32,co=32,3x3,s1,p1): the
            # 128-wide NHW tile with a 32-wide CI block beats the previous
            # (64,64,32) arm 0.164ms vs 0.264ms (1.6x), i.e. a wider NHW tile
            # pays off while the deep CI block only adds DSM staging.
            block_nhw, block_ci, block_co, num_warps = 128, 32, 32, 4
        elif output_channels >= 64:
            block_nhw, block_ci, block_co, num_warps = 128, 16, 64, 8
        else:
            block_nhw, block_ci, block_co, num_warps = 128, 16, 32, 4
    if taps <= 9:
        if input_channels > 128:
            # fp32 tiles pay ~4x the per-element DSM cost of the upcast
            # lowp path, so their widening stops at 32 (measured:
            # (128,64,32) at 9 taps needs 942080B and fails).
            block_ci = max(block_ci, 32 if input_dtype is torch.float32 else 64)
            if block_co > 32:
                # (BLOCK_CI=64, BLOCK_CO=64) at 9 taps needs 1056768B of
                # DSM (over the 917504B limit); 851968B is measured safe
                # with BLOCK_CO=32, so trade the channel-block tile down.
                block_co = 32
        elif input_channels > 64:
            block_ci = max(block_ci, 32)
    block_co = min(block_co, _next_pow2(output_channels))
    return block_nhw, block_ci, block_co, num_warps


_TILED_LAUNCH_CACHE = {}


def _select_conv_transpose2d_transposed_schedule(
    input_dtype,
    input_channels,
    output_channels,
    weight_height,
    weight_width,
    stride_h,
):
    """Pick (BLOCK_NHW, BLOCK_CI, BLOCK_CO, num_warps) for the transposed dot.

    A wide BLOCK_NHW is the whole point of this variant (it is the unit-stride
    axis of both the loaded tile and the accumulator); 128 keeps the input tile
    at 32*128 fp32 = 16KB, so a 3x3 transposed conv (9 taps) needs ~216KB of
    DSM staging in total, well under the 917504B budget.  BLOCK_CI=32 already
    covers the channel counts of the benchmark matrix in a single K step, and
    widening it only costs DSM.  BLOCK_CO follows the output channel count
    (measured: CO=64 wants BLOCK_CO=64, CO=32 wants BLOCK_CO=32).
    """
    block_nhw, block_ci, num_warps = 128, 32, 8
    output_channels_pow2 = _next_pow2(output_channels)
    block_co = 64 if output_channels_pow2 >= 64 else 32
    block_co = min(block_co, output_channels_pow2)
    return block_nhw, block_ci, block_co, num_warps


def _tiled_launch_hooks_active():
    """True when profiling/trace launch hooks are installed and must run."""
    for hook in (
        triton.knobs.runtime.launch_enter_hook,
        triton.knobs.runtime.launch_exit_hook,
    ):
        if hook is None:
            continue
        calls = getattr(hook, "calls", None)
        if calls is None or len(calls) > 0:
            return True
    return False


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
    if output.numel() == 0:
        return output

    # A 3x3 tap set (taps <= 9) fits the transposed-dot variant inside the DSM
    # budget; heavier tap sets keep the narrow-tile non-transposed kernel that
    # was tuned for them.  The wide BLOCK_NHW also needs enough output to fill
    # the machine: on tiny outputs (1x32x32x32, 32K elements) it measures 1.25x
    # slower on fp16 and 1.8x slower on fp32 than the narrow non-transposed
    # schedule, which puts 32 CTAs on the same work.
    use_transposed_dot = weight_height * weight_width <= 9 and output.numel() >= 65536
    if use_transposed_dot:
        (
            block_nhw,
            block_ci,
            block_co,
            num_warps,
        ) = _select_conv_transpose2d_transposed_schedule(
            input.dtype,
            input_channels,
            output_channels,
            weight_height,
            weight_width,
            stride_h,
        )
        tiled_dot_kernel = _conv_transpose2d_tiled_dot_transposed
    else:
        (
            block_nhw,
            block_ci,
            block_co,
            num_warps,
        ) = _select_conv_transpose2d_tiled_schedule(
            input.dtype,
            input_channels,
            output_channels,
            weight_height,
            weight_width,
            stride_h,
        )
        tiled_dot_kernel = _conv_transpose2d_tiled_dot
    # Degenerate spatial extents trip an illegal shared-memory access inside
    # tl.dot at low warp counts on this backend; eight warps avoids it.
    if (output_height <= 1 or output_width <= 1) and num_warps < 8:
        num_warps = 8
    compact_height = triton.cdiv(output_height, stride_h)
    compact_width = triton.cdiv(output_width, stride_w)
    max_sub_spatial = batch * compact_height * compact_width
    grid = (
        triton.cdiv(max_sub_spatial, block_nhw),
        triton.cdiv(output_channels, block_co),
        stride_h * stride_w,
    )

    # Every parameter of this kernel is a constexpr, so a compiled kernel is
    # fully determined by the (device, dtype, geometry, tile) tuple below and
    # only the three tensor pointers are runtime arguments.  Bypassing the
    # per-call JIT binding (measured at ~25us of the ~43us host cost) is the
    # difference between 0.4x and 1x against the native op on the small
    # benchmark shapes, so cache the compiled kernel and launch it directly
    # while no launch hook (profiling/trace tooling) is installed.  Pointer
    # alignment participates in Triton's specialization and is keyed too.
    fast_key = None
    if not _tiled_launch_hooks_active():
        fast_key = (
            input.get_device(),
            input.dtype,
            batch,
            input_channels,
            input_height,
            input_width,
            output_channels,
            output_height,
            output_width,
            weight_height,
            weight_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            block_nhw,
            block_ci,
            block_co,
            num_warps,
            use_transposed_dot,
            (input.data_ptr() | weight.data_ptr()) & 15,
        )
        entry = _TILED_LAUNCH_CACHE.get(fast_key)
        if entry is not None:
            kernel, grid0, grid1, grid2, static_tail = entry
            stream = torch.gcu.current_stream(fast_key[0]).gcu_stream
            kernel.run(
                grid0,
                grid1,
                grid2,
                stream,
                kernel.function,
                kernel.packed_metadata,
                None,
                None,
                None,
                input,
                weight,
                output,
                *static_tail,
            )
            return output
    ci_blocks = triton.cdiv(input_channels, block_ci)
    kernel, _ = tiled_dot_kernel[grid](
        input,
        weight,
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
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        upcast_fp32=_TILED_UPCAST_LOWP and input.dtype in _TRITON_DIRECT_LOWP_DTYPES,
        ci_blocks=ci_blocks,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=num_warps,
        num_stages=2,
    )
    if fast_key is not None:
        # Triton 3.6's GCU launcher still parses every constexpr argument
        # at each launch, so the direct path has to replay them; they are
        # ignored by the compiled binary.
        static_tail = (
            batch,
            input_channels,
            input_height,
            input_width,
            output_channels,
            output_height,
            output_width,
            weight_height,
            weight_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            input.dtype in _TRITON_DIRECT_LOWP_DTYPES,
            ci_blocks,
            block_nhw,
            block_ci,
            block_co,
        )
        _TILED_LAUNCH_CACHE[fast_key] = (
            kernel,
            grid[0],
            grid[1],
            grid[2],
            static_tail,
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
    # fp16/bf16 dots in this path intermittently return garbage on this
    # backend (the same shape gives different results across calls in one
    # process).  Compute in fp32 and round once at the end: same accuracy and
    # deterministic.
    out_dtype = input.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        input = input.float()
        weight = weight.float()
        if bias is not None:
            bias = bias.float()
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
        return output.to(out_dtype)

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
        return output.to(out_dtype)

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

    # Workaround for Triton compiler bug: when the output spatial size is
    # degenerate (e.g. output_width==1 from a 1-D convolution unsqueezed to
    # 2-D), BLOCK_NHW=64 with num_warps=4 triggers an illegal shared-memory
    # access inside tl.dot.  Bumping to 8 warps avoids the mis-compilation.
    if (output_height <= 1 or output_width <= 1) and num_warps < 8:
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
    return output.to(out_dtype)
