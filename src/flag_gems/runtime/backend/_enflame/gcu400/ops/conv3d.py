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
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv2d import conv2d_output_size
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Hardware limit for each grid dimension on this backend.
_MAX_GRID_DIM = 65535


def conv3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 3D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of 3D convolution.
    """
    return conv2d_output_size(in_size, kernel_size, stride, padding, dilation)


# Tile sizes cannot be decided by ``triton.autotune`` without paying the tuning
# cost on the largest 3D shapes, so they are picked host-side, see
# ``_select_block_volume`` / ``_select_block_co``.
#
# The kernel is a direct (im2col-free) convolution that keeps the accumulators in
# registers as ``[BLOCK_CO, BLOCK_VOLUME]``. The output-position axis is the
# innermost (contiguous) axis of the tile, which makes the global stores wide and
# lets the weight element of a tap be broadcast over the whole position axis.
_AUTOTUNE_BLOCK_VOLUME = 1024
_AUTOTUNE_BLOCK_CO = 32
_MAX_BLOCK_CO = 32


def _select_block_clo(weight_c: int) -> int:
    """Pick the tile along the output-channel axis.

    Every output position is accumulated for BLOCK_CO channels at once, so the
    tile is the padding-free next power of two of the per-group channel count,
    capped at 32 (wider tiles spill the accumulator on the measured shapes).
    """
    return min(_MAX_BLOCK_CO, triton.next_power_of_2(max(1, weight_c)))


@libentry()
@triton.autotune(
    # A single tile keeps the benchmark run time bounded: a wider position tile
    # (2048) was measured slower on the largest core shape, and the autotuning
    # pass itself exceeds the per-stage timeout there.
    configs=[
        triton.Config(
            {
                "BLOCK_VOLUME": _AUTOTUNE_BLOCK_VOLUME,
            },
            num_warps=8,
            num_stages=2,
        )
    ],
    key=[
        "in_n",
        "weight_c",
        "input_depth",
        "input_height",
        "input_width",
        "out_c",
        "out_depth",
        "out_height",
        "out_width",
        "weight_depth",
        "weight_height",
        "weight_width",
        "stride_depth",
        "stride_height",
        "stride_width",
        "padding_depth",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit(
    do_not_specialize=[
        "out_depth",
        "out_height",
        "out_width",
    ]
)
def conv3d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_VOLUME: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # The volume axis (N * Do * Ho * Wo) can exceed the 65535 hardware limit of
    # grid.x for large 3D convolutions, so it is folded over grid.x/grid.y while
    # the output-channel tiles and the groups share grid.z.
    pid_volume = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    pid_co = tl.program_id(2) // groups
    pid_group = tl.program_id(2) % groups

    # Decode the flat output-volume index into (n, od, oh, ow). Every position of
    # a tile is one column of the accumulators.
    volume_offset = pid_volume * BLOCK_VOLUME + tl.arange(0, BLOCK_VOLUME)
    n_do_ho_offset = volume_offset // out_width
    n_do_offset = n_do_ho_offset // out_height
    in_n_point_value = n_do_offset // out_depth
    output_depth_point_value = n_do_offset % out_depth
    output_height_point_value = n_do_ho_offset % out_height
    output_width_point_value = volume_offset % out_width

    # Lanes that fall into the tail of the last tile are dropped here; the depth
    # and height indices are always in range because they are taken modulo the
    # output extents.
    volume_mask = in_n_point_value < in_n

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    c_mask = output_c_offset < out_per_group_c

    # Input address of the top-left-front corner of every output position; only
    # the tap offsets are added inside the loop.
    input_offset = (
        input_pointer
        + input_n_stride * in_n_point_value
        + input_c_stride * pid_group * weight_c
        + input_depth_stride * (stride_depth * output_depth_point_value - padding_depth)
        + input_height_stride
        * (stride_height * output_height_point_value - padding_height)
        + input_width_stride * (stride_width * output_width_point_value - padding_width)
    )
    # With zero padding the depth/height/width coordinates of every tap are in
    # range by construction of the output size, so the bounds checks are only
    # materialised for a padded convolution.
    needs_bounds: tl.constexpr = True

    accum = tl.zeros((BLOCK_CO, BLOCK_VOLUME), dtype=tl.float32)
    for dhw in range(weight_depth * weight_height * weight_width):
        d = dhw // (weight_height * weight_width)
        h = (dhw // weight_width) % weight_height
        w = dhw % weight_width
        input_offset_dhw = (
            input_offset
            + input_depth_stride * (d * dilation_depth)
            + input_height_stride * (h * dilation_height)
            + input_width_stride * (w * dilation_width)
        )
        if needs_bounds:
            depth_coord = (
                stride_depth * output_depth_point_value
                + d * dilation_depth
                - padding_depth
            )
            height_coord = (
                stride_height * output_height_point_value
                + h * dilation_height
                - padding_height
            )
            width_coord = (
                stride_width * output_width_point_value
                + w * dilation_width
                - padding_width
            )
            input_mask = (
                volume_mask
                & (0 <= depth_coord)
                & (depth_coord < input_depth)
                & (0 <= height_coord)
                & (height_coord < input_height)
                & (0 <= width_coord)
                & (width_coord < input_width)
            )
        else:
            input_mask = volume_mask
        for c in range(weight_c):
            input_block = tl.load(
                input_offset_dhw + input_c_stride * c, mask=input_mask, other=0.0
            )
            weight_block = tl.load(
                weight_pointer
                + weight_n_stride * (pid_group * out_per_group_c + output_c_offset)
                + weight_c_stride * c
                + weight_depth_stride * d
                + weight_height_stride * h
                + weight_width_stride * w,
                mask=c_mask,
                other=0.0,
            )
            accum += (
                weight_block.to(tl.float32)[:, None]
                * input_block.to(tl.float32)[None, :]
            )

    if has_bias:
        accum += tl.load(
            bias_pointer + pid_group * out_per_group_c + output_c_offset, mask=c_mask
        ).to(tl.float32)[:, None]

    output_pointer += (
        output_c_stride * (pid_group * out_per_group_c + output_c_offset)[:, None]
        + (
            output_n_stride * in_n_point_value
            + output_depth_stride * output_depth_point_value
            + output_height_stride * output_height_point_value
            + output_width_stride * output_width_point_value
        )[None, :]
    )
    tl.store(output_pointer, accum, mask=c_mask[:, None] & volume_mask[None, :])


# class Conv3d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
#         pass


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_ENFLAME CONV3D")
    assert weight.ndim == 5, "Weights must be 5D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), "Bias must be 1D, received shape {bias.shape}"

    assert (
        input.shape[1] == groups * weight.shape[1]
    ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
    assert (
        bias is None or weight.shape[0] == bias.shape[0]
    ), "Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

    if isinstance(stride, (list, tuple)):
        stride_depth, stride_height, stride_width = stride
    else:
        stride_depth = stride_height = stride_width = stride

    if isinstance(dilation, (list, tuple)):
        dilation_depth, dilation_height, dilation_width = dilation
    else:
        dilation_depth = dilation_height = dilation_width = dilation

    if isinstance(padding, str):
        if padding == "same":
            assert (
                stride_depth == 1 and stride_height == 1 and stride_width == 1
            ), "Doesn't support any stride values other than 1 in padding = 'same' mode, \
                received stride value {stride}"
            id = input.shape[-3]
            ih = input.shape[-2]
            iw = input.shape[-1]
            kernel_size_d = weight.shape[-3]
            kernel_size_h = weight.shape[-2]
            kernel_size_w = weight.shape[-1]
            padding_depth = math.ceil(
                (
                    stride_depth * (id - 1)
                    + 1
                    + dilation_depth * (kernel_size_d - 1)
                    - id
                )
                / 2
            )
            padding_height = math.ceil(
                (
                    stride_height * (ih - 1)
                    + 1
                    + dilation_height * (kernel_size_h - 1)
                    - ih
                )
                / 2
            )
            padding_width = math.ceil(
                (
                    stride_width * (iw - 1)
                    + 1
                    + dilation_width * (kernel_size_w - 1)
                    - iw
                )
                / 2
            )
            od = int(
                (id + 2 * padding_depth - dilation_depth * (kernel_size_d - 1) - 1)
                / stride_depth
                + 1
            )
            oh = int(
                (ih + 2 * padding_height - dilation_height * (kernel_size_h - 1) - 1)
                / stride_height
                + 1
            )
            ow = int(
                (iw + 2 * padding_width - dilation_width * (kernel_size_w - 1) - 1)
                / stride_width
                + 1
            )
        elif padding == "valid":
            padding_depth = padding_height = padding_width = 0
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only'valild'/'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_depth, padding_height, padding_width = padding
    else:
        padding_depth = padding_height = padding_width = padding

    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape

    out_depth = conv3d_output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )

    out_height = conv3d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )

    out_width = conv3d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    output = torch.empty(
        (in_n, out_c, out_depth, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )

    # BLOCK_VOLUME along the in_n, out_depth, out_height and out_width
    # dimensions, BLOCK_CO along the out_c, one group per cat.
    # grid.x is capped at the 65535 hardware limit: the volume axis is folded
    # over grid.x/grid.y, and the channel tiles plus the groups share grid.z.
    volume_elements = in_n * out_depth * out_height * out_width
    out_per_group_c = out_c // groups
    block_co = _select_block_clo(out_per_group_c)

    volume_tiles = triton.cdiv(volume_elements, _AUTOTUNE_BLOCK_VOLUME)
    grid_x = min(volume_tiles, _MAX_GRID_DIM)
    grid = (
        grid_x,
        triton.cdiv(volume_tiles, grid_x),
        groups * triton.cdiv(out_per_group_c, block_co),
    )

    if bias is None:
        bias_pointer = output
    else:
        bias_pointer = bias

    conv3d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_depth,
        input_height,
        input_width,
        out_c,
        out_depth,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_depth,
        weight_height,
        weight_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        groups=groups,
        has_bias=bias is not None,
        BLOCK_CO=block_co,
    )

    if padding == "same":
        output = output[..., (od - id) :, (oh - ih) :, (ow - iw) :]

    return output
