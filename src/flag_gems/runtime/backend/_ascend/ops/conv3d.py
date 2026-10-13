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
import operator

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .pad import constant_pad_nd

logger = logging.getLogger(__name__)

_CONV3D_MAX_GRID = 65535


@libentry()
@triton.jit
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
    TILES_M: tl.constexpr,
    TILES_CO: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    bias_stride: tl.constexpr,
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    for tile in range(
        tl.program_id(0), TILES_M * TILES_CO * groups, tl.num_programs(0)
    ):
        pid_ni_do_ho_wo = tile % TILES_M
        pid_co = (tile // TILES_M) % TILES_CO
        pid_group = tile // (TILES_M * TILES_CO)

        # caculate in_n out_depth out_height out_weight value in kernel
        ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(
            0, BLOCK_NI_DO_HO_WO
        )
        ni_do_ho_offset = ni_do_ho_wo_offset // out_width
        ni_do_offset = ni_do_ho_offset // out_height
        in_n_point_value = ni_do_offset // out_depth
        output_depth_point_value = ni_do_offset % out_depth
        output_height_point_value = ni_do_ho_offset % out_height
        output_width_point_value = ni_do_ho_wo_offset % out_width

        # Load the input and weight pointers. input and weight are of shape
        # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
        out_per_group_c = out_c // groups
        output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        tile_input_pointer = (
            input_pointer
            + (
                input_n_stride * in_n_point_value
                + input_c_stride * pid_group * weight_c
            )[:, None]
        )
        tile_weight_pointer = (
            weight_pointer
            + (
                weight_n_stride * output_c_offset
                + weight_n_stride * pid_group * out_per_group_c
            )[None, :]
        )

        accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
        BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
        for dhwc in range(weight_depth * weight_height * weight_width * BLOCK_CI_COUNT):
            c = (dhwc % BLOCK_CI_COUNT) * BLOCK_CI
            dhw = dhwc // BLOCK_CI_COUNT
            dh = dhw // weight_width
            d = dh // weight_height
            h = dh % weight_height
            w = dhw % weight_width

            input_c_offset = c + tl.arange(0, BLOCK_CI)
            input_depth_offset = (
                d * dilation_depth
                - padding_depth
                + stride_depth * output_depth_point_value
            )
            input_height_offset = (
                h * dilation_height
                - padding_height
                + stride_height * output_height_point_value
            )
            input_width_offset = (
                w * dilation_width
                - padding_width
                + stride_width * output_width_point_value
            )

            curr_input_pointer = (
                tile_input_pointer
                + (input_c_stride * input_c_offset)[None, :]
                + (input_depth_stride * input_depth_offset)[:, None]
                + (input_height_stride * input_height_offset)[:, None]
                + (input_width_stride * input_width_offset)[:, None]
            )
            curr_weight_pointer = (
                tile_weight_pointer
                + (weight_c_stride * input_c_offset)[:, None]
                + (weight_depth_stride * d)
                + (weight_height_stride * h)
                + (weight_width_stride * w)
            )

            input_mask = (
                (in_n_point_value < in_n)[:, None]
                & (input_c_offset < weight_c)[None, :]
                & (0 <= input_depth_offset)[:, None]
                & (input_depth_offset < input_depth)[:, None]
                & (0 <= input_height_offset)[:, None]
                & (input_height_offset < input_height)[:, None]
                & (0 <= input_width_offset)[:, None]
                & (input_width_offset < input_width)[:, None]
            )
            weight_mask = (input_c_offset < weight_c)[:, None] & (
                output_c_offset < out_per_group_c
            )[None, :]

            input_block = tl.load(curr_input_pointer, mask=input_mask, other=0)
            weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0)

            accum += tl.dot(input_block, weight_block, allow_tf32=False)
        if HAS_BIAS:
            bias_offset = (pid_group * out_per_group_c + output_c_offset) * bias_stride
            bias = tl.load(
                bias_pointer + bias_offset, output_c_offset < out_per_group_c, other=0
            )
            accum += bias[None, :].to(tl.float32)
        tile_output_pointer = output_pointer + (
            (output_n_stride * in_n_point_value)[:, None]
            + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[
                None, :
            ]
            + (output_depth_stride * output_depth_point_value)[:, None]
            + (output_height_stride * output_height_point_value)[:, None]
            + (output_width_stride * output_width_point_value)[:, None]
        )
        output_mask = (
            (in_n_point_value < in_n)[:, None]
            & (output_c_offset < out_per_group_c)[None, :]
            & (output_depth_point_value < out_depth)[:, None]
            & (output_height_point_value < out_height)[:, None]
            & (output_width_point_value < out_width)[:, None]
        )

        tl.store(tile_output_pointer, accum, mask=output_mask)


def _triple(value, name):
    values = value if isinstance(value, (tuple, list)) else (value,) * 3
    if len(values) != 3:
        raise RuntimeError(f"{name} must contain three values")
    return tuple(operator.index(v) for v in values)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS ASCEND CONV3D")
    for tensor, name in ((weight, "weight"), (bias, "bias")):
        if tensor is not None:
            if tensor.device != input.device:
                raise RuntimeError(f"Input and {name} must be on the same device")
            if tensor.dtype != input.dtype:
                raise RuntimeError(
                    f"Input type ({input.dtype}) and {name} type ({tensor.dtype}) should be the same"
                )
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"Ascend conv3d does not support dtype {input.dtype}")
    unbatched = input.ndim == 4
    if unbatched:
        input = input.unsqueeze(0)
    if input.ndim != 5 or weight.ndim != 5:
        raise RuntimeError("conv3d expects rank-4/rank-5 input and rank-5 weight")
    if min(weight.shape) <= 0:
        raise RuntimeError("conv3d weight dimensions must be nonempty")
    groups = operator.index(groups)
    if (
        groups <= 0
        or weight.shape[0] % groups
        or input.shape[1] != groups * weight.shape[1]
    ):
        raise RuntimeError("Invalid grouped conv3d channel dimensions")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise RuntimeError("conv3d bias must have one value per output channel")
    strides = _triple(stride, "stride")
    dilations = _triple(dilation, "dilation")
    if min(strides) <= 0 or min(dilations) <= 0:
        raise RuntimeError("conv3d stride and dilation must be positive")
    if isinstance(padding, str):
        if padding == "same":
            if strides != (1, 1, 1):
                raise RuntimeError("padding='same' requires stride=1")
            # Native SAME puts any odd extra element on the right. Padding
            # first also avoids the generic wrapper's reversed crop for even K.
            total = tuple(d * (k - 1) for d, k in zip(dilations, weight.shape[2:]))
            spatial_pad = tuple(
                v for p in reversed(total) for v in (p // 2, p - p // 2)
            )
            input = constant_pad_nd(input, spatial_pad)
            paddings = (0, 0, 0)
        elif padding == "valid":
            paddings = (0, 0, 0)
        else:
            raise RuntimeError(f"Invalid padding string {padding}")
    else:
        paddings = _triple(padding, "padding")
        if min(paddings) < 0:
            raise RuntimeError("conv3d padding must be nonnegative")
    n, _, depth, height, width = input.shape
    out_c, weight_c, kd, kh, kw = weight.shape
    # Cube's K tile is always 16. Explicit vendor padding handles the original
    # small-channel rank-5 case and grouped rank-6 case without aten.pad.
    if weight_c < 16:
        pad_c = 16 - weight_c
        if groups == 1:
            input = constant_pad_nd(input, (0, 0, 0, 0, 0, 0, 0, pad_c))
        else:
            input = input.reshape(n, groups, weight_c, depth, height, width)
            input = constant_pad_nd(input, (0, 0, 0, 0, 0, 0, 0, pad_c))
            input = input.reshape(n, groups * 16, depth, height, width)
        weight = constant_pad_nd(weight, (0, 0, 0, 0, 0, 0, 0, pad_c))
        weight_c = 16
    out_spatial = tuple(
        (s + 2 * p - d * (k - 1) - 1) // t + 1
        for s, p, d, k, t in zip(
            (depth, height, width), paddings, dilations, (kd, kh, kw), strides
        )
    )
    if min(out_spatial) <= 0:
        raise RuntimeError("conv3d kernel size exceeds padded input size")
    output = torch.empty(
        (n, out_c, *out_spatial), dtype=input.dtype, device=input.device
    )
    if output.numel():
        tiles_m = triton.cdiv(n * out_spatial[0] * out_spatial[1] * out_spatial[2], 32)
        tiles_co = triton.cdiv(out_c // groups, 32)
        grid = (min(tiles_m * tiles_co * groups, _CONV3D_MAX_GRID),)
        with torch_device_fn.device(input.device):
            conv3d_forward_kernel[grid](
                input,
                weight,
                output,
                input if bias is None else bias,
                n,
                depth,
                height,
                width,
                out_c,
                *out_spatial,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                weight_c,
                kd,
                kh,
                kw,
                *strides,
                *paddings,
                *dilations,
                groups=groups,
                TILES_M=tiles_m,
                TILES_CO=tiles_co,
                HAS_BIAS=bias is not None,
                bias_stride=0 if bias is None else bias.stride(0),
                BLOCK_NI_DO_HO_WO=32,
                BLOCK_CI=16,
                BLOCK_CO=32,
            )
    return output.squeeze(0) if unbatched else output
