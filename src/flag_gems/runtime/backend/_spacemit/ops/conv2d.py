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

from flag_gems import runtime
from flag_gems.ops.conv2d import conv2d as _general_conv2d
from flag_gems.utils import libentry, libtuner

from .bmm import bmm_kernel

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conv2d_forward"),
    key=["in_n", "input_height", "input_width", "out_c", "weight_c", "groups"],
    disable_aabs=True,
)
@triton.jit
def conv2d_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
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
):
    tiles_spatial: tl.constexpr = triton.cdiv(
        in_n * out_height * out_width, BLOCK_NI_HO_WO
    )
    tiles_co: tl.constexpr = triton.cdiv(out_c // groups, BLOCK_CO)
    total_tiles: tl.constexpr = tiles_spatial * tiles_co * groups
    for task_block in range(triton.cdiv(total_tiles, 512)):
        task = task_block * 512 + tl.program_id(0)
        if total_tiles <= 512 or task < total_tiles:
            pid_ni_ho_wo = task // (tiles_co * groups)
            pid_co = task // groups % tiles_co
            pid_group = task % groups
            ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(
                0, BLOCK_NI_HO_WO
            )
            ni_ho_offset = ni_ho_wo_offset // out_width
            in_n_point_value = ni_ho_offset // out_height
            output_height_point_value = ni_ho_offset % out_height
            output_width_point_value = ni_ho_wo_offset % out_width

            # Load the input and weight pointers. input and weight are of shape
            # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
            out_per_group_c = out_c // groups
            output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
            input_tile_pointer = (
                input_pointer
                + (
                    input_n_stride * in_n_point_value
                    + input_c_stride * pid_group * weight_c
                )[:, None]
            )
            weight_tile_pointer = (
                weight_pointer
                + (
                    weight_n_stride * output_c_offset
                    + weight_n_stride * pid_group * out_per_group_c
                )[None, :]
            )

            accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
            BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
            for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
                c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
                hw = hwc // BLOCK_CI_COUNT
                h = hw // weight_width
                w = hw % weight_width

                input_c_offset = c + tl.arange(0, BLOCK_CI)
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
                    input_tile_pointer
                    + (input_c_stride * input_c_offset)[None, :]
                    + (input_height_stride * input_height_offset)[:, None]
                    + (input_width_stride * input_width_offset)[:, None]
                )
                curr_weight_pointer = (
                    weight_tile_pointer
                    + (weight_c_stride * input_c_offset)[:, None]
                    + (weight_height_stride * h)
                    + (weight_width_stride * w)
                )

                input_mask = (
                    (in_n_point_value < in_n)[:, None]
                    & (input_c_offset < weight_c)[None, :]
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
            if bias_pointer is not None:
                bias_tile_pointer = (
                    bias_pointer
                    + (pid_group[None] * out_per_group_c)[None, :]
                    + output_c_offset[None, :]
                )
                mask_bias = (output_c_offset < out_per_group_c)[None, :]
                bias = tl.load(bias_tile_pointer, mask_bias, other=0).to(tl.float32)
                accum += bias
            output_tile_pointer = output_pointer + (
                (output_n_stride * in_n_point_value)[:, None]
                + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[
                    None, :
                ]
                + (output_height_stride * output_height_point_value)[:, None]
                + (output_width_stride * output_width_point_value)[:, None]
            )
            output_mask = (
                (in_n_point_value < in_n)[:, None]
                & (output_c_offset < out_per_group_c)[None, :]
                & (output_height_point_value < out_height)[:, None]
                & (output_width_point_value < out_width)[:, None]
            )

            tl.store(output_tile_pointer, accum, mask=output_mask)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV2D")
    unbatched = input.ndim == 3
    if unbatched:
        input = input.unsqueeze(0)
    # Keep the general implementation's backward for training calls.
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (input, weight, bias)
    ):
        output = _general_conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output.squeeze(0) if unbatched else output
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d expects a 3D/4D input and 4D weight")
    if input.dtype not in (torch.float16, torch.float32):
        raise ValueError("SpacemiT conv2d only supports float16 and float32")
    if input.dtype != weight.dtype or input.device != weight.device:
        raise ValueError("input and weight must have the same dtype and device")
    N, C, IH, IW = input.shape
    OC, CG, KH, KW = weight.shape
    if groups <= 0 or C != groups * CG or OC % groups or min(C, OC, KH, KW) <= 0:
        raise ValueError("incompatible convolution channels or groups")
    if bias is not None and (
        bias.shape != (OC,) or bias.dtype != input.dtype or bias.device != input.device
    ):
        raise ValueError("bias must match the output channels, dtype, and device")
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    if min(sh, sw, dh, dw) <= 0:
        raise ValueError("stride and dilation must be positive")
    if isinstance(padding, str):
        if padding == "valid":
            ph = pw = 0
            oh = (IH - dh * (KH - 1) - 1) // sh + 1
            ow = (IW - dw * (KW - 1) - 1) // sw + 1
        elif padding == "same":
            if (sh, sw) != (1, 1):
                raise ValueError("padding='same' requires stride=1")
            ph, pw = dh * (KH - 1) // 2, dw * (KW - 1) // 2
            oh, ow = IH, IW
        else:
            raise ValueError("padding must be 'same' or 'valid'")
    else:
        ph, pw = (padding, padding) if isinstance(padding, int) else padding
        if min(ph, pw) < 0:
            raise ValueError("padding must be nonnegative")
        oh = (IH + 2 * ph - dh * (KH - 1) - 1) // sh + 1
        ow = (IW + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError("kernel is larger than the padded input")
    output = torch.empty((N, OC, oh, ow), dtype=input.dtype, device=input.device)
    if N == 0:
        return output
    K, L = CG * KH * KW, oh * ow
    if (KH, KW, sh, sw, ph, pw) == (1, 1, 1, 1, 0, 0):
        columns = input.reshape(N * groups, K, L)
        weights = weight.reshape(groups, OC // groups, K)
        weights = (
            weights.unsqueeze(0)
            .expand(N, -1, -1, -1)
            .reshape(N * groups, OC // groups, K)
        )
        if weights.stride(1) != 1 and weights.stride(2) != 1:
            weights = weights.contiguous()
        if columns.stride(1) != 1 and columns.stride(2) != 1:
            columns = columns.contiguous()
        if bias is not None:
            bias = (
                bias.reshape(groups, OC // groups)
                .unsqueeze(0)
                .expand(N, -1, -1)
                .reshape(N * groups, OC // groups)
            )
        result = output.view(N * groups, OC // groups, L)

        def grid(meta):
            tiles = (
                N
                * groups
                * triton.cdiv(OC // groups, meta["TILE_M"])
                * triton.cdiv(L, meta["TILE_N"])
            )
            return (min(tiles, meta["num_ctas"], 512),)

        bmm_kernel[grid](
            weights,
            columns,
            result,
            N * groups,
            OC // groups,
            L,
            K,
            *weights.stride(),
            *columns.stride(),
            *result.stride(),
            Bias=bias,
            stride_bias_b=bias.stride(0) if bias is not None else 0,
            stride_bias_m=bias.stride(1) if bias is not None else 0,
        )
        return output.squeeze(0) if unbatched else output

    if bias is not None:
        bias = bias.contiguous()

    def grid(meta):
        tiles = (
            triton.cdiv(N * oh * ow, meta["BLOCK_NI_HO_WO"])
            * triton.cdiv(OC // groups, meta["BLOCK_CO"])
            * groups
        )
        return (min(tiles, 512),)

    conv2d_kernel[grid](
        input,
        weight,
        output,
        bias,
        N,
        IH,
        IW,
        OC,
        oh,
        ow,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        CG,
        KH,
        KW,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
    )
    return output.squeeze(0) if unbatched else output
