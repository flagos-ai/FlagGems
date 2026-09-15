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

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 2D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of 2D convolution.
    """
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# Layout of the tensors handed to the forward kernel. The kernel takes every
# stride as a runtime argument, so both modes run the same code and differ only
# in where the gathers land. They are not a routing decision: forward always
# packs the weight, and _LAYOUT_NCHW is there for dgrad, which reuses this
# kernel with plain NCHW operands.
_LAYOUT_NCHW = 0  # nothing converted
_LAYOUT_W = 1  # weight packed, input left as-is

# A @triton.jit body may only read a global that is already a constexpr.
_JIT_LAYOUT_NCHW = tl.constexpr(_LAYOUT_NCHW)


_FORWARD_CONFIGS = runtime.get_tuned_config("conv2d_forward")

# Widest BLOCK_CI/BLOCK_CO in the forward table. pack_weight never pads past
# this, and compact tiles are clamped to it.
_MAX_BLOCK_C = max(
    width
    for config in _FORWARD_CONFIGS
    for name in ("BLOCK_CO", "BLOCK_CI")
    if (width := config.kwargs.get(name)) is not None
)


@libentry()
@triton.jit
def _to_channels_last_kernel(
    src,
    dst,
    src_n_stride,
    src_c_stride,
    src_h_stride,
    src_w_stride,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """NCHW -> NHWC with coalesced loads and stores.

    Each program loads a [C, HW] tile (adjacent threads walk W, then H) and
    stores the transpose as [HW, C] (adjacent threads walk C). The old kernel
    linearized with C innermost, so neighbouring threads read H*W elements
    apart -- 88 KB on 210x210 -- and the copy ran at half the bandwidth.
    """
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_mask = offs_hw < height * width
    c_mask = offs_c < channels

    # Decode linear HW into (h, w) and apply the source strides. Using
    # src_h_stride / src_w_stride (not a packed HW stride of 1) keeps sliced
    # or otherwise non-contiguous NCHW inputs correct.
    h = offs_hw // width
    w = offs_hw % width
    tile = tl.load(
        src
        + n * src_n_stride
        + offs_c[:, None] * src_c_stride
        + (h * src_h_stride + w * src_w_stride)[None, :],
        mask=c_mask[:, None] & hw_mask[None, :],
    )
    tl.store(
        dst
        + n * channels * height * width
        + offs_hw[:, None] * channels
        + offs_c[None, :],
        tl.trans(tile),
        mask=hw_mask[:, None] & c_mask[None, :],
    )


_CL_BLOCK_C = 32
_CL_BLOCK_HW = 64


def to_channels_last(input: torch.Tensor):
    n, c, h, w = input.shape
    output = torch.empty(
        input.shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.channels_last,
    )
    grid = (
        triton.cdiv(h * w, _CL_BLOCK_HW),
        triton.cdiv(c, _CL_BLOCK_C),
        n,
    )
    _to_channels_last_kernel[grid](
        input,
        output,
        *input.stride(),
        c,
        h,
        w,
        BLOCK_C=_CL_BLOCK_C,
        BLOCK_HW=_CL_BLOCK_HW,
        num_warps=8,
    )
    return output


@triton.jit
def _pack_weight_kernel(
    src,
    dst,
    per_group_c,
    weight_c,
    khkw,
    co_group,
    co_pad,
    tap_stride,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    """[out_c, ci, kh, kw] -> [kh, kw, ci_pad, co_pad], pad region set to zero.

    Packed W has unit-stride ``co``, channel axes padded so a fitting tile can
    drop its mask, and each group in a ``co_group``-wide slot so the group base
    stays 64B aligned (an unaligned SME load reads the wrong data silently).
    """
    pid_hw = tl.program_id(0)
    off_co = tl.program_id(1) * BLOCK_CO + tl.arange(0, BLOCK_CO)
    off_ci = tl.program_id(2) * BLOCK_CI + tl.arange(0, BLOCK_CI)

    grp = off_co // co_group
    in_grp = off_co % co_group
    src_co = grp * per_group_c + in_grp

    block = tl.load(
        src + src_co[None, :] * weight_c * khkw + off_ci[:, None] * khkw + pid_hw,
        mask=(off_ci[:, None] < weight_c) & (in_grp < per_group_c)[None, :],
        other=0.0,
    )
    tl.store(
        dst + pid_hw * tap_stride + off_ci[:, None] * co_pad + off_co[None, :], block
    )


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def pack_weight(weight: torch.Tensor, groups: int):
    """Rewrite W to [kh, kw, ci_pad, co_pad] for a packed forward load.

    Returns ``(packed, strides, co_group, ci_tile, co_tile)``. ``strides`` fill
    the kernel's weight_*_stride slots as (co, ci, kh, kw). Tiles wider than
    ``ci_tile``/``co_tile`` must use a masked load.
    """
    out_c, weight_c, kh, kw = weight.shape
    ci_tile = min(_MAX_BLOCK_C, max(16, triton.next_power_of_2(weight_c)))
    ci_pad = _round_up(weight_c, ci_tile)
    per_group_c = out_c // groups
    # Floor is 64B so each group's co slot base stays SME-aligned.
    co_alignment = max(16, 64 // weight.element_size())
    co_tile = min(
        _MAX_BLOCK_C,
        max(co_alignment, triton.next_power_of_2(per_group_c)),
    )
    co_group = _round_up(per_group_c, co_tile)
    co_pad = groups * co_group

    packed = torch.empty(
        (kh, kw, ci_pad, co_pad), dtype=weight.dtype, device=weight.device
    )
    grid = (
        kh * kw,
        triton.cdiv(co_pad, co_tile),
        triton.cdiv(ci_pad, ci_tile),
    )
    src = weight if weight.is_contiguous() else weight.contiguous()
    _pack_weight_kernel[grid](
        src,
        packed,
        per_group_c,
        weight_c,
        kh * kw,
        co_group,
        co_pad,
        ci_pad * co_pad,
        BLOCK_CO=co_tile,
        BLOCK_CI=ci_tile,
        num_warps=8,
    )
    return (
        packed,
        (1, co_pad, kw * ci_pad * co_pad, ci_pad * co_pad),
        co_group,
        ci_tile,
        co_tile,
    )


@libentry()
@triton.autotune(
    configs=_FORWARD_CONFIGS,
    key=[
        "in_n",
        "weight_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
        # This key is all shape scalars, so it does not see strides, and the
        # forward call (channels_last weight) and the dgrad call (plain NCHW
        # operands) would share a tuned config whenever their shapes coincide.
        # LibEntry's own kernel cache does separate them, since it keys every
        # non-constexpr scalar by value and the strides differ -- it is only
        # the config that would be inherited.
        "LAYOUT",
    ],
)
@triton.jit
def conv2d_forward_kernel(
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
    weight_group_stride,
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
    HAS_BIAS: tl.constexpr,
    LAYOUT: tl.constexpr,
    PACKED_CI_TILE: tl.constexpr,
    PACKED_CO_TILE: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_height out_weight value in kernel
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    input_height_base = stride_height * output_height_point_value - padding_height
    input_width_base = stride_width * output_width_point_value - padding_width
    input_pointer += (
        input_height_stride * input_height_base + input_width_stride * input_width_base
    )[:, None]
    # weight_group_stride is the step between groups along co. It is not
    # weight_n_stride * out_per_group_c for the packed layout, which rounds each
    # group's slot up so this offset stays 64B aligned.
    weight_pointer += (
        weight_n_stride * output_c_offset + weight_group_stride * pid_group
    )[None, :]

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + input_height_stride * h * dilation_height
            + input_width_stride * w * dilation_width
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        if padding_height == 0 and padding_width == 0:
            # For a valid convolution, every tap of a valid output point is
            # inside the input. Keep the linear-output mask for the tail tile.
            input_mask = (ni_ho_wo_offset < in_n * out_height * out_width)[:, None] & (
                input_c_offset < weight_c
            )[None, :]
        else:
            input_height_offset = input_height_base + h * dilation_height
            input_width_offset = input_width_base + w * dilation_width
            input_mask = (
                (in_n_point_value < in_n)[:, None]
                & (input_c_offset < weight_c)[None, :]
                & (0 <= input_height_offset)[:, None]
                & (input_height_offset < input_height)[:, None]
                & (0 <= input_width_offset)[:, None]
                & (input_width_offset < input_width)[:, None]
            )
        input_block = tl.load(curr_input_pointer, mask=input_mask)
        # Packed W, tile inside the pad: unmasked (SME). dgrad and overflow tiles
        # stay masked; those two loads were identical, so they share this branch.
        if (
            LAYOUT != _JIT_LAYOUT_NCHW
            and BLOCK_CI <= PACKED_CI_TILE
            and BLOCK_CO <= PACKED_CO_TILE
        ):
            weight_block = tl.load(curr_weight_pointer)
        else:
            weight_block = tl.load(
                curr_weight_pointer,
                mask=(input_c_offset < weight_c)[:, None]
                & (output_c_offset < out_per_group_c)[None, :],
            )

        accum += tl.dot(input_block, weight_block, allow_tf32=False)
    if HAS_BIAS:
        bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
            None, :
        ]
        mask_bias = (output_c_offset < out_per_group_c)[None, :]
        bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
        accum += bias
    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_backward_weight"),
    key=[
        "in_n",
        "input_height",
        "input_width",
        "weight_height",
        "weight_width",
        "input_c",
        "stride_height",
        "stride_width",
        "out_height",
        "out_width",
        "out_c",
        "padding_height",
        "padding_width",
    ],
)
@triton.jit
def conv2d_backward_kernel_weight(
    input_pointer,
    out_grad_pointer,
    weight_pointer,
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
    BLOCK_NO: tl.constexpr,
    BLOCK_CI_HK_WK: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # load out_grad n (groups out_c)  ho wo
    # load weight (groups out_c) ci h w
    # load input n (groups ci)  hi wi

    # init pid and offset 0 for ci*hk*wk, 1 for groups, 2 for co.
    pid_ci_hk_wk = tl.program_id(0)
    pid_groups = tl.program_id(1)
    pid_co = tl.program_id(2)

    # caculate ci weight_height weight_weight value in kernel
    ci_hk_wk_offset = pid_ci_hk_wk * BLOCK_CI_HK_WK + tl.arange(0, BLOCK_CI_HK_WK)
    ci_hk_offset = ci_hk_wk_offset // weight_width
    ci_point_value = ci_hk_offset // weight_height
    weight_height_point_value = ci_hk_offset % weight_height
    weight_width_point_value = ci_hk_wk_offset % weight_width

    # caculate init pointer info of tensors
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    out_grad_pointer += (output_c_offset * output_c_stride)[None, :] + (
        pid_groups[None] * output_c_stride * out_c
    )[:, None]

    weight_pointer += (
        pid_groups * weight_n_stride * out_c + output_c_offset * weight_n_stride
    )[None, :] + (
        ci_point_value * weight_c_stride
        + weight_height_point_value * weight_height_stride
        + weight_width_point_value * weight_width_stride
    )[
        :, None
    ]

    input_pointer += (ci_point_value * input_c_stride[None])[:, None] + (
        pid_groups[None] * input_c_stride * input_c
    )[None, :]

    # calculate the values of the input based on the width and height of the output by looping
    accum = tl.zeros((BLOCK_CI_HK_WK, BLOCK_CO), dtype=tl.float32)
    for h in range(0, out_height):
        for w in range(0, out_width):
            for n in range(0, in_n, BLOCK_NO):
                output_n_offset = n + tl.arange(0, BLOCK_NO)

                # caculate input pointer to [cin*kh*kw, *] out_grad pointer to [*, out_c], N*hout*wout as reduce dim
                curr_out_grad_pointer = (
                    out_grad_pointer
                    + (
                        output_n_offset * output_n_stride
                        + h * output_height_stride
                        + w * output_width_stride
                    )[:, None]
                )
                out_grad_mask = (output_n_offset < in_n)[:, None] & (
                    output_c_offset < out_c
                )[None, :]

                curr_out_grad = tl.load(curr_out_grad_pointer, mask=out_grad_mask)

                input_height_offset = (
                    weight_height_point_value * dilation_height
                    - padding_height
                    + stride_height * h
                )

                input_width_offset = (
                    weight_width_point_value * dilation_width
                    - padding_width
                    + stride_width * w
                )

                curr_input_pointer = (
                    input_pointer
                    + (input_n_stride * output_n_offset)[None, :]
                    + (input_height_stride * input_height_offset)[:, None]
                    + (input_width_stride * input_width_offset)[:, None]
                )
                input_mask = (
                    (output_n_offset < in_n)[None, :]
                    & (ci_point_value < input_c)[:, None]
                    & (0 <= input_height_offset)[:, None]
                    & (input_height_offset < input_height)[:, None]
                    & (0 <= input_width_offset)[:, None]
                    & (input_width_offset < input_width)[:, None]
                )

                curr_input = tl.load(curr_input_pointer, mask=input_mask)
                accum += tl.dot(curr_input, curr_out_grad, allow_tf32=False)

    weight_mask = (
        (ci_point_value < input_c)[:, None]
        & (output_c_offset < out_c)[None, :]
        & (weight_height_point_value < weight_height)[:, None]
        & (weight_width_point_value < weight_width)[:, None]
    )
    tl.store(weight_pointer, accum, weight_mask)


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logger.debug("GEMS_ILUVATAR CONV2D")
        assert weight.ndim == 4, "Weights must be 4D, received shape {weight.shape}"
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
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(padding, (list, tuple)):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        if isinstance(dilation, (list, tuple)):
            dilation_height, dilation_width = dilation
        else:
            dilation_height = dilation_width = dilation

        in_n, _, input_height, input_width = input.shape
        out_c, weight_c, weight_height, weight_width = weight.shape

        # weight_c may be below the K >= 16 that tl.dot requires, but the kernel's
        # K dimension is BLOCK_CI, not weight_c, and every conv2d_forward config in
        # tune_configs.yaml has BLOCK_CI >= 16. Channels below 16 are handled by the
        # input_mask / weight_mask instead. This holds only as long as nothing
        # shrinks BLOCK_CI behind our back: a flagtree build with AABS enabled would
        # rewrite it to next_power_of_2(weight_c) and break the constraint.

        out_height = conv2d_output_size(
            input_height, weight_height, stride_height, padding_height, dilation_height
        )
        out_width = conv2d_output_size(
            input_width, weight_width, stride_width, padding_width, dilation_width
        )

        output_dtype = input.dtype
        output = torch.empty(
            (in_n, out_c, out_height, out_width),
            device=input.device,
            dtype=output_dtype,
        )

        # BLOCK_NI_HO_WO along the in_n, out_height, and out_width dimensions,
        # BLOCK_CO along the out_c,
        # one group per cat
        grid = lambda META: (
            triton.cdiv(in_n * out_height * out_width, META["BLOCK_NI_HO_WO"]),
            triton.cdiv(int(out_c // groups), META["BLOCK_CO"]),
            groups,
        )

        def launch(x, w, w_strides, w_group_stride, packed_ci_tile, packed_co_tile):
            conv2d_forward_kernel[grid](
                x,
                w,
                output,
                bias,
                in_n,
                input_height,
                input_width,
                out_c,
                out_height,
                out_width,
                *x.stride(),
                *w_strides,
                w_group_stride,
                *output.stride(),
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
                HAS_BIAS=bias is not None,
                LAYOUT=_LAYOUT_W,
                PACKED_CI_TILE=packed_ci_tile,
                PACKED_CO_TILE=packed_co_tile,
            )

        # W is always packed. A is copied to channels_last only for long 2-byte
        # reductions; that load stays masked and does not put A on SME.
        if input.element_size() == 2 and weight_height * weight_width >= 16:
            input = to_channels_last(input)
        packed, w_strides, co_group, ci_tile, co_tile = pack_weight(weight, groups)
        launch(input, packed, w_strides, co_group, ci_tile, co_tile)

        ctx.save_for_backward(weight, input, bias)

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.dilation = (dilation_height, dilation_width)

        ctx.weight_info = (
            int(out_c / groups),
            weight_c,
            weight_height,
            weight_width,
        )
        ctx.input_info = (in_n, input_height, input_width)
        ctx.out_info = (out_height, out_width)

        ctx.device = input.device
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_ILUVATAR CONV2D VJP")
        weight, input, bias = ctx.saved_tensors
        # (out_c equals origin cout divide groups)
        out_c, weight_c, weight_height, weight_width = ctx.weight_info
        in_n, input_height, input_width = ctx.input_info
        out_height, out_width = ctx.out_info

        device = ctx.device
        groups = ctx.groups

        stride_height, stride_width = ctx.stride
        dilation_height, dilation_width = ctx.dilation
        padding_height, padding_width = ctx.padding

        revert_padding_height = dilation_height * (weight_height - 1) - padding_height
        revert_padding_width = dilation_width * (weight_width - 1) - padding_width
        revert_weight = weight.clone()
        revert_weight = torch.flip(revert_weight, dims=[2, 3]).contiguous()

        if groups != 1:
            revert_weight = revert_weight.reshape(
                groups, out_c, weight_c, weight_height, weight_width
            )
            revert_weight = revert_weight.transpose(1, 2)
            revert_weight = revert_weight.reshape(
                groups * weight_c, out_c, weight_height, weight_width
            ).contiguous()
        else:
            revert_weight = revert_weight.transpose(0, 1).contiguous()

        new_out_height = out_grad.shape[2] + (stride_height - 1) * (
            out_grad.shape[2] - 1
        )
        new_out_width = out_grad.shape[3] + (stride_width - 1) * (out_grad.shape[3] - 1)

        new_out = torch.zeros(
            out_grad.shape[0],
            out_grad.shape[1],
            new_out_height,
            new_out_width,
            device=device,
            dtype=out_grad.dtype,
        )

        # copy out_grad to new_out
        if stride_height > 1 or stride_width > 1:
            for i in range(out_grad.shape[2]):
                for j in range(out_grad.shape[3]):
                    new_out[:, :, i * (stride_height), j * (stride_width)] = out_grad[
                        :, :, i, j
                    ]
        else:
            new_out = out_grad

        input_back = torch.zeros(
            in_n,
            weight_c * groups,
            input_height,
            input_width,
            dtype=torch.float32,
            device=device,
        )

        grid = lambda META: (
            triton.cdiv(
                out_grad.shape[0] * input_height * input_width, META["BLOCK_NI_HO_WO"]
            ),
            triton.cdiv(int(weight_c), META["BLOCK_CO"]),
            groups,
        )

        # Backward path: out_c is passed as kernel's weight_c (constexpr K dim), so
        # this is the same situation the forward path documents above and the padding
        # is equally unnecessary here. Left in place for now because, unlike forward,
        # the out_c < 16 backward path has not been measured or verified.
        _MIN_DOT_K = 16
        bwd_weight_c = out_c  # This becomes the kernel's weight_c constexpr
        if bwd_weight_c < _MIN_DOT_K:
            pad_c = _MIN_DOT_K - bwd_weight_c
            # Pad revert_weight on dim=1 (the out_c/group dimension)
            # revert_weight shape: [groups*weight_c, out_c, kH, kW]
            revert_weight = torch.nn.functional.pad(
                revert_weight, (0, 0, 0, 0, 0, pad_c)
            )
            # Pad new_out on channel dim (dim=1).
            # new_out shape: [N, groups*out_c, H, W]
            # Each group has out_c channels, need to pad each group to _MIN_DOT_K.
            if groups == 1:
                new_out = torch.nn.functional.pad(new_out, (0, 0, 0, 0, 0, pad_c))
            else:
                N_bwd, C_bwd, H_bwd, W_bwd = new_out.shape
                new_out = new_out.reshape(N_bwd, groups, out_c, H_bwd, W_bwd)
                new_out = torch.nn.functional.pad(new_out, (0, 0, 0, 0, 0, pad_c))
                new_out = new_out.reshape(N_bwd, groups * (out_c + pad_c), H_bwd, W_bwd)
            bwd_weight_c = bwd_weight_c + pad_c

        conv2d_forward_kernel[grid](
            new_out,
            revert_weight,
            input_back,
            None,
            out_grad.shape[0],
            new_out_height,
            new_out_width,
            groups * weight_c,
            input_height,
            input_width,
            *new_out.stride(),
            *revert_weight.stride(),
            # Plain NCHW: the kernel's out_c here is groups * weight_c, so the
            # per-group step along co is weight_c rows.
            revert_weight.stride()[0] * weight_c,
            *input_back.stride(),
            bwd_weight_c,
            weight_height,
            weight_width,
            1,
            1,
            revert_padding_height,
            revert_padding_width,
            dilation_height,
            dilation_width,
            groups=groups,
            HAS_BIAS=False,
            # dgrad builds its operands with pad/flip/transpose and hands them
            # over as plain NCHW; converting them is a separate question from
            # the forward one and is not answered here.
            LAYOUT=_LAYOUT_NCHW,
            PACKED_CI_TILE=0,
            PACKED_CO_TILE=0,
        )

        weight_back = torch.zeros(
            out_c * groups,
            weight_c,
            weight_height,
            weight_width,
            dtype=weight.dtype,
            device=device,
        )

        grid_weight = lambda meta: (
            triton.cdiv(
                weight_c * weight_height * weight_width, meta["BLOCK_CI_HK_WK"]
            ),
            groups,
            triton.cdiv(out_c, meta["BLOCK_CO"]),
        )
        conv2d_backward_kernel_weight[grid_weight](
            input,
            out_grad,
            weight_back,
            *input.stride(),
            *weight.stride(),
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
        )
        if bias is not None:
            bias_grad = out_grad.sum(dim=(0, 2, 3))
        else:
            bias_grad = None
        return (
            input_back,
            weight_back,
            bias_grad,
            None,
            None,
            None,
            None,
        )


# todo test SymInt[2] of stride or padding
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(padding, str):
        if padding == "same":
            assert stride == 1, "Doesn't support any stride values other than 1 \
                in padding = 'same' mode, received stride value {stride}"
            ih = input.shape[-2]
            iw = input.shape[-1]
            kernel_size_h = weight.shape[-2]
            kernel_size_w = weight.shape[-1]
            padding_h = int(
                math.ceil(
                    (stride * (ih - 1) + 1 + dilation * (kernel_size_h - 1) - ih) / 2
                )
            )
            padding_w = int(
                math.ceil(
                    (stride * (iw - 1) + 1 + dilation * (kernel_size_w - 1) - iw) / 2
                )
            )
            oh = int(
                (ih + 2 * padding_h - dilation * (kernel_size_h - 1) - 1) / stride + 1
            )
            ow = int(
                (iw + 2 * padding_w - dilation * (kernel_size_w - 1) - 1) / stride + 1
            )
            # Use per-dimension padding so asymmetric kernels (kh != kw) pad each
            # spatial axis independently; the trailing slice trims the extra pad
            # that ceil() introduces on the bottom/right, matching torch's "same".
            padding = (padding_h, padding_w)
            return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)[
                ..., (oh - ih) :, (ow - iw) :
            ]
        elif padding == "valid":
            return Conv2d.apply(input, weight, bias, stride, 0, dilation, groups)
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only'valild'/'same' are allowed."
            )
    else:
        return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)
