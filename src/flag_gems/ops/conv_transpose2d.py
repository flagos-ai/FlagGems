import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)

_CONV_TRANSPOSE2D_DIRECT_KEYS = [
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
]

_CONV_TRANSPOSE2D_RESIDUE_KEYS = [
    "in_n",
    "weight_c",
    "input_height",
    "input_width",
    "out_c",
    "out_height",
    "out_width",
    "weight_height",
    "weight_width",
    "padding_height",
    "padding_width",
    "groups",
]

_CONV_TRANSPOSE2D_RESIDUE_FP16_CONFIGS = [
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 512, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=8,
        num_stages=3,
    ),
]


_CONV_TRANSPOSE2D_RESIDUE_LOW_PRECISION_CONFIGS = _CONV_TRANSPOSE2D_RESIDUE_FP16_CONFIGS


_CONV_TRANSPOSE2D_RESIDUE_LOW_PRECISION_LARGE_KERNEL_CONFIGS = (
    _CONV_TRANSPOSE2D_RESIDUE_FP16_CONFIGS
    + [
        triton.Config(
            {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 16, "BLOCK_CI": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 16, "BLOCK_CI": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_NI_HO_WO": 512, "BLOCK_CO": 16, "BLOCK_CI": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 32, "BLOCK_CI": 16},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_NI_HO_WO": 512, "BLOCK_CO": 32, "BLOCK_CI": 16},
            num_warps=8,
            num_stages=2,
        ),
    ]
)


_CONV_TRANSPOSE2D_RESIDUE_COMPACT_CONFIGS = [
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
]


_CONV_TRANSPOSE2D_RESIDUE_FP32_CONFIGS = runtime.get_tuned_config("conv2d_forward")


_CONV_TRANSPOSE2D_RESIDUE_FP32_LARGE_KERNEL_CONFIGS = (
    runtime.get_tuned_config("conv2d_forward")
    + _CONV_TRANSPOSE2D_RESIDUE_COMPACT_CONFIGS
)


_CONV_TRANSPOSE2D_DIRECT_CONFIGS = [
    triton.Config(
        {"BLOCK_NI_HO_WO": 32, "BLOCK_CO": 16, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 32, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 16, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 16, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 256, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
]


_CONV_TRANSPOSE2D_SCATTER_CONFIGS = [
    triton.Config(
        {"BLOCK_NI_HI_WI": 64, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HI_WI": 128, "BLOCK_CO": 32, "BLOCK_CI": 16},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HI_WI": 128, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_NI_HI_WI": 256, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=3,
    ),
]


_CONV_TRANSPOSE2D_SCATTER_KEYS = [
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
]


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


def _to_pair(value):
    if isinstance(value, (list, tuple)):
        assert len(value) == 2, f"Expected length-2 value, got {value}"
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _validate_conv_transpose2d_params(
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    output_padding_height,
    output_padding_width,
    dilation_height,
    dilation_width,
    groups,
):
    assert groups > 0, f"groups must be positive, got {groups}"
    assert (
        stride_height > 0 and stride_width > 0
    ), f"stride must be positive, got {(stride_height, stride_width)}"
    assert (
        padding_height >= 0 and padding_width >= 0
    ), f"padding must be non-negative, got {(padding_height, padding_width)}"
    assert output_padding_height >= 0 and output_padding_width >= 0, (
        "output_padding must be non-negative, "
        f"got {(output_padding_height, output_padding_width)}"
    )
    assert (
        dilation_height > 0 and dilation_width > 0
    ), f"dilation must be positive, got {(dilation_height, dilation_width)}"
    assert (
        output_padding_height < stride_height or output_padding_height < dilation_height
    ), (
        "output_padding height must be smaller than either stride or dilation, "
        f"got output_padding={output_padding_height}, stride={stride_height}, "
        f"dilation={dilation_height}"
    )
    assert (
        output_padding_width < stride_width or output_padding_width < dilation_width
    ), (
        "output_padding width must be smaller than either stride or dilation, "
        f"got output_padding={output_padding_width}, stride={stride_width}, "
        f"dilation={dilation_width}"
    )


@libentry()
@triton.autotune(
    configs=_CONV_TRANSPOSE2D_DIRECT_CONFIGS,
    key=_CONV_TRANSPOSE2D_DIRECT_KEYS,
)
@triton.jit
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
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

    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += weight_n_stride * pid_group * weight_c

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_height_numerator = (
            output_height_point_value + padding_height - (h * dilation_height)
        )
        input_width_numerator = (
            output_width_point_value + padding_width - (w * dilation_width)
        )

        height_aligned = input_height_numerator % stride_height == 0
        width_aligned = input_width_numerator % stride_width == 0
        input_height_offset = tl.where(
            height_aligned, input_height_numerator // stride_height, 0
        )
        input_width_offset = tl.where(
            width_aligned, input_width_numerator // stride_width, 0
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_n_stride * input_c_offset)[:, None]
            + (weight_c_stride * output_c_offset)[None, :]
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (input_height_numerator >= 0)[:, None]
            & height_aligned[:, None]
            & (input_height_offset < input_height)[:, None]
            & (input_width_numerator >= 0)[:, None]
            & width_aligned[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)
        accum += tl.dot(input_block, weight_block, input_precision="ieee")

    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias, other=0.0).to(tl.float32)
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


@triton.jit
def _conv_transpose2d_forward_residue_body(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    compact_out_height: tl.constexpr,
    compact_out_width: tl.constexpr,
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
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    output_residue_height: tl.constexpr,
    output_residue_width: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    compact_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    compact_plane = compact_out_height * compact_out_width
    compact_ni_ho_offset = compact_offset // compact_out_width
    in_n_point_value = compact_offset // compact_plane
    compact_height_value = compact_ni_ho_offset % compact_out_height
    compact_width_value = compact_offset % compact_out_width
    output_height_point_value = (
        compact_height_value * stride_height + output_residue_height
    )
    output_width_point_value = compact_width_value * stride_width + output_residue_width

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += weight_n_stride * pid_group * weight_c

    height_residue: tl.constexpr = (
        output_residue_height + padding_height
    ) % stride_height
    width_residue: tl.constexpr = (output_residue_width + padding_width) % stride_width
    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT: tl.constexpr = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for h in range(weight_height):
        if (h * dilation_height) % stride_height == height_residue:
            input_height_numerator = (
                output_height_point_value + padding_height - h * dilation_height
            )
            input_height_offset = input_height_numerator // stride_height
            for w in range(weight_width):
                if (w * dilation_width) % stride_width == width_residue:
                    input_width_numerator = (
                        output_width_point_value + padding_width - w * dilation_width
                    )
                    input_width_offset = input_width_numerator // stride_width
                    for c_iter in range(BLOCK_CI_COUNT):
                        input_c_offset = c_iter * BLOCK_CI + tl.arange(0, BLOCK_CI)

                        curr_input_pointer = (
                            input_pointer
                            + (input_c_stride * input_c_offset)[None, :]
                            + (input_height_stride * input_height_offset)[:, None]
                            + (input_width_stride * input_width_offset)[:, None]
                        )
                        curr_weight_pointer = (
                            weight_pointer
                            + (weight_n_stride * input_c_offset)[:, None]
                            + (weight_c_stride * output_c_offset)[None, :]
                            + (weight_height_stride * h)
                            + (weight_width_stride * w)
                        )

                        input_mask = (
                            (in_n_point_value < in_n)[:, None]
                            & (input_c_offset < weight_c)[None, :]
                            & (input_height_numerator >= 0)[:, None]
                            & (input_height_offset < input_height)[:, None]
                            & (input_width_numerator >= 0)[:, None]
                            & (input_width_offset < input_width)[:, None]
                        )
                        weight_mask = (input_c_offset < weight_c)[:, None] & (
                            output_c_offset < out_per_group_c
                        )[None, :]

                        input_block = tl.load(
                            curr_input_pointer, mask=input_mask, other=0.0
                        )
                        weight_block = tl.load(
                            curr_weight_pointer, mask=weight_mask, other=0.0
                        )
                        accum += tl.dot(
                            input_block, weight_block, input_precision="ieee"
                        )

    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias, other=0.0).to(tl.float32)
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
    configs=_CONV_TRANSPOSE2D_RESIDUE_FP32_CONFIGS,
    key=_CONV_TRANSPOSE2D_RESIDUE_KEYS,
)
@triton.jit
def conv_transpose2d_forward_residue_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    compact_out_height: tl.constexpr,
    compact_out_width: tl.constexpr,
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
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    output_residue_height: tl.constexpr,
    output_residue_width: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv_transpose2d_forward_residue_body(
        input_pointer,
        weight_pointer,
        output_pointer,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        compact_out_height,
        compact_out_width,
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
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        padding_height,
        padding_width,
        groups,
        output_residue_height,
        output_residue_width,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.autotune(
    configs=_CONV_TRANSPOSE2D_RESIDUE_FP32_LARGE_KERNEL_CONFIGS,
    key=_CONV_TRANSPOSE2D_RESIDUE_KEYS,
)
@triton.jit
def conv_transpose2d_forward_residue_fp32_large_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    compact_out_height: tl.constexpr,
    compact_out_width: tl.constexpr,
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
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    output_residue_height: tl.constexpr,
    output_residue_width: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv_transpose2d_forward_residue_body(
        input_pointer,
        weight_pointer,
        output_pointer,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        compact_out_height,
        compact_out_width,
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
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        padding_height,
        padding_width,
        groups,
        output_residue_height,
        output_residue_width,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.autotune(
    configs=_CONV_TRANSPOSE2D_RESIDUE_LOW_PRECISION_CONFIGS,
    key=_CONV_TRANSPOSE2D_RESIDUE_KEYS,
)
@triton.jit
def conv_transpose2d_forward_residue_fp16_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    compact_out_height: tl.constexpr,
    compact_out_width: tl.constexpr,
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
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    output_residue_height: tl.constexpr,
    output_residue_width: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv_transpose2d_forward_residue_body(
        input_pointer,
        weight_pointer,
        output_pointer,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        compact_out_height,
        compact_out_width,
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
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        padding_height,
        padding_width,
        groups,
        output_residue_height,
        output_residue_width,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.autotune(
    configs=_CONV_TRANSPOSE2D_RESIDUE_LOW_PRECISION_LARGE_KERNEL_CONFIGS,
    key=_CONV_TRANSPOSE2D_RESIDUE_KEYS,
)
@triton.jit
def conv_transpose2d_forward_residue_low_precision_large_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    compact_out_height: tl.constexpr,
    compact_out_width: tl.constexpr,
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
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    output_residue_height: tl.constexpr,
    output_residue_width: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    _conv_transpose2d_forward_residue_body(
        input_pointer,
        weight_pointer,
        output_pointer,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_c,
        out_height,
        out_width,
        compact_out_height,
        compact_out_width,
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
        weight_c,
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        padding_height,
        padding_width,
        groups,
        output_residue_height,
        output_residue_width,
        BLOCK_NI_HO_WO,
        BLOCK_CI,
        BLOCK_CO,
    )


@libentry()
@triton.jit
def conv_transpose2d_fill_kernel(
    output_pointer,
    bias_pointer,
    total_elements,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    ow = offsets % out_width
    oh = (offsets // out_width) % out_height
    oc = (offsets // (out_height * out_width)) % out_c
    n = offsets // (out_c * out_height * out_width)

    output_offset = (
        n * output_n_stride
        + oc * output_c_stride
        + oh * output_height_stride
        + ow * output_width_stride
    )
    values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if HAS_BIAS:
        values = tl.load(bias_pointer + oc, mask=mask, other=0.0)
    tl.store(output_pointer + output_offset, values, mask=mask)


@libentry()
@triton.autotune(
    configs=_CONV_TRANSPOSE2D_SCATTER_CONFIGS,
    key=_CONV_TRANSPOSE2D_SCATTER_KEYS,
)
@triton.jit
def conv_transpose2d_forward_scatter_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
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
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HI_WI: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_hi_wi = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group_tap = tl.program_id(2)

    tap_count: tl.constexpr = weight_height * weight_width
    input_plane: tl.constexpr = input_height * input_width
    pid_group = pid_group_tap // tap_count
    tap_id = pid_group_tap % tap_count
    kh = tap_id // weight_width
    kw = tap_id % weight_width

    offsets = pid_ni_hi_wi * BLOCK_NI_HI_WI + tl.arange(0, BLOCK_NI_HI_WI)
    iw = offsets % input_width
    ih = (offsets // input_width) % input_height
    n = offsets // input_plane

    oh = ih * stride_height - padding_height + kh * dilation_height
    ow = iw * stride_width - padding_width + kw * dilation_width

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_pointer += (input_n_stride * n + input_c_stride * pid_group * weight_c)[
        :, None
    ]
    weight_pointer += weight_n_stride * pid_group * weight_c

    accum = tl.zeros((BLOCK_NI_HI_WI, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT: tl.constexpr = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    for c_iter in range(BLOCK_CI_COUNT):
        input_c_offset = c_iter * BLOCK_CI + tl.arange(0, BLOCK_CI)
        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_height_stride * ih)[:, None]
            + (input_width_stride * iw)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_n_stride * input_c_offset)[:, None]
            + (weight_c_stride * output_c_offset)[None, :]
            + (weight_height_stride * kh)
            + (weight_width_stride * kw)
        )

        task_mask = (
            (n < in_n) & (oh >= 0) & (oh < out_height) & (ow >= 0) & (ow < out_width)
        )
        input_mask = task_mask[:, None] & (input_c_offset < weight_c)[None, :]
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)
        accum += tl.dot(input_block, weight_block, input_precision="ieee")

    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias, other=0.0).to(tl.float32)
    accum += bias

    output_pointer += (
        (output_n_stride * n)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * oh)[:, None]
        + (output_width_stride * ow)[:, None]
    )
    output_mask = (
        (n < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (oh >= 0)[:, None]
        & (oh < out_height)[:, None]
        & (ow >= 0)[:, None]
        & (ow < out_width)[:, None]
    )
    tl.store(output_pointer, accum, mask=output_mask)


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
    stride_height, stride_width = _to_pair(stride)
    padding_height, padding_width = _to_pair(padding)
    output_padding_height, output_padding_width = _to_pair(output_padding)
    dilation_height, dilation_width = _to_pair(dilation)
    _validate_conv_transpose2d_params(
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        output_padding_height,
        output_padding_width,
        dilation_height,
        dilation_width,
        groups,
    )

    if torch.is_grad_enabled():
        return torch.ops.aten.conv_transpose2d.input.redispatch(
            _FALLBACK_KEYSET,
            input,
            weight,
            bias,
            [stride_height, stride_width],
            [padding_height, padding_width],
            [output_padding_height, output_padding_width],
            groups,
            [dilation_height, dilation_width],
        )

    assert input.ndim == 4, f"Input must be 4D, received shape {input.shape}"
    assert weight.ndim == 4, f"Weights must be 4D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, received shape {bias.shape}"
    assert (
        input.shape[1] == weight.shape[0]
    ), f"Incompatible input ({input.shape}) and weight ({weight.shape}) channels"
    assert (
        input.shape[1] % groups == 0
    ), f"Input channels {input.shape[1]} must be divisible by groups {groups}"

    in_n, input_channels, input_height, input_width = input.shape
    _, out_channels_per_group, kernel_height, kernel_width = weight.shape
    out_channels = groups * out_channels_per_group

    assert (
        bias is None or bias.shape[0] == out_channels
    ), f"Incompatible weight ({weight.shape}) and bias ({bias.shape}) shape"

    out_height = conv_transpose2d_output_size(
        input_height,
        kernel_height,
        stride_height,
        padding_height,
        output_padding_height,
        dilation_height,
    )
    out_width = conv_transpose2d_output_size(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        output_padding_width,
        dilation_width,
    )

    output = torch.empty(
        (in_n, out_channels, out_height, out_width),
        dtype=input.dtype,
        device=input.device,
    )

    if bias is None:
        bias_pointer = torch.zeros(out_channels, dtype=input.dtype, device=input.device)
    else:
        bias_pointer = bias

    effective_kernel_height = dilation_height * (kernel_height - 1) + 1
    effective_kernel_width = dilation_width * (kernel_width - 1) + 1
    use_scatter_kernel = (
        stride_height * stride_width > 4
        and effective_kernel_height <= stride_height
        and effective_kernel_width <= stride_width
    )
    if use_scatter_kernel:
        total_elements = output.numel()
        fill_grid = (triton.cdiv(total_elements, 1024),)
        conv_transpose2d_fill_kernel[fill_grid](
            output,
            bias_pointer,
            total_elements,
            out_channels,
            out_height,
            out_width,
            *output.stride(),
            HAS_BIAS=bias is not None,
            BLOCK_SIZE=1024,
        )
        scatter_grid = lambda meta: (
            triton.cdiv(in_n * input_height * input_width, meta["BLOCK_NI_HI_WI"]),
            triton.cdiv(int(out_channels // groups), meta["BLOCK_CO"]),
            groups * kernel_height * kernel_width,
        )
        conv_transpose2d_forward_scatter_kernel[scatter_grid](
            input,
            weight,
            output,
            bias_pointer,
            in_n,
            input_height,
            input_width,
            out_channels,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            input_channels // groups,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            padding_height,
            padding_width,
            groups=groups,
        )
        return output

    stride_product = stride_height * stride_width
    large_stride3_output = (
        stride_height == 3
        and stride_width == 3
        and in_n * out_height * out_width >= 131072
    )
    use_residue_kernel = (stride_height > 1 or stride_width > 1) and (
        stride_product <= 4 or large_stride3_output
    )
    if use_residue_kernel:
        for output_residue_height in range(stride_height):
            compact_out_height = (
                out_height + stride_height - 1 - output_residue_height
            ) // stride_height
            for output_residue_width in range(stride_width):
                compact_out_width = (
                    out_width + stride_width - 1 - output_residue_width
                ) // stride_width
                grid = lambda meta: (
                    triton.cdiv(
                        in_n * compact_out_height * compact_out_width,
                        meta["BLOCK_NI_HO_WO"],
                    ),
                    triton.cdiv(int(out_channels // groups), meta["BLOCK_CO"]),
                    groups,
                )
                if input.dtype in (torch.float16, torch.bfloat16) and (
                    kernel_height >= 5 or kernel_width >= 5
                ):
                    residue_kernel = (
                        conv_transpose2d_forward_residue_low_precision_large_kernel
                    )
                elif input.dtype in (torch.float16, torch.bfloat16):
                    residue_kernel = conv_transpose2d_forward_residue_fp16_kernel
                elif kernel_height >= 5 or kernel_width >= 5:
                    residue_kernel = conv_transpose2d_forward_residue_fp32_large_kernel
                else:
                    residue_kernel = conv_transpose2d_forward_residue_kernel
                residue_kernel[grid](
                    input,
                    weight,
                    output,
                    bias_pointer,
                    in_n,
                    input_height,
                    input_width,
                    out_channels,
                    out_height,
                    out_width,
                    compact_out_height,
                    compact_out_width,
                    *input.stride(),
                    *weight.stride(),
                    *output.stride(),
                    input_channels // groups,
                    kernel_height,
                    kernel_width,
                    stride_height,
                    stride_width,
                    dilation_height,
                    dilation_width,
                    padding_height,
                    padding_width,
                    groups=groups,
                    output_residue_height=output_residue_height,
                    output_residue_width=output_residue_width,
                )
        return output

    grid = lambda meta: (
        triton.cdiv(in_n * out_height * out_width, meta["BLOCK_NI_HO_WO"]),
        triton.cdiv(int(out_channels // groups), meta["BLOCK_CO"]),
        groups,
    )
    conv_transpose2d_forward_kernel[grid](
        input,
        weight,
        output,
        bias_pointer,
        in_n,
        input_height,
        input_width,
        out_channels,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        input_channels // groups,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups=groups,
    )
    return output
