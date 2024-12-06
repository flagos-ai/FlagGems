import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


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

    Returns:
        Output size of 2D convolution.
    """
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv2d_forward_config(
    BLOCK_NI_HO_WO: int,
    BLOCK_CI: int,
    BLOCK_CO: int,
    n_warps: int,
    num_stages: int,
) -> triton.Config:
    """
    Creates a triton.Config object for conv2d_forward_kernel
    given meta-parameters for auto-tuning.

    Args:
        BLOCK_NI_HO_WO: Block size across the input n, output height, and
                        output width dimensions.
        BLOCK_CI: Block size across the input c dimension.
        BLOCK_CO: Block size across the output c dimension.
        n_warps: Number of warps to use for the kernel when compiled for GPUs.
        num_stages: Number of stages the compiler uses to software-pipeline.

    Returns:
        Kernel configuration.
    """
    return triton.Config(
        {"BLOCK_NI_HO_WO": BLOCK_NI_HO_WO, "BLOCK_CI": BLOCK_CI, "BLOCK_CO": BLOCK_CO},
        num_warps=n_warps,
        num_stages=num_stages,
    )


@libentry()
@triton.autotune(
    configs=[
        conv2d_forward_config(128, 32, 128, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 32, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 32, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 64, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 64, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 32, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 32, 128, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 32, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 32, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 32, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 128, 128, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 128, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 64, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 64, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 16, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 16, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 16, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(32, 16, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 16, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 16, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 64, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 32, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(128, 32, 32, n_warps=2, num_stages=4),
    ],
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
    ],
)
@triton.jit
def conv2d_forward_kernel(
    input_pointer,
    weight_pointer,
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
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

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
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
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

        input_block = tl.load(curr_input_pointer, mask=input_mask)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

        accum += tl.dot(input_block, weight_block, allow_tf32=False)

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
    configs=[
        triton.Config(
            {"BLOCK_NO_HO_WO": 16, "BLOCK_CO": 16, "BLOCK_CI": 16}, num_stages=5
        ),
    ],
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
def conv2d_backward_kernel(
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
    BLOCK_CI: tl.constexpr,
    BLOCK_NO_HO_WO: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # load out_grad n (groups out_c)  ho wo
    # load weight (groups out_c) ci h w

    # init pid and offset 0 for nout*hout*wout 1 for groups 2 for ci
    pid_no_ho_wo = tl.program_id(0)
    pid_groups = tl.program_id(1)
    pid_ci = tl.program_id(2)

    # caculate in_n out_height out_weight value in kernel
    no_ho_wo_offset = pid_no_ho_wo * BLOCK_NO_HO_WO + tl.arange(0, BLOCK_NO_HO_WO)
    no_ho_offset = no_ho_wo_offset // out_width
    no_point_value = no_ho_offset // out_height
    output_height_point_value = no_ho_offset % out_height
    output_width_point_value = no_ho_wo_offset % out_width

    # caculate init pointer info of tensors
    out_grad_pointer += (no_point_value * output_n_stride)[:, None] + (
        pid_groups * output_c_stride * out_c
        + output_height_point_value * output_height_stride
        + output_width_point_value * output_width_stride
    )[:, None]

    input_c_offset = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    weight_pointer += (pid_groups * weight_n_stride * out_c)[None, :] + (
        input_c_offset * weight_c_stride
    )[None, :]

    input_pointer += (
        (no_point_value * input_n_stride)[:, None]
        + (pid_groups * input_c_stride * input_c)[:, None]
        + (input_c_stride * input_c_offset)[None, :]
    )

    # calculate the values of the input based on the width and height of the weight by looping
    for h in range(0, weight_height):
        for w in range(0, weight_width):
            accum = tl.zeros((BLOCK_NO_HO_WO, BLOCK_CI), dtype=tl.float32)
            for c in range(0, out_c, BLOCK_CO):
                output_c_offset = c + tl.arange(0, BLOCK_CO)

                # caculate weight pointer to [out_c, *] out_grad pointer to [*, out_c], out_c as reduce dim
                curr_weight_pointer = (
                    weight_pointer
                    + (h * weight_height_stride + w * weight_width_stride)[None, :]
                    + (output_c_offset * weight_n_stride)[:, None]
                )
                weight_mask = (input_c_offset < input_c)[None, :] & (
                    output_c_offset < out_c
                )[:, None]

                curr_out_grad_pointer = (
                    out_grad_pointer + (output_c_offset * output_c_stride)[None, :]
                )
                out_grad_mask = (
                    (no_point_value < in_n)[:, None]
                    & (output_c_offset < out_c)[None, :]
                    & (output_height_point_value < out_height)[:, None]
                    & (output_width_point_value < out_width)[:, None]
                )

                curr_out_grad = tl.load(curr_out_grad_pointer, mask=out_grad_mask)
                curr_weight = tl.load(curr_weight_pointer, mask=weight_mask)
                accum += tl.dot(curr_out_grad, curr_weight, allow_tf32=False)

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
                input_pointer
                + (input_height_stride * input_height_offset)[:, None]
                + (input_width_stride * input_width_offset)[:, None]
            )
            input_mask = (
                (no_point_value < in_n)[:, None]
                & (input_c_offset < input_c)[None, :]
                & (0 <= input_height_offset)[:, None]
                & (input_height_offset < input_height)[:, None]
                & (0 <= input_width_offset)[:, None]
                & (input_width_offset < input_width)[:, None]
            )
            tl.atomic_add(curr_input_pointer, accum, input_mask)


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logging.debug("GEMS CONV2D")
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
            triton.cdiv(out_c, META["BLOCK_CO"]),
            groups,
        )
        conv2d_forward_kernel[grid](
            input,
            weight,
            output,
            in_n,
            input_height,
            input_width,
            out_c,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
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
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        ctx.save_for_backward(weight)

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.dilation = (dilation_height, dilation_width)

        ctx.weight_info = (int(out_c / groups), weight_c, weight_height, weight_width)
        ctx.input_info = (in_n, input_height, input_width)
        ctx.out_info = (out_height, out_width)

        ctx.device = input.device
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CONV2D VJP")
        (weight,) = ctx.saved_tensors
        # (out_c equals origin cout divide groups)
        out_c, weight_c, weight_height, weight_width = ctx.weight_info
        in_n, input_height, input_width = ctx.input_info
        out_height, out_width = ctx.out_info

        device = ctx.device
        groups = ctx.groups

        stride_height, stride_width = ctx.stride
        dilation_height, dilation_width = ctx.dilation

        input = torch.zeros(
            in_n,
            weight_c * groups,
            input_height,
            input_width,
            dtype=torch.float32,
            device=device,
        )

        grid = lambda meta: (
            triton.cdiv(in_n * out_height * out_width, meta["BLOCK_NO_HO_WO"]),
            groups,
            triton.cdiv(weight_c, meta["BLOCK_CI"]),
        )

        padding_height, padding_width = ctx.padding
        # return dx,None,None,None,None,None,None
        conv2d_backward_kernel[grid](
            input,
            out_grad,
            weight,
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

        return (
            input,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# todo test SymInt[2] of stride or padding
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)
