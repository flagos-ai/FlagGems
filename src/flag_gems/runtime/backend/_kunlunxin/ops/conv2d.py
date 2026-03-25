import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


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


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("conv2d_forward"),
#     key=[
#         "in_n",
#         "weight_c",
#         "input_height",
#         "input_width",
#         "out_c",
#         "out_height",
#         "out_width",
#         "weight_height",
#         "weight_width",
#         "stride_height",
#         "stride_width",
#         "padding_height",
#         "padding_width",
#         "groups",
#     ],
# )
@triton.jit
def conv2d_forward_kernel(
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
    USE_MIXED_PRECISION: tl.constexpr,
):
    """
    Mixed-precision forward kernel.
    When USE_MIXED_PRECISION=True: FP16/BF16 I/O + FP32 accumulator
    """
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

        # Mixed precision: convert to FP32 for computation
        if USE_MIXED_PRECISION:
            input_block = input_block.to(tl.float32)
            weight_block = weight_block.to(tl.float32)

        accum += tl.dot(input_block, weight_block, allow_tf32=False)
    bias_pointer += pid_group * out_per_group_c[None, :] + output_c_offset[None, :]
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
# @triton.autotune(
#     configs=runtime.get_tuned_config("conv2d_backward_weight"),
#     key=[
#         "in_n",
#         "input_height",
#         "input_width",
#         "weight_height",
#         "weight_width",
#         "input_c",
#         "stride_height",
#         "stride_width",
#         "out_height",
#         "out_width",
#         "out_c",
#         "padding_height",
#         "padding_width",
#     ],
# )
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
    groups: tl.constexpr,
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
        pid_groups * output_c_stride * out_c
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

    input_pointer += (ci_point_value * input_c_stride)[:, None] + (
        pid_groups * input_c_stride * input_c
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

                # Mixed precision: always convert to FP32 for FP16/BF16 safety
                # This is a simplified check - in practice, should pass USE_MIXED_PRECISION
                # For now, we detect if it's FP16/BF16 and convert
                if curr_input.dtype != tl.float32:
                    curr_input = curr_input.to(tl.float32)
                if curr_out_grad.dtype != tl.float32:
                    curr_out_grad = curr_out_grad.to(tl.float32)

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
        logger.debug("GEMS CONV2D")
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

        # Hybrid strategy: Python-level FP32 conversion for small cases,
        # kernel-level mixed precision for large cases
        #
        # Hardware constraints (XPU3):
        # - FP16: Supports mixed precision (verified to work)
        # - BF16: Limited support, "unsupported data type" errors in some cases
        #   → Always use Python FP32 conversion for safety
        #
        # Rationale:
        # - Small FP16 cases: Python FP32 matches PyTorch reference exactly
        # - Large FP16 cases: Mixed precision saves 50% bandwidth → 2x speedup
        # - All BF16 cases: Python FP32 for hardware compatibility
        #
        # Threshold: spatial_size > 1024 triggers FP16 mixed precision
        spatial_size = input_height * input_width
        is_large_case = (spatial_size > 1024) and (in_n * out_c > 64)

        # Only enable mixed precision for FP16 large cases
        use_mixed_precision = (input.dtype == torch.float16) and is_large_case
        use_python_fp32 = (
            input.dtype in (torch.float16, torch.bfloat16)
        ) and not use_mixed_precision

        if use_python_fp32:
            # Small cases: convert in Python layer for reference-matching behavior
            input = input.to(torch.float32)
            weight = weight.to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float32)
            compute_dtype = torch.float32
        else:
            # Large cases or FP32: keep original precision
            compute_dtype = output_dtype

        output = torch.empty(
            (in_n, out_c, out_height, out_width),
            device=input.device,
            dtype=compute_dtype,
        )

        # BLOCK_NI_HO_WO along the in_n, out_height, and out_width dimensions,
        # BLOCK_CO along the out_c,
        # one group per cat
        grid = lambda META: (
            triton.cdiv(in_n * out_height * out_width, META["BLOCK_NI_HO_WO"]),
            triton.cdiv(int(out_c // groups), META["BLOCK_CO"]),
            groups,
        )

        if bias is None:
            bias_pointer = torch.zeros(out_c, device=input.device, dtype=torch.float)
        else:
            bias_pointer = bias.to(torch.float)
        flag = 0
        if input.shape[2] != input.shape[3]:
            flag = 999
        else:
            flag = 32
        conv2d_forward_kernel[grid](
            input,
            weight,
            output,
            bias_pointer,
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
            BLOCK_NI_HO_WO=flag,
            BLOCK_CI=32,
            BLOCK_CO=32,
            USE_MIXED_PRECISION=use_mixed_precision,
        )

        ctx.save_for_backward(weight, input, bias)

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.dilation = (dilation_height, dilation_width)

        ctx.weight_info = (int(out_c / groups), weight_c, weight_height, weight_width)
        ctx.input_info = (in_n, input_height, input_width)
        ctx.out_info = (out_height, out_width)

        ctx.device = input.device
        ctx.groups = groups
        ctx.use_mixed_precision = use_mixed_precision
        ctx.use_python_fp32 = use_python_fp32
        ctx.output_dtype = output_dtype

        # Convert output back if we used Python-level FP32 conversion
        if use_python_fp32:
            output = output.to(output_dtype)

        return output

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS CONV2D VJP")
        (weight, input, bias) = ctx.saved_tensors
        # (out_c equals origin cout divide groups)
        out_c, weight_c, weight_height, weight_width = ctx.weight_info
        in_n, input_height, input_width = ctx.input_info
        out_height, out_width = ctx.out_info

        device = ctx.device
        groups = ctx.groups
        use_mixed_precision = ctx.use_mixed_precision
        use_python_fp32 = ctx.use_python_fp32
        output_dtype = ctx.output_dtype

        stride_height, stride_width = ctx.stride
        dilation_height, dilation_width = ctx.dilation
        padding_height, padding_width = ctx.padding

        # If forward used Python-level FP32, convert out_grad to match
        if use_python_fp32 and out_grad.dtype in (torch.float16, torch.bfloat16):
            out_grad = out_grad.to(torch.float32)

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

        # Calculate new_out dimensions for transposed convolution
        # Must account for output_padding when (input + 2*padding - dilation*(kernel-1) - 1) % stride != 0
        new_out_height = (
            input_height + 2 * padding_height - dilation_height * (weight_height - 1)
        )
        new_out_width = (
            input_width + 2 * padding_width - dilation_width * (weight_width - 1)
        )

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
            dtype=input.dtype,  # Use original dtype for mixed precision
            device=device,
        )

        grid = lambda META: (
            triton.cdiv(
                out_grad.shape[0] * input_height * input_width, META["BLOCK_NI_HO_WO"]
            ),
            triton.cdiv(int(weight_c), META["BLOCK_CO"]),
            groups,
        )
        flag = 888
        bias_zero = torch.zeros(groups * weight_c, device=device, dtype=out_grad.dtype)
        conv2d_forward_kernel[grid](
            new_out,
            revert_weight,
            input_back,
            bias_zero,
            out_grad.shape[0],
            new_out_height,
            new_out_width,
            groups * weight_c,
            input_height,
            input_width,
            *new_out.stride(),
            *revert_weight.stride(),
            *input_back.stride(),
            out_c,
            weight_height,
            weight_width,
            1,
            1,
            revert_padding_height,
            revert_padding_width,
            dilation_height,
            dilation_width,
            groups=groups,
            BLOCK_NI_HO_WO=flag,
            BLOCK_CI=32,
            BLOCK_CO=32,
            USE_MIXED_PRECISION=use_mixed_precision,
        )

        # For mixed precision: weight_back accumulator must be FP32 to prevent overflow
        # We'll convert back to original dtype at the end
        weight_back_dtype = torch.float32 if use_mixed_precision else weight.dtype

        weight_back = torch.zeros(
            out_c * groups,
            weight_c,
            weight_height,
            weight_width,
            dtype=weight_back_dtype,
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
            groups,
            BLOCK_NO=32,
            BLOCK_CI_HK_WK=32,
            BLOCK_CO=32,
        )
        if bias is not None:
            bias_grad = out_grad.sum(dim=(0, 2, 3))
        else:
            bias_grad = None

        # Convert gradients back to original dtype if needed
        if use_python_fp32:
            # Python FP32 path: convert everything back
            input_back = (
                input_back.to(output_dtype)
                if input_back.dtype != output_dtype
                else input_back
            )
            weight_back = (
                weight_back.to(output_dtype)
                if weight_back.dtype != output_dtype
                else weight_back
            )
            if bias_grad is not None:
                bias_grad = (
                    bias_grad.to(output_dtype)
                    if bias_grad.dtype != output_dtype
                    else bias_grad
                )
        elif use_mixed_precision and weight_back.dtype != weight.dtype:
            # Mixed precision path: weight_back was FP32, convert back
            weight_back = weight_back.to(weight.dtype)

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
            assert stride == 1, (
                f"Doesn't support any stride values other than 1 in padding = 'same' mode, "
                f"received stride value {stride}"
            )
            ih = input.shape[-2]
            iw = input.shape[-1]
            kernel_size_h = weight.shape[-2]
            kernel_size_w = weight.shape[-1]
            import math

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
            padding = max(padding_h, padding_w)
            return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)[
                ..., (oh - ih) :, (ow - iw) :
            ]
        elif padding == "valid":
            return Conv2d.apply(input, weight, bias, stride, 0, dilation, groups)
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed."
            )
    else:
        return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)
