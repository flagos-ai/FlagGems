import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.conv2d import conv2d
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _pair(value):
    return value if isinstance(value, (list, tuple)) else (value, value)


def _conv_transpose2d_output_size(
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


def _dot_input_precision(input):
    if runtime.device.vendor_name != "nvidia":
        return "ieee"

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    allow_tf32 = bool(getattr(matmul_backend, "allow_tf32", False))
    if input.dtype == torch.float32 and allow_tf32:
        return "tf32"
    return "tf32x3"


def _conv_transpose2d_block_m(input, groups, weight_height):
    if runtime.device.vendor_name != "nvidia" and input.dtype == torch.float32:
        return 128
    use_fp32_4x4_tile = input.dtype == torch.float32 and weight_height == 4
    return 128 if groups > 1 or use_fp32_4x4_tile else 256


@libentry()
@triton.jit
def _conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_i_stride: tl.constexpr,
    weight_o_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_oh = m_offsets // out_width
    n_offsets = n_oh // out_height
    oh_offsets = n_oh % out_height
    ow_offsets = m_offsets % out_width

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    accum = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    input_pointer += (
        input_n_stride * n_offsets + input_c_stride * pid_group * in_per_group_c
    )[:, None]
    weight_pointer += weight_i_stride * pid_group * in_per_group_c

    for kh in range(weight_height):
        ih_numer = oh_offsets + padding_height - kh * dilation_height
        ih = ih_numer // stride_height
        ih_valid = (ih_numer % stride_height == 0) & (0 <= ih) & (ih < input_height)
        for kw in range(weight_width):
            iw_numer = ow_offsets + padding_width - kw * dilation_width
            iw = iw_numer // stride_width
            iw_valid = (iw_numer % stride_width == 0) & (0 <= iw) & (iw < input_width)
            spatial_mask = ih_valid & iw_valid & (n_offsets < in_n)

            for ci_start in range(0, in_per_group_c, BLOCK_CI):
                ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
                input_offsets = (
                    input_pointer
                    + (input_c_stride * ci_offsets)[None, :]
                    + (input_height_stride * ih)[:, None]
                    + (input_width_stride * iw)[:, None]
                )
                weight_offsets = (
                    weight_pointer
                    + (weight_i_stride * ci_offsets)[:, None]
                    + (weight_o_stride * co_offsets)[None, :]
                    + weight_height_stride * kh
                    + weight_width_stride * kw
                )
                input_mask = (
                    spatial_mask[:, None] & (ci_offsets < in_per_group_c)[None, :]
                )
                weight_mask = (
                    (kh < weight_height)
                    & (kw < weight_width)
                    & (ci_offsets < in_per_group_c)[:, None]
                    & (co_offsets < out_per_group_c)[None, :]
                )

                input_block = tl.load(input_offsets, mask=input_mask, other=0.0)
                weight_block = tl.load(weight_offsets, mask=weight_mask, other=0.0)
                accum += tl.dot(
                    input_block, weight_block, input_precision=INPUT_PRECISION
                )

    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + pid_group * out_per_group_c + co_offsets,
            mask=co_offsets < out_per_group_c,
            other=0.0,
        ).to(tl.float32)
        accum += bias[None, :]

    output_offsets = (
        output_pointer
        + (output_n_stride * n_offsets)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + co_offsets))[None, :]
        + (output_height_stride * oh_offsets)[:, None]
        + (output_width_stride * ow_offsets)[:, None]
    )
    output_mask = (m_offsets < in_n * out_height * out_width)[:, None] & (
        co_offsets < out_per_group_c
    )[None, :]
    tl.store(output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_stride2_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_i_stride: tl.constexpr,
    weight_o_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group_parity = tl.program_id(2)
    pid_group = pid_group_parity // 4
    parity = pid_group_parity - pid_group * 4
    parity_h = parity // 2
    parity_w = parity - parity_h * 2

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups
    out_height2: tl.constexpr = out_height // 2
    out_width2: tl.constexpr = out_width // 2

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_oh2 = m_offsets // out_width2
    n_offsets = n_oh2 // out_height2
    oh2_offsets = n_oh2 - n_offsets * out_height2
    ow2_offsets = m_offsets - n_oh2 * out_width2
    oh_offsets = oh2_offsets * 2 + parity_h
    ow_offsets = ow2_offsets * 2 + parity_w

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    accum = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    input_pointer += (
        input_n_stride * n_offsets + input_c_stride * pid_group * in_per_group_c
    )[:, None]
    weight_pointer += weight_i_stride * pid_group * in_per_group_c

    for kh_idx in range(2):
        kh = 1 - parity_h + kh_idx * 2
        ih = (oh_offsets + 1 - kh) // 2
        ih_valid = (kh < weight_height) & (0 <= ih) & (ih < input_height)
        for kw_idx in range(2):
            kw = 1 - parity_w + kw_idx * 2
            iw = (ow_offsets + 1 - kw) // 2
            spatial_mask = (
                ih_valid
                & (kw < weight_width)
                & (0 <= iw)
                & (iw < input_width)
                & (n_offsets < in_n)
            )

            for ci_start in range(0, in_per_group_c, BLOCK_CI):
                ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
                input_offsets = (
                    input_pointer
                    + (input_c_stride * ci_offsets)[None, :]
                    + (input_height_stride * ih)[:, None]
                    + (input_width_stride * iw)[:, None]
                )
                weight_offsets = (
                    weight_pointer
                    + (weight_i_stride * ci_offsets)[:, None]
                    + (weight_o_stride * co_offsets)[None, :]
                    + weight_height_stride * kh
                    + weight_width_stride * kw
                )
                input_mask = (
                    spatial_mask[:, None] & (ci_offsets < in_per_group_c)[None, :]
                )
                weight_mask = (
                    (kh < weight_height)
                    & (kw < weight_width)
                    & (ci_offsets < in_per_group_c)[:, None]
                    & (co_offsets < out_per_group_c)[None, :]
                )

                input_block = tl.load(input_offsets, mask=input_mask, other=0.0)
                weight_block = tl.load(weight_offsets, mask=weight_mask, other=0.0)
                accum += tl.dot(
                    input_block, weight_block, input_precision=INPUT_PRECISION
                )

    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + pid_group * out_per_group_c + co_offsets,
            mask=co_offsets < out_per_group_c,
            other=0.0,
        ).to(tl.float32)
        accum += bias[None, :]

    output_offsets = (
        output_pointer
        + (output_n_stride * n_offsets)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + co_offsets))[None, :]
        + (output_height_stride * oh_offsets)[:, None]
        + (output_width_stride * ow_offsets)[:, None]
    )
    output_mask = (m_offsets < in_n * out_height2 * out_width2)[:, None] & (
        co_offsets < out_per_group_c
    )[None, :]
    tl.store(output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_stride1_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_i_stride: tl.constexpr,
    weight_o_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_oh = m_offsets // out_width
    n_offsets = n_oh // out_height
    oh_offsets = n_oh - n_offsets * out_height
    ow_offsets = m_offsets - n_oh * out_width

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    accum = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    input_pointer += (
        input_n_stride * n_offsets + input_c_stride * pid_group * in_per_group_c
    )[:, None]
    weight_pointer += weight_i_stride * pid_group * in_per_group_c

    for kh in range(weight_height):
        ih = oh_offsets + padding_height - kh
        ih_valid = (0 <= ih) & (ih < input_height)
        for kw in range(weight_width):
            iw = ow_offsets + padding_width - kw
            spatial_mask = (
                ih_valid & (0 <= iw) & (iw < input_width) & (n_offsets < in_n)
            )

            for ci_start in range(0, in_per_group_c, BLOCK_CI):
                ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
                input_offsets = (
                    input_pointer
                    + (input_c_stride * ci_offsets)[None, :]
                    + (input_height_stride * ih)[:, None]
                    + (input_width_stride * iw)[:, None]
                )
                weight_offsets = (
                    weight_pointer
                    + (weight_i_stride * ci_offsets)[:, None]
                    + (weight_o_stride * co_offsets)[None, :]
                    + weight_height_stride * kh
                    + weight_width_stride * kw
                )
                input_mask = (
                    spatial_mask[:, None] & (ci_offsets < in_per_group_c)[None, :]
                )
                weight_mask = (ci_offsets < in_per_group_c)[:, None] & (
                    co_offsets < out_per_group_c
                )[None, :]

                input_block = tl.load(input_offsets, mask=input_mask, other=0.0)
                weight_block = tl.load(weight_offsets, mask=weight_mask, other=0.0)
                accum += tl.dot(
                    input_block, weight_block, input_precision=INPUT_PRECISION
                )

    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + pid_group * out_per_group_c + co_offsets,
            mask=co_offsets < out_per_group_c,
            other=0.0,
        ).to(tl.float32)
        accum += bias[None, :]

    output_offsets = (
        output_pointer
        + (output_n_stride * n_offsets)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + co_offsets))[None, :]
        + (output_height_stride * oh_offsets)[:, None]
        + (output_width_stride * ow_offsets)[:, None]
    )
    output_mask = (m_offsets < in_n * out_height * out_width)[:, None] & (
        co_offsets < out_per_group_c
    )[None, :]
    tl.store(output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _conv_transpose2d_stride2_3x3_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_i_stride: tl.constexpr,
    weight_o_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PARITY_H: tl.constexpr,
    PARITY_W: tl.constexpr,
    KH_BASE: tl.constexpr,
    KW_BASE: tl.constexpr,
    KH_COUNT: tl.constexpr,
    KW_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups
    out_height2: tl.constexpr = out_height // 2
    out_width2: tl.constexpr = out_width // 2

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_oh2 = m_offsets // out_width2
    n_offsets = n_oh2 // out_height2
    oh2_offsets = n_oh2 - n_offsets * out_height2
    ow2_offsets = m_offsets - n_oh2 * out_width2
    oh_offsets = oh2_offsets * 2 + PARITY_H
    ow_offsets = ow2_offsets * 2 + PARITY_W

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    accum = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    input_pointer += (
        input_n_stride * n_offsets + input_c_stride * pid_group * in_per_group_c
    )[:, None]
    weight_pointer += weight_i_stride * pid_group * in_per_group_c

    for kh_idx in range(KH_COUNT):
        kh = KH_BASE + kh_idx * 2
        ih = (oh_offsets + 1 - kh) // 2
        ih_valid = (0 <= ih) & (ih < input_height)
        for kw_idx in range(KW_COUNT):
            kw = KW_BASE + kw_idx * 2
            iw = (ow_offsets + 1 - kw) // 2
            spatial_mask = (
                ih_valid & (0 <= iw) & (iw < input_width) & (n_offsets < in_n)
            )

            for ci_start in range(0, in_per_group_c, BLOCK_CI):
                ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
                input_offsets = (
                    input_pointer
                    + (input_c_stride * ci_offsets)[None, :]
                    + (input_height_stride * ih)[:, None]
                    + (input_width_stride * iw)[:, None]
                )
                weight_offsets = (
                    weight_pointer
                    + (weight_i_stride * ci_offsets)[:, None]
                    + (weight_o_stride * co_offsets)[None, :]
                    + weight_height_stride * kh
                    + weight_width_stride * kw
                )
                input_mask = (
                    spatial_mask[:, None] & (ci_offsets < in_per_group_c)[None, :]
                )
                weight_mask = (ci_offsets < in_per_group_c)[:, None] & (
                    co_offsets < out_per_group_c
                )[None, :]
                input_block = tl.load(input_offsets, mask=input_mask, other=0.0)
                weight_block = tl.load(weight_offsets, mask=weight_mask, other=0.0)
                accum += tl.dot(
                    input_block, weight_block, input_precision=INPUT_PRECISION
                )

    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + pid_group * out_per_group_c + co_offsets,
            mask=co_offsets < out_per_group_c,
            other=0.0,
        ).to(tl.float32)
        accum += bias[None, :]

    output_offsets = (
        output_pointer
        + (output_n_stride * n_offsets)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + co_offsets))[None, :]
        + (output_height_stride * oh_offsets)[:, None]
        + (output_width_stride * ow_offsets)[:, None]
    )
    output_mask = (m_offsets < in_n * out_height2 * out_width2)[:, None] & (
        co_offsets < out_per_group_c
    )[None, :]
    tl.store(output_offsets, accum, mask=output_mask)


def _can_use_triton_forward(
    input, weight, bias, stride, padding, output_padding, groups, dilation
):
    if input.ndim != 4 or weight.ndim != 4 or (bias is not None and bias.ndim != 1):
        return False
    if input.device.type != runtime.device.name:
        return False
    if not input.is_floating_point():
        return False
    if input.dtype == torch.bfloat16 and not runtime.device.support_bf16:
        return False
    if weight.dtype != input.dtype or (bias is not None and bias.dtype != input.dtype):
        return False
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    output_padding_height, output_padding_width = output_padding
    dilation_height, dilation_width = dilation
    if min(stride_height, stride_width, dilation_height, dilation_width) <= 0:
        return False
    if (
        min(padding_height, padding_width, output_padding_height, output_padding_width)
        < 0
    ):
        return False
    if input.shape[1] != weight.shape[0] or input.shape[1] % groups != 0:
        return False
    if bias is not None and bias.shape[0] != weight.shape[1] * groups:
        return False
    return True


def _can_use_stride2_kernel(stride, padding, dilation, out_height, out_width, weight):
    return (
        stride == (2, 2)
        and padding == (1, 1)
        and dilation == (1, 1)
        and out_height % 2 == 0
        and out_width % 2 == 0
        and 3 <= weight.shape[2] <= 4
        and 3 <= weight.shape[3] <= 4
    )


def _can_use_stride1_kernel(stride, dilation, output_padding):
    return stride == (1, 1) and dilation == (1, 1) and output_padding == (0, 0)


def _conv_transpose2d_stride1_via_conv2d(
    input, weight, bias, padding, dilation, groups
):
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation
    input_c, out_per_group_c, weight_height, weight_width = weight.shape
    in_per_group_c = input_c // groups
    out_c = out_per_group_c * groups
    conv_padding = (
        dilation_height * (weight_height - 1) - padding_height,
        dilation_width * (weight_width - 1) - padding_width,
    )
    conv_weight = (
        weight.reshape(
            groups, in_per_group_c, out_per_group_c, weight_height, weight_width
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(out_c, in_per_group_c, weight_height, weight_width)
        .flip(-1, -2)
        .contiguous()
    )
    return conv2d(
        input,
        conv_weight,
        bias=bias,
        stride=1,
        padding=conv_padding,
        dilation=dilation,
        groups=groups,
    )


@libentry()
@triton.jit
def _conv2d_nobias_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    weight_n_stride: tl.constexpr,
    weight_c_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    output_n_stride: tl.constexpr,
    output_c_stride: tl.constexpr,
    output_height_stride: tl.constexpr,
    output_width_stride: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_oh = m_offsets // out_width
    n_offsets = n_oh // out_height
    oh_offsets = n_oh - n_offsets * out_height
    ow_offsets = m_offsets - n_oh * out_width
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_pointer += (
        input_n_stride * n_offsets + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (weight_n_stride * (pid_group * out_per_group_c + co_offsets))[
        None, :
    ]

    accum = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)
    for kh in range(weight_height):
        ih = kh * dilation_height - padding_height + stride_height * oh_offsets
        ih_mask = (0 <= ih) & (ih < input_height)
        for kw in range(weight_width):
            iw = kw * dilation_width - padding_width + stride_width * ow_offsets
            spatial_mask = (n_offsets < in_n) & ih_mask & (0 <= iw) & (iw < input_width)
            for ci_start in range(0, weight_c, BLOCK_CI):
                ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
                input_offsets = (
                    input_pointer
                    + (input_c_stride * ci_offsets)[None, :]
                    + (input_height_stride * ih)[:, None]
                    + (input_width_stride * iw)[:, None]
                )
                weight_offsets = (
                    weight_pointer
                    + (weight_c_stride * ci_offsets)[:, None]
                    + weight_height_stride * kh
                    + weight_width_stride * kw
                )
                input_mask = spatial_mask[:, None] & (ci_offsets < weight_c)[None, :]
                weight_mask = (ci_offsets < weight_c)[:, None] & (
                    co_offsets < out_per_group_c
                )[None, :]
                input_block = tl.load(input_offsets, mask=input_mask, other=0.0)
                weight_block = tl.load(weight_offsets, mask=weight_mask, other=0.0)
                accum += tl.dot(
                    input_block, weight_block, input_precision=INPUT_PRECISION
                )

    output_offsets = (
        output_pointer
        + (output_n_stride * n_offsets)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + co_offsets))[None, :]
        + (output_height_stride * oh_offsets)[:, None]
        + (output_width_stride * ow_offsets)[:, None]
    )
    output_mask = (m_offsets < in_n * out_height * out_width)[:, None] & (
        co_offsets < out_per_group_c
    )[None, :]
    tl.store(output_offsets, accum, mask=output_mask)


@libentry()
@triton.jit
def _zero_kernel(output_pointer, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        output_pointer + offsets,
        tl.zeros((BLOCK_SIZE,), dtype=tl.float32),
        mask=offsets < n_elements,
    )


@libentry()
@triton.jit
def _conv_transpose2d_backward_weight_atomic_kernel(
    input_pointer,
    grad_output_pointer,
    grad_weight_pointer,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    grad_output_n_stride: tl.constexpr,
    grad_output_c_stride: tl.constexpr,
    grad_output_height_stride: tl.constexpr,
    grad_output_width_stride: tl.constexpr,
    grad_weight_i_stride: tl.constexpr,
    grad_weight_o_stride: tl.constexpr,
    grad_weight_height_stride: tl.constexpr,
    grad_weight_width_stride: tl.constexpr,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_CI_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ci_hw = tl.program_id(1)
    pid_co_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups
    co_blocks: tl.constexpr = tl.cdiv(out_per_group_c, BLOCK_CO)
    pid_group = pid_co_group // co_blocks
    pid_co = pid_co_group - pid_group * co_blocks
    input_spatial: tl.constexpr = input_height * input_width
    reduce_size: tl.constexpr = in_n * input_spatial

    reduce_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_offsets = reduce_offsets // input_spatial
    spatial_offsets = reduce_offsets - n_offsets * input_spatial
    ih_offsets = spatial_offsets // input_width
    iw_offsets = spatial_offsets - ih_offsets * input_width

    ci_hw_offsets = pid_ci_hw * BLOCK_CI_HW + tl.arange(0, BLOCK_CI_HW)
    ci_kh = ci_hw_offsets // weight_width
    ci_offsets = ci_kh // weight_height
    kh_offsets = ci_kh - ci_offsets * weight_height
    kw_offsets = ci_hw_offsets - ci_kh * weight_width
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_offsets = (
        input_pointer
        + (input_n_stride * n_offsets)[None, :]
        + (input_c_stride * (pid_group * in_per_group_c + ci_offsets))[:, None]
        + (input_height_stride * ih_offsets)[None, :]
        + (input_width_stride * iw_offsets)[None, :]
    )
    input_mask = (reduce_offsets < reduce_size)[None, :] & (
        ci_offsets < in_per_group_c
    )[:, None]
    input_block = tl.load(input_offsets, mask=input_mask, other=0.0)

    oh_offsets = (
        ih_offsets[None, :] * stride_height
        - padding_height
        + kh_offsets[:, None] * dilation_height
    )
    ow_offsets = (
        iw_offsets[None, :] * stride_width
        - padding_width
        + kw_offsets[:, None] * dilation_width
    )

    accum = tl.zeros((BLOCK_CI_HW, BLOCK_CO), dtype=tl.float32)
    for co_inner in range(BLOCK_CO):
        co = pid_co * BLOCK_CO + co_inner
        grad_output_offsets = (
            grad_output_pointer
            + (grad_output_n_stride * n_offsets)[None, :]
            + grad_output_c_stride * (pid_group * out_per_group_c + co)
            + grad_output_height_stride * oh_offsets
            + grad_output_width_stride * ow_offsets
        )
        grad_output_mask = (
            (reduce_offsets < reduce_size)[None, :]
            & (0 <= oh_offsets)
            & (oh_offsets < out_height)
            & (0 <= ow_offsets)
            & (ow_offsets < out_width)
            & (co < out_per_group_c)
        )
        grad_output_block = tl.load(
            grad_output_offsets, mask=grad_output_mask, other=0.0
        )
        partial = tl.sum(
            input_block.to(tl.float32) * grad_output_block.to(tl.float32), axis=1
        )
        accum += tl.where(co_offsets[None, :] == co, partial[:, None], 0.0)

    grad_weight_offsets = (
        grad_weight_pointer
        + (grad_weight_i_stride * (pid_group * in_per_group_c + ci_offsets))[:, None]
        + (grad_weight_o_stride * co_offsets)[None, :]
        + (grad_weight_height_stride * kh_offsets)[:, None]
        + (grad_weight_width_stride * kw_offsets)[:, None]
    )
    grad_weight_mask = (
        (ci_offsets < in_per_group_c)
        & (kh_offsets < weight_height)
        & (kw_offsets < weight_width)
    )[:, None] & (co_offsets < out_per_group_c)[None, :]
    tl.atomic_add(grad_weight_offsets, accum, sem="relaxed", mask=grad_weight_mask)


@libentry()
@triton.jit
def _conv_transpose2d_backward_input_kernel(
    grad_output_pointer,
    weight_pointer,
    grad_input_pointer,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    grad_output_n_stride: tl.constexpr,
    grad_output_c_stride: tl.constexpr,
    grad_output_height_stride: tl.constexpr,
    grad_output_width_stride: tl.constexpr,
    weight_i_stride: tl.constexpr,
    weight_o_stride: tl.constexpr,
    weight_height_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    grad_input_n_stride: tl.constexpr,
    grad_input_c_stride: tl.constexpr,
    grad_input_height_stride: tl.constexpr,
    grad_input_width_stride: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_ci = tl.program_id(1)
    pid_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_ih = m_offsets // input_width
    n_offsets = n_ih // input_height
    ih_offsets = n_ih - n_offsets * input_height
    iw_offsets = m_offsets - n_ih * input_width
    ci_offsets = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)

    accum = tl.zeros((BLOCK_M, BLOCK_CI), dtype=tl.float32)
    for kh in range(weight_height):
        oh = ih_offsets * stride_height - padding_height + kh * dilation_height
        oh_valid = (0 <= oh) & (oh < out_height)
        for kw in range(weight_width):
            ow = iw_offsets * stride_width - padding_width + kw * dilation_width
            spatial_mask = (n_offsets < in_n) & oh_valid & (0 <= ow) & (ow < out_width)
            for co_start in range(0, out_per_group_c, BLOCK_CO):
                co_offsets = co_start + tl.arange(0, BLOCK_CO)
                grad_output_offsets = (
                    grad_output_pointer
                    + (grad_output_n_stride * n_offsets)[:, None]
                    + (
                        grad_output_c_stride
                        * (pid_group * out_per_group_c + co_offsets)
                    )[None, :]
                    + (grad_output_height_stride * oh)[:, None]
                    + (grad_output_width_stride * ow)[:, None]
                )
                weight_offsets = (
                    weight_pointer
                    + (weight_i_stride * (pid_group * in_per_group_c + ci_offsets))[
                        :, None
                    ]
                    + (weight_o_stride * co_offsets)[None, :]
                    + weight_height_stride * kh
                    + weight_width_stride * kw
                )
                grad_output_mask = (
                    spatial_mask[:, None] & (co_offsets < out_per_group_c)[None, :]
                )
                weight_mask = (ci_offsets < in_per_group_c)[:, None] & (
                    co_offsets < out_per_group_c
                )[None, :]
                grad_output_block = tl.load(
                    grad_output_offsets, mask=grad_output_mask, other=0.0
                )
                weight_block = tl.load(weight_offsets, mask=weight_mask, other=0.0)
                accum += tl.dot(
                    grad_output_block,
                    tl.trans(weight_block),
                    input_precision=INPUT_PRECISION,
                )

    grad_input_offsets = (
        grad_input_pointer
        + (grad_input_n_stride * n_offsets)[:, None]
        + (grad_input_c_stride * (pid_group * in_per_group_c + ci_offsets))[None, :]
        + (grad_input_height_stride * ih_offsets)[:, None]
        + (grad_input_width_stride * iw_offsets)[:, None]
    )
    grad_input_mask = (m_offsets < in_n * input_height * input_width)[:, None] & (
        ci_offsets < in_per_group_c
    )[None, :]
    tl.store(grad_input_offsets, accum, mask=grad_input_mask)


@libentry()
@triton.jit
def _conv_transpose2d_backward_weight_kernel(
    input_pointer,
    grad_output_pointer,
    grad_weight_pointer,
    input_n_stride: tl.constexpr,
    input_c_stride: tl.constexpr,
    input_height_stride: tl.constexpr,
    input_width_stride: tl.constexpr,
    grad_output_n_stride: tl.constexpr,
    grad_output_c_stride: tl.constexpr,
    grad_output_height_stride: tl.constexpr,
    grad_output_width_stride: tl.constexpr,
    grad_weight_i_stride: tl.constexpr,
    grad_weight_o_stride: tl.constexpr,
    grad_weight_height_stride: tl.constexpr,
    grad_weight_width_stride: tl.constexpr,
    in_n: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    input_c: tl.constexpr,
    out_c: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_CI_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_ci_hw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    out_per_group_c: tl.constexpr = out_c // groups
    in_per_group_c: tl.constexpr = input_c // groups
    input_spatial: tl.constexpr = input_height * input_width
    reduce_size: tl.constexpr = in_n * input_spatial

    ci_hw_offsets = pid_ci_hw * BLOCK_CI_HW + tl.arange(0, BLOCK_CI_HW)
    ci_kh = ci_hw_offsets // weight_width
    ci_offsets = ci_kh // weight_height
    kh_offsets = ci_kh - ci_offsets * weight_height
    kw_offsets = ci_hw_offsets - ci_kh * weight_width
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    accum = tl.zeros((BLOCK_CI_HW, BLOCK_CO), dtype=tl.float32)
    for start in range(0, reduce_size, BLOCK_N):
        reduce_offsets = start + tl.arange(0, BLOCK_N)
        n_offsets = reduce_offsets // input_spatial
        spatial_offsets = reduce_offsets - n_offsets * input_spatial
        ih_offsets = spatial_offsets // input_width
        iw_offsets = spatial_offsets - ih_offsets * input_width

        input_offsets = (
            input_pointer
            + (input_n_stride * n_offsets)[None, :]
            + (input_c_stride * (pid_group * in_per_group_c + ci_offsets))[:, None]
            + (input_height_stride * ih_offsets)[None, :]
            + (input_width_stride * iw_offsets)[None, :]
        )
        oh_offsets = (
            ih_offsets[None, :] * stride_height
            - padding_height
            + kh_offsets[:, None] * dilation_height
        )
        ow_offsets = (
            iw_offsets[None, :] * stride_width
            - padding_width
            + kw_offsets[:, None] * dilation_width
        )
        input_mask = (reduce_offsets < reduce_size)[None, :] & (
            ci_offsets < in_per_group_c
        )[:, None]
        input_block = tl.load(input_offsets, mask=input_mask, other=0.0)
        for co_inner in range(BLOCK_CO):
            co = pid_co * BLOCK_CO + co_inner
            grad_output_offsets = (
                grad_output_pointer
                + (grad_output_n_stride * n_offsets)[None, :]
                + grad_output_c_stride * (pid_group * out_per_group_c + co)
                + grad_output_height_stride * oh_offsets
                + grad_output_width_stride * ow_offsets
            )
            grad_output_mask = (
                (reduce_offsets < reduce_size)[None, :]
                & (0 <= oh_offsets)
                & (oh_offsets < out_height)
                & (0 <= ow_offsets)
                & (ow_offsets < out_width)
                & (co < out_per_group_c)
            )
            grad_output_block = tl.load(
                grad_output_offsets,
                mask=grad_output_mask,
                other=0.0,
            )
            partial = tl.sum(
                input_block.to(tl.float32) * grad_output_block.to(tl.float32), axis=1
            )
            accum += tl.where(co_offsets[None, :] == co, partial[:, None], 0.0)

    grad_weight_offsets = (
        grad_weight_pointer
        + (grad_weight_i_stride * (pid_group * in_per_group_c + ci_offsets))[:, None]
        + (grad_weight_o_stride * co_offsets)[None, :]
        + (grad_weight_height_stride * kh_offsets)[:, None]
        + (grad_weight_width_stride * kw_offsets)[:, None]
    )
    grad_weight_mask = (
        (ci_offsets < in_per_group_c)
        & (kh_offsets < weight_height)
        & (kw_offsets < weight_width)
    )[:, None] & (co_offsets < out_per_group_c)[None, :]
    tl.store(grad_weight_offsets, accum, mask=grad_weight_mask)


@libentry()
@triton.jit
def _conv_transpose2d_backward_bias_kernel(
    grad_output_pointer,
    grad_bias_pointer,
    total: tl.constexpr,
    out_c: tl.constexpr,
    grad_output_n_stride: tl.constexpr,
    grad_output_c_stride: tl.constexpr,
    grad_output_height_stride: tl.constexpr,
    grad_output_width_stride: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    co = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    accum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, total, BLOCK_N):
        idx = start + offsets
        n_offsets = idx // (out_height * out_width)
        spatial_offsets = idx - n_offsets * out_height * out_width
        oh_offsets = spatial_offsets // out_width
        ow_offsets = spatial_offsets - oh_offsets * out_width
        grad_output_offsets = (
            grad_output_pointer
            + grad_output_n_stride * n_offsets
            + grad_output_c_stride * co
            + grad_output_height_stride * oh_offsets
            + grad_output_width_stride * ow_offsets
        )
        accum += tl.load(grad_output_offsets, mask=idx < total, other=0.0)
    tl.store(grad_bias_pointer + co, tl.sum(accum, axis=0), mask=co < out_c)


def _conv_transpose2d_forward(
    input, weight, bias, stride, padding, output_padding, groups, dilation
):
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    output_padding_height, output_padding_width = output_padding
    dilation_height, dilation_width = dilation

    in_n, input_c, input_height, input_width = input.shape
    _, out_per_group_c, weight_height, weight_width = weight.shape
    out_c = out_per_group_c * groups
    out_height = _conv_transpose2d_output_size(
        input_height,
        weight_height,
        stride_height,
        padding_height,
        output_padding_height,
        dilation_height,
    )
    out_width = _conv_transpose2d_output_size(
        input_width,
        weight_width,
        stride_width,
        padding_width,
        output_padding_width,
        dilation_width,
    )
    output = torch.empty(
        (in_n, out_c, out_height, out_width), device=input.device, dtype=input.dtype
    )

    if _can_use_stride2_kernel(
        stride, padding, dilation, out_height, out_width, weight
    ):
        block_m = _conv_transpose2d_block_m(input, groups, weight_height)
        use_fp32_4x4_tile = input.dtype == torch.float32 and weight_height == 4
        block_co = 16 if groups > 1 else 32
        num_warps = 4 if groups > 1 or use_fp32_4x4_tile else 8
        if input.dtype == torch.float32 and weight_height == 3 and weight_width == 3:
            parity_configs = (
                (0, 0, 1, 1, 1, 1),
                (0, 1, 1, 0, 1, 2),
                (1, 0, 0, 1, 2, 1),
                (1, 1, 0, 0, 2, 2),
            )
            grid = lambda META: (
                triton.cdiv(
                    in_n * (out_height // 2) * (out_width // 2), META["BLOCK_M"]
                ),
                triton.cdiv(out_per_group_c, META["BLOCK_CO"]),
                groups,
            )
            for (
                parity_h,
                parity_w,
                kh_base,
                kw_base,
                kh_count,
                kw_count,
            ) in parity_configs:
                _conv_transpose2d_stride2_3x3_forward_kernel[grid](
                    input,
                    weight,
                    input if bias is None else bias,
                    output,
                    in_n,
                    input_height,
                    input_width,
                    input_c,
                    out_c,
                    out_height,
                    out_width,
                    *input.stride(),
                    *weight.stride(),
                    *output.stride(),
                    groups,
                    HAS_BIAS=bias is not None,
                    PARITY_H=parity_h,
                    PARITY_W=parity_w,
                    KH_BASE=kh_base,
                    KW_BASE=kw_base,
                    KH_COUNT=kh_count,
                    KW_COUNT=kw_count,
                    BLOCK_M=block_m,
                    BLOCK_CI=32,
                    BLOCK_CO=block_co,
                    INPUT_PRECISION=_dot_input_precision(input),
                    num_warps=num_warps,
                )
            return output

        grid = lambda META: (
            triton.cdiv(in_n * (out_height // 2) * (out_width // 2), META["BLOCK_M"]),
            triton.cdiv(out_per_group_c, META["BLOCK_CO"]),
            groups * 4,
        )
        _conv_transpose2d_stride2_forward_kernel[grid](
            input,
            weight,
            input if bias is None else bias,
            output,
            in_n,
            input_height,
            input_width,
            input_c,
            out_c,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            weight_height,
            weight_width,
            groups,
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_CI=32,
            BLOCK_CO=block_co,
            INPUT_PRECISION=_dot_input_precision(input),
            num_warps=num_warps,
        )
        return output

    if _can_use_stride1_kernel(stride, dilation, output_padding):
        return _conv_transpose2d_stride1_via_conv2d(
            input, weight, bias, padding, dilation, groups
        )

    grid = lambda META: (
        triton.cdiv(in_n * out_height * out_width, META["BLOCK_M"]),
        triton.cdiv(out_per_group_c, META["BLOCK_CO"]),
        groups,
    )
    _conv_transpose2d_forward_kernel[grid](
        input,
        weight,
        input if bias is None else bias,
        output,
        in_n,
        input_height,
        input_width,
        input_c,
        out_c,
        out_height,
        out_width,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_height,
        weight_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        dilation_height,
        dilation_width,
        groups,
        HAS_BIAS=bias is not None,
        BLOCK_M=64,
        BLOCK_CI=32,
        BLOCK_CO=32,
        INPUT_PRECISION=_dot_input_precision(input),
        num_warps=4,
    )
    return output


def _conv_transpose2d_backward(
    grad_output,
    input,
    weight,
    grad_input_bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
    has_bias,
    needs_input_grad,
    needs_weight_grad,
    needs_bias_grad,
):
    del output_padding
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    in_n, input_c, input_height, input_width = input.shape
    _, out_per_group_c, weight_height, weight_width = weight.shape
    out_c = out_per_group_c * groups
    out_height, out_width = grad_output.shape[2:]

    grad_input = None
    if needs_input_grad:
        if (
            groups == 1
            and weight_height == 3
            and weight_width == 3
            and stride == (2, 2)
            and grad_output.dtype != torch.float32
        ):
            grad_input = conv2d(
                grad_output,
                weight,
                bias=grad_input_bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            grad_input = torch.empty_like(input)
            block_m = 128
            if groups > 1:
                block_co = 16
            elif weight_height == 4 and weight_width == 4:
                block_co = 64
            else:
                block_co = 32
            grid = lambda META: (
                triton.cdiv(in_n * input_height * input_width, META["BLOCK_M"]),
                triton.cdiv(input_c // groups, META["BLOCK_CO"]),
                groups,
            )
            _conv2d_nobias_kernel[grid](
                grad_output,
                weight,
                grad_input,
                in_n,
                out_height,
                out_width,
                input_c,
                input_height,
                input_width,
                *grad_output.stride(),
                *weight.stride(),
                *grad_input.stride(),
                out_per_group_c,
                weight_height,
                weight_width,
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                dilation_height,
                dilation_width,
                groups,
                BLOCK_M=block_m,
                BLOCK_CI=32,
                BLOCK_CO=block_co,
                INPUT_PRECISION=_dot_input_precision(grad_output),
                num_warps=4,
            )

    grad_weight = None
    if needs_weight_grad:
        grad_weight = torch.empty_like(weight)
        _zero_kernel[(triton.cdiv(grad_weight.numel(), 1024),)](
            grad_weight, grad_weight.numel(), BLOCK_SIZE=1024
        )
        weight_block_n = 128 if groups == 1 and stride == (2, 2) else 256
        grad_weight_grid = lambda META: (
            triton.cdiv(in_n * input_height * input_width, META["BLOCK_N"]),
            triton.cdiv(
                (input_c // groups) * weight_height * weight_width,
                META["BLOCK_CI_HW"],
            ),
            triton.cdiv(out_per_group_c, META["BLOCK_CO"]) * groups,
        )
        _conv_transpose2d_backward_weight_atomic_kernel[grad_weight_grid](
            input,
            grad_output,
            grad_weight,
            *input.stride(),
            *grad_output.stride(),
            *grad_weight.stride(),
            in_n,
            input_height,
            input_width,
            input_c,
            out_c,
            out_height,
            out_width,
            weight_height,
            weight_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            groups,
            BLOCK_CI_HW=16,
            BLOCK_CO=16,
            BLOCK_N=weight_block_n,
            num_warps=4,
        )

    grad_bias = None
    if has_bias and needs_bias_grad:
        grad_bias = torch.empty(
            (out_c,), device=grad_output.device, dtype=grad_output.dtype
        )
        total = in_n * out_height * out_width
        _conv_transpose2d_backward_bias_kernel[(out_c,)](
            grad_output,
            grad_bias,
            total,
            out_c,
            *grad_output.stride(),
            out_height,
            out_width,
            BLOCK_N=1024,
            num_warps=8,
        )
    return grad_input, grad_weight, grad_bias


class ConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, stride, padding, output_padding, groups, dilation
    ):
        output = _conv_transpose2d_forward(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
        ctx.save_for_backward(input, weight)
        ctx.grad_input_bias = torch.zeros(
            (input.shape[1],), device=input.device, dtype=input.dtype
        )
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        ctx.dilation = dilation
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = _conv_transpose2d_backward(
            grad_output,
            input,
            weight,
            ctx.grad_input_bias,
            ctx.stride,
            ctx.padding,
            ctx.output_padding,
            ctx.groups,
            ctx.dilation,
            ctx.has_bias,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


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
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)
    if not _can_use_triton_forward(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    ):
        raise NotImplementedError(
            "flag_gems.conv_transpose2d currently supports floating-point 4D "
            f"tensors on {runtime.device.name} with matching 4D weights."
        )
    return ConvTranspose2d.apply(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )
