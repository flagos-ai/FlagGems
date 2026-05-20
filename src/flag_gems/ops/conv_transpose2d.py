import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """ConvTranspose2d output extent along one spatial axis."""
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv_transpose2d"),
    key=[
        "batch_size",
        "in_channels",
        "input_height",
        "input_width",
        "out_channels",
        "out_height",
        "out_width",
        "kernel_height",
        "kernel_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    batch_size,
    input_height,
    input_width,
    out_channels,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_h_stride,
    input_w_stride,
    weight_ic_stride,
    weight_oc_stride,
    weight_h_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_h_stride,
    output_w_stride,
    in_channels: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_N_OHW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    Forward kernel for 2D transposed convolution.

    Layout:
      input  : (N, in_channels,                 H_in,  W_in)
      weight : (in_channels, out_channels/groups, K_h,  K_w)
      output : (N, out_channels,                H_out, W_out)

    Output mapping (per output element):
        h_out = h_in * stride_h - padding_h + k_h * dilation_h
        w_out = w_in * stride_w - padding_w + k_w * dilation_w

    Inverted, for each (h_out, w_out) the contributing input position is
        h_in = (h_out + padding_h - k_h * dilation_h) / stride_h
        w_in = (w_out + padding_w - k_w * dilation_w) / stride_w
    only when the numerator is non-negative, in-range and exactly divisible
    by the stride.
    """
    pid_n_ohw = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    # Decompose the (batch × out_h × out_w) tile index.
    n_ohw_offset = pid_n_ohw * BLOCK_N_OHW + tl.arange(0, BLOCK_N_OHW)
    n_oh_offset = n_ohw_offset // out_width
    batch_idx = n_oh_offset // out_height
    out_h_idx = n_oh_offset % out_height
    out_w_idx = n_ohw_offset % out_width

    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels  # caller already divided by groups
    oc_offset = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    accum = tl.zeros((BLOCK_N_OHW, BLOCK_OC), dtype=tl.float32)

    input_base = (
        input_pointer
        + (input_n_stride * batch_idx)[:, None]
        + (input_c_stride * pid_group * in_channels_per_group)
    )
    weight_base = (
        weight_pointer
        + (weight_ic_stride * pid_group * in_channels_per_group)
        + (weight_oc_stride * oc_offset)[None, :]
    )

    BLOCK_IC_COUNT = (in_channels_per_group + BLOCK_IC - 1) // BLOCK_IC
    for ic_khw in range(BLOCK_IC_COUNT * kernel_height * kernel_width):
        ic_block = (ic_khw // (kernel_height * kernel_width)) * BLOCK_IC
        kh_kw = ic_khw % (kernel_height * kernel_width)
        kh = kh_kw // kernel_width
        kw = kh_kw % kernel_width

        ic_offset = ic_block + tl.arange(0, BLOCK_IC)

        h_numer = out_h_idx + padding_height - kh * dilation_height
        w_numer = out_w_idx + padding_width - kw * dilation_width

        h_div = (h_numer % stride_height) == 0
        w_div = (w_numer % stride_width) == 0

        in_h_idx = h_numer // stride_height
        in_w_idx = w_numer // stride_width

        in_valid = (
            h_div
            & w_div
            & (in_h_idx >= 0)
            & (in_h_idx < input_height)
            & (in_w_idx >= 0)
            & (in_w_idx < input_width)
        )

        curr_input_pointer = (
            input_base
            + (input_c_stride * ic_offset)[None, :]
            + (input_h_stride * in_h_idx)[:, None]
            + (input_w_stride * in_w_idx)[:, None]
        )
        input_mask = (
            (batch_idx < batch_size)[:, None]
            & (ic_offset < in_channels_per_group)[None, :]
            & in_valid[:, None]
        )
        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)

        curr_weight_pointer = (
            weight_base
            + (weight_ic_stride * ic_offset)[:, None]
            + (weight_h_stride * kh)
            + (weight_w_stride * kw)
        )
        weight_mask = (ic_offset < in_channels_per_group)[:, None] & (
            oc_offset < out_channels_per_group
        )[None, :]
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

        accum += tl.dot(
            input_block.to(tl.float32),
            weight_block.to(tl.float32),
            allow_tf32=False,
        )

    bias_ptr = bias_pointer + pid_group * out_channels_per_group + oc_offset
    bias_mask = oc_offset < out_channels_per_group
    bias = tl.load(bias_ptr, mask=bias_mask, other=0.0).to(tl.float32)
    accum += bias[None, :]

    output_ptr = (
        output_pointer
        + (output_n_stride * batch_idx)[:, None]
        + (output_c_stride * (pid_group * out_channels_per_group + oc_offset))[None, :]
        + (output_h_stride * out_h_idx)[:, None]
        + (output_w_stride * out_w_idx)[:, None]
    )
    output_mask = (
        (batch_idx < batch_size)[:, None]
        & (oc_offset < out_channels_per_group)[None, :]
        & (out_h_idx < out_height)[:, None]
        & (out_w_idx < out_width)[:, None]
    )
    tl.store(output_ptr, accum, mask=output_mask)


def _pair(value):
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


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
    """torch.nn.functional.conv_transpose2d equivalent on Triton.

    Args:
        input: (N, C_in, H_in, W_in)
        weight: (C_in, C_out/groups, K_h, K_w)
        bias: optional (C_out,)
        stride, padding, output_padding, dilation: int or 2-tuple
        groups: int
    Returns:
        (N, C_out, H_out, W_out)
    """
    logger.debug("GEMS CONV_TRANSPOSE2D")

    assert input.ndim == 4, f"Input must be 4D, got shape {tuple(input.shape)}"
    assert weight.ndim == 4, f"Weight must be 4D, got shape {tuple(weight.shape)}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, got shape {tuple(bias.shape)}"

    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    out_padding_h, out_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)

    batch_size, in_channels, input_height, input_width = input.shape
    in_channels_w, out_channels_per_group, kernel_height, kernel_width = weight.shape

    assert in_channels == in_channels_w, (
        f"Input channels ({in_channels}) must match weight in_channels "
        f"({in_channels_w})"
    )
    assert (
        in_channels % groups == 0
    ), f"in_channels ({in_channels}) must be divisible by groups ({groups})"
    assert (
        out_padding_h < stride_h and out_padding_w < stride_w
    ), "output_padding must be smaller than stride along each spatial axis"

    out_channels = out_channels_per_group * groups
    assert bias is None or bias.shape[0] == out_channels, (
        f"Bias shape ({tuple(bias.shape)}) does not match out_channels "
        f"({out_channels})"
    )

    out_height = conv_transpose2d_output_size(
        input_height, kernel_height, stride_h, padding_h, out_padding_h, dilation_h
    )
    out_width = conv_transpose2d_output_size(
        input_width, kernel_width, stride_w, padding_w, out_padding_w, dilation_w
    )
    assert (
        out_height > 0 and out_width > 0
    ), "Computed output spatial extent is non-positive — check parameters"

    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )

    input_contig = input.contiguous()
    weight_contig = weight.contiguous()
    if bias is None:
        bias_pointer = torch.zeros(
            out_channels, device=input.device, dtype=input.dtype
        )
    else:
        bias_pointer = bias.contiguous()

    in_channels_per_group = in_channels // groups

    grid = lambda META: (
        triton.cdiv(batch_size * out_height * out_width, META["BLOCK_N_OHW"]),
        triton.cdiv(out_channels_per_group, META["BLOCK_OC"]),
        groups,
    )

    conv_transpose2d_forward_kernel[grid](
        input_contig,
        weight_contig,
        output,
        bias_pointer,
        batch_size,
        input_height,
        input_width,
        out_channels,
        out_height,
        out_width,
        *input_contig.stride(),
        *weight_contig.stride(),
        *output.stride(),
        in_channels_per_group,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups=groups,
    )

    return output
