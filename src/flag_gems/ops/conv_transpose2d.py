"""``torch.nn.functional.conv_transpose2d`` backed by a single Triton GEMM.

Design summary
--------------
The transposed 2D convolution is computed directly from the gather form::

    out[n, co, oh, ow] = bias[co]
        + sum_{ci, kh, kw}
              input[n, ci, ih, iw] * weight[ci, co_in_group, kh, kw]

with ``ih = (oh + padding_h - kh*dilation_h) / stride_h`` (and analogously for
``iw``).  The division is only valid when the numerator is a non-negative
multiple of the stride, so output positions partition naturally into
``stride_h * stride_w`` *phases* keyed by ``(oh % stride_h, ow % stride_w)``.
Within a single phase the validity pattern over ``(kh, kw)`` is uniform across
the tile and the inner loop reduces to a plain GEMM ``[BLOCK_NHW, BLOCK_CI] @
[BLOCK_CI, BLOCK_CO]``.  We launch one program per ``(NHW tile, Co tile,
phase * group)``; ``triton.autotune`` picks the GEMM tile sizes.

The same kernel handles every legal combination of ``stride``, ``padding``,
``output_padding``, ``dilation``, ``groups``, ``bias``, and fp16/bf16/fp32 with
no shape-specific code paths.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _pair(value):
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _output_size(in_size, kernel_size, stride, padding, output_padding, dilation):
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
        "batch",
        "in_channels",
        "input_height",
        "input_width",
        "out_channels",
        "kernel_height",
        "kernel_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "dilation_height",
        "dilation_width",
        "groups",
    ],
)
@triton.jit
def _conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    max_compact_h,
    max_compact_w,
    input_n_stride,
    input_c_stride,
    input_h_stride,
    input_w_stride,
    weight_ci_stride,
    weight_co_stride,
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
    in_channels_per_group: tl.constexpr,
    out_channels_per_group: tl.constexpr,
    has_bias: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_phase_group = tl.program_id(2)

    n_phases: tl.constexpr = stride_height * stride_width
    pid_phase = pid_phase_group % n_phases
    pid_group = pid_phase_group // n_phases
    oh_phase = pid_phase // stride_width
    ow_phase = pid_phase % stride_width

    nhw = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_w_idx = nhw % max_compact_w
    nh = nhw // max_compact_w
    compact_h_idx = nh % max_compact_h
    n = nh // max_compact_h
    oh = compact_h_idx * stride_height + oh_phase
    ow = compact_w_idx * stride_width + ow_phase

    co_local = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_global = pid_group * out_channels_per_group + co_local
    co_valid = co_local < out_channels_per_group

    in_bounds = (n < batch) & (oh < output_height) & (ow < output_width)

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)

    for ci_base in range(0, in_channels_per_group, BLOCK_CI):
        ci_local = ci_base + tl.arange(0, BLOCK_CI)
        ci_global = pid_group * in_channels_per_group + ci_local
        ci_valid = ci_local < in_channels_per_group

        for kh in range(kernel_height):
            ih_num = oh_phase + padding_height - kh * dilation_height
            if ih_num % stride_height == 0:
                ih_base = oh + padding_height - kh * dilation_height
                ih = ih_base // stride_height
                valid_h = in_bounds & (ih_base >= 0) & (ih < input_height)
                for kw in range(kernel_width):
                    iw_num = ow_phase + padding_width - kw * dilation_width
                    if iw_num % stride_width == 0:
                        iw_base = ow + padding_width - kw * dilation_width
                        iw = iw_base // stride_width
                        valid_hw = valid_h & (iw_base >= 0) & (iw < input_width)

                        input_offsets = (
                            n[:, None] * input_n_stride
                            + ci_global[None, :] * input_c_stride
                            + ih[:, None] * input_h_stride
                            + iw[:, None] * input_w_stride
                        )
                        weight_offsets = (
                            ci_global[:, None] * weight_ci_stride
                            + co_local[None, :] * weight_co_stride
                            + kh * weight_h_stride
                            + kw * weight_w_stride
                        )
                        input_block = tl.load(
                            input_ptr + input_offsets,
                            mask=valid_hw[:, None] & ci_valid[None, :],
                            other=0.0,
                        )
                        weight_block = tl.load(
                            weight_ptr + weight_offsets,
                            mask=ci_valid[:, None] & co_valid[None, :],
                            other=0.0,
                        )
                        accum += tl.dot(
                            input_block,
                            weight_block,
                            input_precision=PRECISION,
                            out_dtype=tl.float32,
                        )

    if has_bias:
        bias_block = tl.load(bias_ptr + co_global, mask=co_valid, other=0.0)
        accum += bias_block[None, :].to(tl.float32)

    output_offsets = (
        n[:, None] * output_n_stride
        + co_global[None, :] * output_c_stride
        + oh[:, None] * output_h_stride
        + ow[:, None] * output_w_stride
    )
    output_mask = in_bounds[:, None] & co_valid[None, :]
    tl.store(output_ptr + output_offsets, accum, mask=output_mask)


def _validate(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    dilation_h,
    dilation_w,
    groups,
):
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError(
            f"conv_transpose2d expects 4D input/weight, got input dim {input.dim()} "
            f"and weight dim {weight.dim()}"
        )
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
    if (output_padding_h >= stride_h and output_padding_h >= dilation_h) or (
        output_padding_w >= stride_w and output_padding_w >= dilation_w
    ):
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )

    in_channels = input.shape[1]
    weight_in_channels = weight.shape[0]
    if in_channels != weight_in_channels:
        raise RuntimeError(
            "expected input channel dimension to match weight input channels"
        )
    if in_channels % groups != 0:
        raise RuntimeError("input channels must be divisible by groups")
    if bias is not None and bias.numel() != weight.shape[1] * groups:
        raise RuntimeError("expected bias to have one element per output channel")


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

    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    output_padding_h, output_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)

    _validate(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        dilation_h,
        dilation_w,
        groups,
    )

    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch, in_channels, input_height, input_width = input.shape
    _, out_channels_per_group, kernel_height, kernel_width = weight.shape
    out_channels = out_channels_per_group * groups
    in_channels_per_group = in_channels // groups

    output_height = _output_size(
        input_height, kernel_height, stride_h, padding_h, output_padding_h, dilation_h
    )
    output_width = _output_size(
        input_width, kernel_width, stride_w, padding_w, output_padding_w, dilation_w
    )
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError("calculated output size is too small")

    output = torch.empty(
        (batch, out_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output

    max_compact_h = triton.cdiv(output_height, stride_h)
    max_compact_w = triton.cdiv(output_width, stride_w)
    max_nhw = batch * max_compact_h * max_compact_w
    n_phase_group = stride_h * stride_w * groups

    grid = lambda meta: (
        triton.cdiv(max_nhw, meta["BLOCK_NHW"]),
        triton.cdiv(out_channels_per_group, meta["BLOCK_CO"]),
        n_phase_group,
    )

    bias_arg = bias if bias is not None else input
    precision = "tf32x3" if input.dtype is torch.float32 else "ieee"

    _conv_transpose2d_kernel[grid](
        input,
        weight,
        bias_arg,
        output,
        batch,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        max_compact_h,
        max_compact_w,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        in_channels,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels_per_group,
        out_channels_per_group,
        bias is not None,
        precision,
    )
    return output
