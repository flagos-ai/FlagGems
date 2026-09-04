# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MThreads implementation of 1D transposed convolution.

This backend owns its scheduling and deliberately does not use the generic
NVIDIA-derived autotune list. Every masked tensor address is sanitized before
pointer arithmetic, avoiding reliance on masked out-of-bounds lowering.
"""

import torch
import triton
import triton.language as tl


def conv_transpose1d_output_size(
    in_size, kernel_size, stride, padding, output_padding, dilation
):
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


@triton.jit
def _conv_transpose1d_safe_gather_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    batch_size,
    input_width,
    out_channels,
    out_width,
    input_n_stride,
    input_c_stride,
    input_w_stride,
    weight_ic_stride,
    weight_oc_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_w_stride,
    in_channels: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n_ow = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    flat = pid_n_ow * BLOCK_N_OW + tl.arange(0, BLOCK_N_OW)
    batch_idx = flat // out_width
    out_w_idx = flat % out_width
    output_valid = (batch_idx < batch_size) & (out_w_idx < out_width)
    safe_batch_idx = tl.where(output_valid, batch_idx, 0)
    safe_out_w_idx = tl.where(output_valid, out_w_idx, 0)

    out_channels_per_group = out_channels // groups
    oc_offset = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_valid = oc_offset < out_channels_per_group
    safe_oc_offset = tl.where(oc_valid, oc_offset, 0)

    input_base = (
        input_pointer
        + (input_n_stride * safe_batch_idx)[:, None]
        + input_c_stride * pid_group * in_channels
    )
    weight_base = (
        weight_pointer
        + weight_ic_stride * pid_group * in_channels
        + (weight_oc_stride * safe_oc_offset)[None, :]
    )
    accum = tl.zeros((BLOCK_N_OW, BLOCK_OC), dtype=tl.float32)

    BLOCK_IC_COUNT = (in_channels + BLOCK_IC - 1) // BLOCK_IC
    for ic_k in range(BLOCK_IC_COUNT * kernel_width):
        ic_block = (ic_k // kernel_width) * BLOCK_IC
        k = ic_k % kernel_width
        ic_offset = ic_block + tl.arange(0, BLOCK_IC)
        ic_valid = ic_offset < in_channels
        safe_ic_offset = tl.where(ic_valid, ic_offset, 0)

        numerator = safe_out_w_idx + padding_width - k * dilation_width
        nonnegative = numerator >= 0
        # Never feed a negative value to div/rem. Invalid lanes are then
        # redirected to address zero before any input pointer is built.
        nonnegative_numerator = tl.where(nonnegative, numerator, 0)
        divisible = (nonnegative_numerator % stride_width) == 0
        in_w_idx = nonnegative_numerator // stride_width
        input_valid = output_valid & nonnegative & divisible & (in_w_idx < input_width)
        safe_in_w_idx = tl.where(input_valid, in_w_idx, 0)

        input_ptr = (
            input_base
            + (input_c_stride * safe_ic_offset)[None, :]
            + (input_w_stride * safe_in_w_idx)[:, None]
        )
        input_block = tl.load(
            input_ptr,
            mask=input_valid[:, None] & ic_valid[None, :],
            other=0.0,
        )

        weight_ptr = (
            weight_base
            + (weight_ic_stride * safe_ic_offset)[:, None]
            + weight_w_stride * k
        )
        weight_block = tl.load(
            weight_ptr,
            mask=ic_valid[:, None] & oc_valid[None, :],
            other=0.0,
        )
        accum += tl.dot(
            input_block.to(tl.float32), weight_block.to(tl.float32), allow_tf32=False
        )

    if HAS_BIAS:
        bias_ptr = bias_pointer + pid_group * out_channels_per_group + safe_oc_offset
        bias = tl.load(bias_ptr, mask=oc_valid, other=0.0).to(tl.float32)
        accum += bias[None, :]
    output_ptr = (
        output_pointer
        + (output_n_stride * safe_batch_idx)[:, None]
        + (output_c_stride * (pid_group * out_channels_per_group + safe_oc_offset))[
            None, :
        ]
        + (output_w_stride * safe_out_w_idx)[:, None]
    )
    tl.store(output_ptr, accum, mask=output_valid[:, None] & oc_valid[None, :])


@triton.jit
def _conv_transpose1d_phase_gather_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    input_width,
    out_channels,
    phase_width,
    input_n_stride,
    input_c_stride,
    input_w_stride,
    weight_ic_stride,
    weight_oc_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_w_stride,
    in_channels: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_width: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PHASE: tl.constexpr,
    BLOCK_N_Q: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_bg = tl.program_id(2)
    batch_idx = pid_bg // groups
    pid_group = pid_bg % groups

    q = pid_q * BLOCK_N_Q + tl.arange(0, BLOCK_N_Q)
    output_valid = q < phase_width
    safe_q = tl.where(output_valid, q, 0)
    out_w_idx = PHASE + safe_q * stride_width

    out_channels_per_group = out_channels // groups
    oc_offset = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_valid = oc_offset < out_channels_per_group
    safe_oc_offset = tl.where(oc_valid, oc_offset, 0)

    input_base = (
        input_pointer
        + input_n_stride * batch_idx
        + input_c_stride * pid_group * in_channels
    )
    weight_base = (
        weight_pointer
        + weight_ic_stride * pid_group * in_channels
        + (weight_oc_stride * safe_oc_offset)[None, :]
    )
    accum = tl.zeros((BLOCK_N_Q, BLOCK_OC), dtype=tl.float32)

    BLOCK_IC_COUNT = (in_channels + BLOCK_IC - 1) // BLOCK_IC
    for ic_k in range(BLOCK_IC_COUNT * kernel_width):
        ic_block = (ic_k // kernel_width) * BLOCK_IC
        k = ic_k % kernel_width
        # Phase/tap compatibility is scalar control flow. Matching positions
        # use q plus a scalar offset, not vector per-lane div/rem.
        phase_numerator = PHASE + padding_width - k * dilation_width
        if phase_numerator % stride_width == 0:
            ic_offset = ic_block + tl.arange(0, BLOCK_IC)
            ic_valid = ic_offset < in_channels
            safe_ic_offset = tl.where(ic_valid, ic_offset, 0)
            in_w_idx = safe_q + phase_numerator // stride_width
            input_valid = output_valid & (in_w_idx >= 0) & (in_w_idx < input_width)
            safe_in_w_idx = tl.where(input_valid, in_w_idx, 0)

            input_ptr = (
                input_base
                + (input_c_stride * safe_ic_offset)[None, :]
                + (input_w_stride * safe_in_w_idx)[:, None]
            )
            input_block = tl.load(
                input_ptr,
                mask=input_valid[:, None] & ic_valid[None, :],
                other=0.0,
            )
            weight_ptr = (
                weight_base
                + (weight_ic_stride * safe_ic_offset)[:, None]
                + weight_w_stride * k
            )
            weight_block = tl.load(
                weight_ptr,
                mask=ic_valid[:, None] & oc_valid[None, :],
                other=0.0,
            )
            accum += tl.dot(
                input_block.to(tl.float32),
                weight_block.to(tl.float32),
                allow_tf32=False,
            )

    if HAS_BIAS:
        bias_ptr = bias_pointer + pid_group * out_channels_per_group + safe_oc_offset
        bias = tl.load(bias_ptr, mask=oc_valid, other=0.0).to(tl.float32)
        accum += bias[None, :]
    output_ptr = (
        output_pointer
        + output_n_stride * batch_idx
        + (output_c_stride * (pid_group * out_channels_per_group + safe_oc_offset))[
            None, :
        ]
        + (output_w_stride * out_w_idx)[:, None]
    )
    tl.store(output_ptr, accum, mask=output_valid[:, None] & oc_valid[None, :])


def _gather_config(in_channels_per_group, out_channels_per_group, groups):
    """Structural MThreads-safe tiles; never dispatch by benchmark case ID."""
    if groups > 1 or min(in_channels_per_group, out_channels_per_group) <= 16:
        return 16, 8, 8, 2, 1
    return 32, 16, 16, 4, 1


def _phase_config():
    return 32, 16, 16, 4, 1


def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    assert input.ndim == 3 and weight.ndim == 3
    assert bias is None or bias.ndim == 1

    stride_width = stride[0] if isinstance(stride, (list, tuple)) else stride
    padding_width = padding[0] if isinstance(padding, (list, tuple)) else padding
    output_padding_width = (
        output_padding[0]
        if isinstance(output_padding, (list, tuple))
        else output_padding
    )
    dilation_width = dilation[0] if isinstance(dilation, (list, tuple)) else dilation

    batch_size, in_channels, input_width = input.shape
    weight_in_channels, out_channels_per_group, kernel_width = weight.shape
    assert in_channels == weight_in_channels and in_channels % groups == 0
    out_channels = out_channels_per_group * groups
    assert bias is None or bias.shape[0] == out_channels

    out_width = conv_transpose1d_output_size(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        output_padding_width,
        dilation_width,
    )
    output = torch.empty(
        (batch_size, out_channels, out_width), device=input.device, dtype=input.dtype
    )
    input_contig = input.contiguous()
    weight_contig = weight.contiguous()
    bias_pointer = input_contig if bias is None else bias
    has_bias = bias is not None
    in_channels_per_group = in_channels // groups

    use_phase_gather = (
        stride_width == 2
        and groups == 1
        and kernel_width >= 5
        and min(in_channels_per_group, out_channels_per_group) > 32
        and out_width >= 128
        and batch_size * out_width * min(in_channels_per_group, out_channels_per_group)
        >= 262144
    )
    if not use_phase_gather:
        block_n, block_ic, block_oc, num_warps, num_stages = _gather_config(
            in_channels_per_group, out_channels_per_group, groups
        )
        grid = lambda META: (
            triton.cdiv(batch_size * out_width, META["BLOCK_N_OW"]),
            triton.cdiv(out_channels_per_group, META["BLOCK_OC"]),
            groups,
        )
        _conv_transpose1d_safe_gather_kernel[grid](
            input_contig,
            weight_contig,
            output,
            bias_pointer,
            batch_size,
            input_width,
            out_channels,
            out_width,
            *input_contig.stride(),
            *weight_contig.stride(),
            *output.stride(),
            in_channels_per_group,
            kernel_width,
            stride_width,
            padding_width,
            dilation_width,
            groups=groups,
            HAS_BIAS=has_bias,
            BLOCK_N_OW=block_n,
            BLOCK_IC=block_ic,
            BLOCK_OC=block_oc,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        block_n, block_ic, block_oc, num_warps, num_stages = _phase_config()
        for phase in range(min(stride_width, out_width)):
            phase_width = (out_width - phase + stride_width - 1) // stride_width
            grid = lambda META: (
                triton.cdiv(phase_width, META["BLOCK_N_Q"]),
                triton.cdiv(out_channels_per_group, META["BLOCK_OC"]),
                batch_size * groups,
            )
            _conv_transpose1d_phase_gather_kernel[grid](
                input_contig,
                weight_contig,
                output,
                bias_pointer,
                input_width,
                out_channels,
                phase_width,
                *input_contig.stride(),
                *weight_contig.stride(),
                *output.stride(),
                in_channels_per_group,
                kernel_width,
                stride_width,
                padding_width,
                dilation_width,
                groups=groups,
                HAS_BIAS=has_bias,
                PHASE=phase,
                BLOCK_N_Q=block_n,
                BLOCK_IC=block_ic,
                BLOCK_OC=block_oc,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return output


__all__ = ["conv_transpose1d", "conv_transpose1d_output_size"]
