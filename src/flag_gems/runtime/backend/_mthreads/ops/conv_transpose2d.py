# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MTHREADS conv_transpose2d with a safe-gather fallback and gated FP32 K3 path."""

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _can_use_common_residue(
    input,
    weight,
    bias,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    dilation_h,
    dilation_w,
):
    return (
        input.dim() == 4
        and weight.dim() == 4
        and bias is None
        and groups == 1
        and dilation_h == dilation_w == 1
        and stride_h == stride_w
        # Residue launch decomposition pays off for upsampling; stride=1 has
        # no phase sparsity and currently regresses on S5000, so it remains on
        # the frozen safe-gather baseline until a separate tiled schedule is
        # proven.
        and stride_h == 2
        and padding_h == padding_w
        and output_padding_h == output_padding_w == 0
        and input.is_contiguous()
        and weight.is_contiguous()
        # Low-precision configurations retain the validated safe-gather path.
        and input.dtype is torch.float32
        and weight.dtype == input.dtype
        and weight.shape[2] in (3, 5)
        and weight.shape[3] in (3, 5)
        and input.shape[1] == weight.shape[0]
    )


def _can_use_affine_residue(input, weight):
    """Narrow, independently validated affine schedule eligibility.

    The 5x5 affine lowering has a reproducible FP32 correctness failure on
    S5000 for some channel/tile combinations.  It stays on the proven
    Residue-A schedule; only the 3x3 geometry is promoted here.
    """
    return (
        # Threshold sweep (8..32 plus asymmetric boundaries) found a stable
        # 2.30x--2.60x affine win from 8x8 upward.  Smaller spatial domains
        # remain on Residue-A until separately measured.
        input.shape[2] >= 8
        and input.shape[3] >= 8
        and weight.shape[2:] == (3, 3)
    )


def _pair(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise RuntimeError("expected a single int or a pair of ints")
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


@triton.jit
def _conv_transpose2d_safe_gather_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    batch_size,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    output_channels_per_group,
    input_channels_per_group: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_ci_stride,
    weight_co_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    groups: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    flat = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    total_nhw = batch_size * output_height * output_width
    output_valid = flat < total_nhw
    safe_flat = tl.where(output_valid, flat, 0).to(tl.int64)
    plane = output_height * output_width
    n = safe_flat // plane
    rem = safe_flat - n * plane
    oh = rem // output_width
    ow = rem - oh * output_width

    co_in = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_valid = co_in < output_channels_per_group
    safe_co_in = tl.where(co_valid, co_in, 0).to(tl.int64)
    co = pid_group * output_channels_per_group + safe_co_in

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)
    for ci_block in range(ci_blocks):
        ci_in = ci_block * BLOCK_CI + tl.arange(0, BLOCK_CI)
        ci_valid = ci_in < input_channels_per_group
        safe_ci_in = tl.where(ci_valid, ci_in, 0).to(tl.int64)
        ci = pid_group * input_channels_per_group + safe_ci_in

        for kh in tl.static_range(0, weight_height):
            numerator_h = oh + padding_height - kh * dilation_height
            nonnegative_h = numerator_h >= 0
            safe_numerator_h = tl.where(nonnegative_h, numerator_h, 0)
            divisible_h = (safe_numerator_h % stride_height) == 0
            ih = safe_numerator_h // stride_height
            valid_h = nonnegative_h & divisible_h & (ih < input_height)
            safe_ih = tl.where(valid_h, ih, 0).to(tl.int64)

            for kw in tl.static_range(0, weight_width):
                numerator_w = ow + padding_width - kw * dilation_width
                nonnegative_w = numerator_w >= 0
                safe_numerator_w = tl.where(nonnegative_w, numerator_w, 0)
                divisible_w = (safe_numerator_w % stride_width) == 0
                iw = safe_numerator_w // stride_width
                valid_w = nonnegative_w & divisible_w & (iw < input_width)
                safe_iw = tl.where(valid_w, iw, 0).to(tl.int64)
                valid = output_valid & valid_h & valid_w

                input_offsets = (
                    n[:, None] * input_n_stride
                    + ci[None, :] * input_c_stride
                    + safe_ih[:, None] * input_height_stride
                    + safe_iw[:, None] * input_width_stride
                )
                weight_offsets = (
                    ci[:, None] * weight_ci_stride
                    + safe_co_in[None, :] * weight_co_stride
                    + kh * weight_height_stride
                    + kw * weight_width_stride
                )
                input_block = tl.load(
                    input_pointer + input_offsets,
                    mask=valid[:, None] & ci_valid[None, :],
                    other=0.0,
                )
                weight_block = tl.load(
                    weight_pointer + weight_offsets,
                    mask=ci_valid[:, None] & co_valid[None, :],
                    other=0.0,
                )
                accum += tl.dot(
                    input_block.to(tl.float32),
                    weight_block.to(tl.float32),
                    input_precision="ieee",
                )

    if has_bias:
        bias_values = tl.load(
            bias_pointer + co,
            mask=co_valid,
            other=0.0,
        ).to(tl.float32)
        accum += bias_values[None, :]

    output_offsets = (
        n[:, None] * output_n_stride
        + co[None, :] * output_c_stride
        + oh[:, None] * output_height_stride
        + ow[:, None] * output_width_stride
    )
    tl.store(
        output_pointer + output_offsets,
        accum,
        mask=output_valid[:, None] & co_valid[None, :],
    )


@triton.jit
def _conv_transpose2d_common_residue_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_size,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    compact_height,
    compact_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_ci_stride,
    weight_co_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    input_channels_per_group: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    RESIDUE_H: tl.constexpr,
    RESIDUE_W: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    compact_flat = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    compact_total = batch_size * compact_height * compact_width
    valid_compact = compact_flat < compact_total
    safe_compact = tl.where(valid_compact, compact_flat, 0).to(tl.int64)
    compact_plane = compact_height * compact_width
    n = safe_compact // compact_plane
    rem = safe_compact - n * compact_plane
    compact_h = rem // compact_width
    compact_w = rem - compact_h * compact_width
    oh = compact_h * stride_height + RESIDUE_H
    ow = compact_w * stride_width + RESIDUE_W
    output_valid = valid_compact & (oh < output_height) & (ow < output_width)
    safe_oh = tl.where(output_valid, oh, 0).to(tl.int64)
    safe_ow = tl.where(output_valid, ow, 0).to(tl.int64)

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_valid = co < output_channels
    safe_co = tl.where(co_valid, co, 0).to(tl.int64)
    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels_per_group, BLOCK_CI)

    for kh in tl.static_range(0, weight_height):
        if kh % stride_height == (RESIDUE_H + padding_height) % stride_height:
            numerator_h = safe_oh + padding_height - kh
            valid_h = numerator_h >= 0
            safe_num_h = tl.where(valid_h, numerator_h, 0)
            ih = safe_num_h // stride_height
            valid_h = valid_h & (ih < input_height)
            safe_ih = tl.where(valid_h, ih, 0).to(tl.int64)
            for kw in tl.static_range(0, weight_width):
                if kw % stride_width == (RESIDUE_W + padding_width) % stride_width:
                    numerator_w = safe_ow + padding_width - kw
                    valid_w = numerator_w >= 0
                    safe_num_w = tl.where(valid_w, numerator_w, 0)
                    iw = safe_num_w // stride_width
                    valid_w = valid_w & (iw < input_width)
                    safe_iw = tl.where(valid_w, iw, 0).to(tl.int64)
                    valid = output_valid & valid_h & valid_w
                    for ci_block in range(ci_blocks):
                        ci = ci_block * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        ci_valid = ci < input_channels_per_group
                        safe_ci = tl.where(ci_valid, ci, 0).to(tl.int64)
                        input_offsets = (
                            n[:, None] * input_n_stride
                            + safe_ci[None, :] * input_c_stride
                            + safe_ih[:, None] * input_height_stride
                            + safe_iw[:, None] * input_width_stride
                        )
                        weight_offsets = (
                            safe_ci[:, None] * weight_ci_stride
                            + safe_co[None, :] * weight_co_stride
                            + kh * weight_height_stride
                            + kw * weight_width_stride
                        )
                        a = tl.load(
                            input_pointer + input_offsets,
                            mask=valid[:, None] & ci_valid[None, :],
                            other=0.0,
                        )
                        b = tl.load(
                            weight_pointer + weight_offsets,
                            mask=ci_valid[:, None] & co_valid[None, :],
                            other=0.0,
                        )
                        accum += tl.dot(
                            a, b, out_dtype=tl.float32, input_precision="ieee"
                        )

    output_offsets = (
        n[:, None] * output_n_stride
        + safe_co[None, :] * output_c_stride
        + safe_oh[:, None] * output_height_stride
        + safe_ow[:, None] * output_width_stride
    )
    tl.store(
        output_pointer + output_offsets,
        accum,
        mask=output_valid[:, None] & co_valid[None, :],
    )


@triton.jit
def _conv_transpose2d_common_residue_affine_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    batch_size,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    compact_height,
    compact_width,
    compact_h_tiles,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_ci_stride,
    weight_co_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    input_channels: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    RESIDUE_H: tl.constexpr,
    RESIDUE_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Affine 2-D tile for one fixed stride-2 output phase.

    Unlike the flattened Residue-A schedule, lane coordinates are formed from
    two affine `arange` vectors.  This intentionally avoids vector i64 quotient
    and remainder operations for compact H/W reconstruction.
    """
    pid_w = tl.program_id(0)
    pid_nh = tl.program_id(1)
    pid_co = tl.program_id(2)

    # Scalar program-id decomposition: this is outside the lane vector path.
    n = pid_nh // compact_h_tiles
    tile_h = pid_nh - n * compact_h_tiles
    # Keep the dot operand natively two-dimensional.  On MTHREADS, reshaping
    # a [H, W, CI] tensor into [H*W, CI] before tl.dot is not layout-safe for
    # every input; form the 64 lanes directly instead.  `lane_h` is built from
    # compile-time row ranges, avoiding a vector quotient/remainder operation.
    m = tl.arange(0, BLOCK_H * BLOCK_W)
    lane_h = tl.zeros((BLOCK_H * BLOCK_W,), dtype=tl.int32)
    for row in tl.static_range(0, BLOCK_H):
        lane_h = tl.where((m >= row * BLOCK_W) & (m < (row + 1) * BLOCK_W), row, lane_h)
    lane_w = m - lane_h * BLOCK_W
    compact_h = tile_h * BLOCK_H + lane_h
    compact_w = pid_w * BLOCK_W + lane_w
    oh = compact_h * 2 + RESIDUE_H
    ow = compact_w * 2 + RESIDUE_W
    output_valid = (
        (n < batch_size)
        & (compact_h < compact_height)
        & (compact_w < compact_width)
        & (oh < output_height)
        & (ow < output_width)
    )
    safe_oh = tl.where(output_valid, oh, 0).to(tl.int64)
    safe_ow = tl.where(output_valid, ow, 0).to(tl.int64)

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_valid = co < output_channels
    safe_co = tl.where(co_valid, co, 0).to(tl.int64)
    accum = tl.zeros((BLOCK_H * BLOCK_W, BLOCK_CO), dtype=tl.float32)
    ci_blocks: tl.constexpr = tl.cdiv(input_channels, BLOCK_CI)

    for kh in tl.static_range(0, weight_height):
        if kh % 2 == (RESIDUE_H + padding_height) % 2:
            # With fixed phase, this quotient is compile-time constant. The
            # lane coordinate remains `tile base + arange + constant`.
            ih = compact_h + (RESIDUE_H + padding_height - kh) // 2
            valid_h = (ih >= 0) & (ih < input_height)
            safe_ih = tl.where(valid_h, ih, 0).to(tl.int64)
            for kw in tl.static_range(0, weight_width):
                if kw % 2 == (RESIDUE_W + padding_width) % 2:
                    iw = compact_w + (RESIDUE_W + padding_width - kw) // 2
                    valid_w = (iw >= 0) & (iw < input_width)
                    safe_iw = tl.where(valid_w, iw, 0).to(tl.int64)
                    valid = output_valid & valid_h & valid_w
                    for ci_block in range(ci_blocks):
                        ci = ci_block * BLOCK_CI + tl.arange(0, BLOCK_CI)
                        ci_valid = ci < input_channels
                        safe_ci = tl.where(ci_valid, ci, 0).to(tl.int64)
                        input_offsets = (
                            n * input_n_stride
                            + safe_ci[None, :] * input_c_stride
                            + safe_ih[:, None] * input_height_stride
                            + safe_iw[:, None] * input_width_stride
                        )
                        weight_offsets = (
                            safe_ci[:, None] * weight_ci_stride
                            + safe_co[None, :] * weight_co_stride
                            + kh * weight_height_stride
                            + kw * weight_width_stride
                        )
                        a = tl.load(
                            input_pointer + input_offsets,
                            mask=valid[:, None] & ci_valid[None, :],
                            other=0.0,
                        )
                        b = tl.load(
                            weight_pointer + weight_offsets,
                            mask=ci_valid[:, None] & co_valid[None, :],
                            other=0.0,
                        )
                        accum += tl.dot(
                            a,
                            b,
                            out_dtype=tl.float32,
                            input_precision="ieee",
                        )

    output_offsets = (
        n * output_n_stride
        + safe_co[None, :] * output_c_stride
        + safe_oh[:, None] * output_height_stride
        + safe_ow[:, None] * output_width_stride
    )
    tl.store(
        output_pointer + output_offsets,
        accum,
        mask=output_valid[:, None] & co_valid[None, :],
    )


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
    groups,
    dilation_h,
    dilation_w,
):
    if input.device.type != "musa" or weight.device != input.device:
        raise NotImplementedError("MTHREADS conv_transpose2d requires MUSA tensors")
    if input.dim() != 4 or weight.dim() != 4:
        raise NotImplementedError("MTHREADS conv_transpose2d requires 4-D tensors")
    if input.dtype not in _SUPPORTED_DTYPES or weight.dtype != input.dtype:
        raise NotImplementedError(
            "MTHREADS conv_transpose2d supports float32/float16/bfloat16"
        )
    if bias is not None:
        if bias.device != input.device or bias.dtype != input.dtype or bias.dim() != 1:
            raise RuntimeError(
                "bias must be a 1-D tensor on the input device with matching dtype"
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
    if output_padding_h >= stride_h and output_padding_h >= dilation_h:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )
    if output_padding_w >= stride_w and output_padding_w >= dilation_w:
        raise RuntimeError(
            "output padding must be smaller than either stride or dilation"
        )

    batch, input_channels, input_height, input_width = input.shape
    weight_input_channels, output_channels_per_group, kh, kw = weight.shape
    if input_channels <= 0 or output_channels_per_group <= 0 or kh <= 0 or kw <= 0:
        raise RuntimeError(
            "non-empty input channels and weight dimensions are required"
        )
    if input_channels != weight_input_channels:
        raise RuntimeError(
            "expected input channel dimension to match weight input channels"
        )
    if input_channels % groups != 0:
        raise RuntimeError("input channels must be divisible by groups")
    output_channels = output_channels_per_group * groups
    if bias is not None and bias.numel() != output_channels:
        raise RuntimeError("expected bias to have one element per output channel")
    output_height = _output_size(
        input_height, kh, stride_h, padding_h, output_padding_h, dilation_h
    )
    output_width = _output_size(
        input_width, kw, stride_w, padding_w, output_padding_w, dilation_w
    )
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError("calculated output size is too small")
    return output_height, output_width


def _conv_transpose2d_common_residue(
    input, weight, stride_h, stride_w, padding_h, padding_w, config=None
):
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels, weight_height, weight_width = weight.shape
    output_height = _output_size(input_height, weight_height, stride_h, padding_h, 0, 1)
    output_width = _output_size(input_width, weight_width, stride_w, padding_w, 0, 1)

    configs = {
        "A": (64, 32, 32, 4, 1),
        "B": (128, 32, 16, 4, 1),
        "C": (64, 32, 64, 8, 1),
        "D": (32, 64, 32, 4, 1),
    }
    block_nhw, block_ci, block_co, num_warps, num_stages = configs[config or "A"]
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output

    for residue_h in range(stride_h):
        compact_height = max((output_height + stride_h - 1 - residue_h) // stride_h, 0)
        if compact_height == 0:
            continue
        for residue_w in range(stride_w):
            compact_width = max(
                (output_width + stride_w - 1 - residue_w) // stride_w, 0
            )
            if compact_width == 0:
                continue
            grid = (
                triton.cdiv(batch * compact_height * compact_width, block_nhw),
                triton.cdiv(output_channels, block_co),
            )
            _conv_transpose2d_common_residue_kernel[grid](
                input,
                weight,
                output,
                batch,
                input_channels,
                input_height,
                input_width,
                output_channels,
                output_height,
                output_width,
                compact_height,
                compact_width,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                input_channels,
                weight_height,
                weight_width,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                RESIDUE_H=residue_h,
                RESIDUE_W=residue_w,
                BLOCK_NHW=block_nhw,
                BLOCK_CI=block_ci,
                BLOCK_CO=block_co,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return output


def _conv_transpose2d_common_residue_affine(
    input, weight, stride_h, stride_w, padding_h, padding_w
):
    """Affine 3x3 FP32 residue schedule for the validated S5000 regime."""
    if stride_h != 2 or stride_w != 2:
        raise ValueError("affine residue path requires stride=2")
    batch, input_channels, input_height, input_width = input.shape
    _, output_channels, weight_height, weight_width = weight.shape
    output_height = _output_size(input_height, weight_height, stride_h, padding_h, 0, 1)
    output_width = _output_size(input_width, weight_width, stride_w, padding_w, 0, 1)
    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    block_h, block_w, block_ci, block_co, num_warps, num_stages = 4, 16, 32, 32, 4, 1
    for residue_h in range(2):
        compact_height = max((output_height + 1 - residue_h) // 2, 0)
        compact_h_tiles = triton.cdiv(compact_height, block_h)
        for residue_w in range(2):
            compact_width = max((output_width + 1 - residue_w) // 2, 0)
            grid = (
                triton.cdiv(compact_width, block_w),
                batch * compact_h_tiles,
                triton.cdiv(output_channels, block_co),
            )
            _conv_transpose2d_common_residue_affine_kernel[grid](
                input,
                weight,
                output,
                batch,
                input_height,
                input_width,
                output_channels,
                output_height,
                output_width,
                compact_height,
                compact_width,
                compact_h_tiles,
                *input.stride(),
                *weight.stride(),
                *output.stride(),
                input_channels,
                weight_height,
                weight_width,
                padding_h,
                padding_w,
                RESIDUE_H=residue_h,
                RESIDUE_W=residue_w,
                BLOCK_H=block_h,
                BLOCK_W=block_w,
                BLOCK_CI=block_ci,
                BLOCK_CO=block_co,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return output


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
    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)
    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    output_padding_h, output_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)
    output_height, output_width = _validate(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    )

    input_contig = input.contiguous()
    weight_contig = weight.contiguous()
    bias_contig = None if bias is None else bias.contiguous()
    batch, input_channels, input_height, input_width = input_contig.shape
    _, output_channels_per_group, kh, kw = weight_contig.shape
    output_channels = output_channels_per_group * groups
    # Preserve the fast-path layout contract.  Materialization below is needed
    # by the conservative gather implementation, but must not silently turn a
    # caller-provided non-contiguous tensor into an affine/residue candidate.
    if _can_use_common_residue(
        input,
        weight,
        bias_contig,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w,
    ):
        # The affine schedule keeps the same 64-output tile but derives H/W
        # from 2-D launch coordinates, eliminating vector i64 flat-NHW
        # quotient/remainder operations.  The affine K3 schedule is enabled
        # from the measured 8x8 crossover; K5 remains on Residue-A because
        # its affine lowering has an independently reproduced correctness bug.
        if _can_use_affine_residue(input, weight):
            output = _conv_transpose2d_common_residue_affine(
                input_contig,
                weight_contig,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
            )
        else:
            output = _conv_transpose2d_common_residue(
                input_contig,
                weight_contig,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
            )
        return output.squeeze(0) if input_was_unbatched else output

    output = torch.empty(
        (batch, output_channels, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output.squeeze(0) if input_was_unbatched else output

    block_nhw = 32
    block_ci = 16
    block_co = 16
    bias_pointer = input_contig if bias_contig is None else bias_contig
    grid = (
        triton.cdiv(batch * output_height * output_width, block_nhw),
        triton.cdiv(output_channels_per_group, block_co),
        groups,
    )
    _conv_transpose2d_safe_gather_kernel[grid](
        input_contig,
        weight_contig,
        bias_pointer,
        output,
        batch,
        input_channels,
        input_height,
        input_width,
        output_channels,
        output_height,
        output_width,
        output_channels_per_group,
        input_channels // groups,
        kh,
        kw,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        *input_contig.stride(),
        *weight_contig.stride(),
        *output.stride(),
        groups=groups,
        has_bias=bias_contig is not None,
        BLOCK_NHW=block_nhw,
        BLOCK_CI=block_ci,
        BLOCK_CO=block_co,
        num_warps=4,
        num_stages=1,
    )
    return output.squeeze(0) if input_was_unbatched else output


__all__ = ["conv_transpose2d"]
