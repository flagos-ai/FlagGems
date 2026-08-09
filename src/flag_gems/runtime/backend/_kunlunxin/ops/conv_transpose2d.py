import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv_transpose2d import _pair, _validate_conv_transpose2d_args
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Flat 1-D tile widths that survive the TritonXPU `TritonXPUUnrollControl` pass for
# this kernel shape. Measured on kl3 (2026-08-31): 128/1024/2048 compile, while
# 256/512/4096/8192 all abort with
# `OutOfResources: uni_sram PassManager::run failed, Required: 0, Hardware limit: 0`
# (both zeros => not a real resource problem). The envelope is not monotonic, so
# only known-good widths are offered here.
_BLOCK_CANDIDATES = (1024, 2048)

# Output-channel unroll (U) reuses one input tile for U accumulators, cutting input
# traffic by U. The unrolled body is emitted kernel_h*kernel_w*U times, and beyond
# roughly 128 copies the same uni_sram pass aborts (measured: 3x3 with U=8 => 72 ok,
# 5x5 with U=8 => 200 fails, 5x5 with U=4 => 100 ok).
_MAX_UNROLLED_TAPS = 128
_U_CANDIDATES = (8, 4, 2, 1)


def _pick_block(spatial: int) -> int:
    """Pick the tile width that minimises the over-allocated padding."""
    best_block, best_pad = None, None
    for block in _BLOCK_CANDIDATES:
        padded = triton.cdiv(spatial, block) * block
        if best_pad is None or padded <= best_pad:
            best_block, best_pad = block, padded
    return best_block


def _pick_unroll(output_channels_per_group: int, taps: int) -> int:
    for candidate in _U_CANDIDATES:
        if (
            output_channels_per_group % candidate == 0
            and taps * candidate <= _MAX_UNROLLED_TAPS
        ):
            return candidate
    return 1


@triton.jit
def _conv_transpose2d_gather_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    spatial_padded: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    input_channels_per_group: tl.constexpr,
    output_channels_per_group: tl.constexpr,
    groups: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    U: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Output-centric gather formulation of conv_transpose2d.

    One program owns a flat 1-D tile of BLOCK output positions for U consecutive
    output channels of one batch element. Every input address is clamped into range
    so all loads are maskless; stride divisibility and bounds are applied with
    ``tl.where`` on the register-resident partial sum. The destination is
    over-allocated to ``spatial_padded`` elements per plane so the stores carry no
    mask either. No atomics and no discontiguous scatter are used.
    """
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    channel_tiles: tl.constexpr = output_channels_per_group // U
    batch = pid // (groups * channel_tiles)
    rest = pid % (groups * channel_tiles)
    group = rest // channel_tiles
    channel_base = (rest % channel_tiles) * U

    pos = tile * BLOCK + tl.arange(0, BLOCK)
    out_row = pos // output_width
    out_col = pos - out_row * output_width
    row_active = out_row < output_height

    input_plane = input_height * input_width
    input_base = (batch * input_channels + group * input_channels_per_group) * (
        input_plane
    )
    taps: tl.constexpr = kernel_height * kernel_width
    weight_base = (
        group * input_channels_per_group * output_channels_per_group + channel_base
    ) * taps
    weight_stride_ci = output_channels_per_group * taps

    acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK,), dtype=tl.float32)

    for kh in tl.static_range(0, kernel_height):
        row_numerator = out_row + padding_height - kh * dilation_height
        input_row = row_numerator // stride_height
        row_ok = (
            row_active
            & (row_numerator >= 0)
            & (row_numerator % stride_height == 0)
            & (input_row < input_height)
        )
        safe_row = tl.maximum(tl.minimum(input_row, input_height - 1), 0)
        for kw in tl.static_range(0, kernel_width):
            col_numerator = out_col + padding_width - kw * dilation_width
            input_col = col_numerator // stride_width
            tap_ok = (
                row_ok
                & (col_numerator >= 0)
                & (col_numerator % stride_width == 0)
                & (input_col < input_width)
            )
            safe_col = tl.maximum(tl.minimum(input_col, input_width - 1), 0)
            offsets = input_base + safe_row * input_width + safe_col
            weight_offset = weight_base + kh * kernel_width + kw

            sum0 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum1 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum2 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum3 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum4 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum5 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum6 = tl.zeros((BLOCK,), dtype=tl.float32)
            sum7 = tl.zeros((BLOCK,), dtype=tl.float32)
            for ci in range(0, input_channels_per_group):
                values = tl.load(input_ptr + offsets + ci * input_plane).to(tl.float32)
                weight_ci = weight_ptr + weight_offset + ci * weight_stride_ci
                sum0 += values * tl.load(weight_ci).to(tl.float32)
                if U > 1:
                    sum1 += values * tl.load(weight_ci + taps).to(tl.float32)
                if U > 2:
                    sum2 += values * tl.load(weight_ci + 2 * taps).to(tl.float32)
                    sum3 += values * tl.load(weight_ci + 3 * taps).to(tl.float32)
                if U > 4:
                    sum4 += values * tl.load(weight_ci + 4 * taps).to(tl.float32)
                    sum5 += values * tl.load(weight_ci + 5 * taps).to(tl.float32)
                    sum6 += values * tl.load(weight_ci + 6 * taps).to(tl.float32)
                    sum7 += values * tl.load(weight_ci + 7 * taps).to(tl.float32)
            acc0 += tl.where(tap_ok, sum0, 0.0)
            if U > 1:
                acc1 += tl.where(tap_ok, sum1, 0.0)
            if U > 2:
                acc2 += tl.where(tap_ok, sum2, 0.0)
                acc3 += tl.where(tap_ok, sum3, 0.0)
            if U > 4:
                acc4 += tl.where(tap_ok, sum4, 0.0)
                acc5 += tl.where(tap_ok, sum5, 0.0)
                acc6 += tl.where(tap_ok, sum6, 0.0)
                acc7 += tl.where(tap_ok, sum7, 0.0)

    channel = group * output_channels_per_group + channel_base
    plane = batch * (groups * output_channels_per_group) + channel
    if HAS_BIAS:
        acc0 += tl.load(bias_ptr + channel).to(tl.float32)
        if U > 1:
            acc1 += tl.load(bias_ptr + channel + 1).to(tl.float32)
        if U > 2:
            acc2 += tl.load(bias_ptr + channel + 2).to(tl.float32)
            acc3 += tl.load(bias_ptr + channel + 3).to(tl.float32)
        if U > 4:
            acc4 += tl.load(bias_ptr + channel + 4).to(tl.float32)
            acc5 += tl.load(bias_ptr + channel + 5).to(tl.float32)
            acc6 += tl.load(bias_ptr + channel + 6).to(tl.float32)
            acc7 += tl.load(bias_ptr + channel + 7).to(tl.float32)

    out = output_ptr + plane * spatial_padded + pos
    tl.store(out, acc0)
    if U > 1:
        tl.store(out + spatial_padded, acc1)
    if U > 2:
        tl.store(out + 2 * spatial_padded, acc2)
        tl.store(out + 3 * spatial_padded, acc3)
    if U > 4:
        tl.store(out + 4 * spatial_padded, acc4)
        tl.store(out + 5 * spatial_padded, acc5)
        tl.store(out + 6 * spatial_padded, acc6)
        tl.store(out + 7 * spatial_padded, acc7)


def _launch(input, weight, bias, padded, meta, unroll, block, tiles):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels_per_group,
        kernel_height,
        kernel_width,
        groups,
        output_height,
        output_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = meta
    grid = (
        batch_size * groups * (output_channels_per_group // unroll),
        tiles,
    )
    _conv_transpose2d_gather_kernel[grid](
        input,
        weight,
        bias,
        padded,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        tiles * block,
        kernel_height,
        kernel_width,
        input_channels // groups,
        output_channels_per_group,
        groups,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        HAS_BIAS=bias is not None,
        U=unroll,
        BLOCK=block,
    )


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
    logger.debug("GEMS_KUNLUNXIN CONV_TRANSPOSE2D")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (
        input.dtype not in supported_dtypes
        or weight.dtype not in supported_dtypes
        or (bias is not None and bias.dtype not in supported_dtypes)
    ):
        raise NotImplementedError(
            "conv_transpose2d does not support the requested dtype"
        )
    input_was_unbatched = input.dim() == 3
    if input_was_unbatched:
        input = input.unsqueeze(0)
    stride_h, stride_w = _pair(stride)
    padding_h, padding_w = _pair(padding)
    output_padding_h, output_padding_w = _pair(output_padding)
    dilation_h, dilation_w = _pair(dilation)
    _validate_conv_transpose2d_args(
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
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    batch_size, input_channels, input_height, input_width = input.shape
    _, output_channels_per_group, kernel_height, kernel_width = weight.shape
    output_channels = output_channels_per_group * groups
    output_height = (
        (input_height - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (kernel_height - 1)
        + output_padding_h
        + 1
    )
    output_width = (
        (input_width - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (kernel_width - 1)
        + output_padding_w
        + 1
    )
    spatial = output_height * output_width
    if batch_size == 0 or output_channels == 0 or spatial == 0:
        output = torch.empty(
            (batch_size, output_channels, output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )
        return output.squeeze(0) if input_was_unbatched else output

    meta = (
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_channels_per_group,
        kernel_height,
        kernel_width,
        groups,
        output_height,
        output_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )
    taps = kernel_height * kernel_width
    unroll = _pick_unroll(output_channels_per_group, taps)
    ladder = []
    for candidate_unroll in _U_CANDIDATES:
        if candidate_unroll > unroll or output_channels_per_group % candidate_unroll:
            continue
        for candidate_block in sorted(_BLOCK_CANDIDATES, reverse=True):
            ladder.append((candidate_unroll, candidate_block))
    preferred_block = _pick_block(spatial)
    ladder.insert(0, (unroll, preferred_block))

    last_error = None
    with torch_device_fn.device(input.device):
        for candidate_unroll, candidate_block in ladder:
            tiles = triton.cdiv(spatial, candidate_block)
            padded = torch.empty(
                (batch_size * output_channels, tiles * candidate_block),
                device=input.device,
                dtype=input.dtype,
            )
            try:
                _launch(
                    input,
                    weight,
                    bias,
                    padded,
                    meta,
                    candidate_unroll,
                    candidate_block,
                    tiles,
                )
            except triton.runtime.errors.OutOfResources as err:
                # The TritonXPU uni_sram pass rejects some (tile width, unroll)
                # combinations at compile time; walk down to a smaller shape.
                last_error = err
                del padded
                continue
            spatial_padded = tiles * candidate_block
            if spatial_padded == spatial:
                output = padded.view(
                    batch_size, output_channels, output_height, output_width
                )
            else:
                # `reshape` on the narrowed over-allocated buffer yields a strided
                # view; materialise it so the result matches ATen's contiguous
                # output (the copy is ~4 orders of magnitude cheaper than the conv).
                output = (
                    padded[:, :spatial]
                    .reshape(batch_size, output_channels, output_height, output_width)
                    .contiguous()
                )
            return output.squeeze(0) if input_was_unbatched else output
    raise last_error
