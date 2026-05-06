import logging
import math
import struct
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_MAX_RANGE_SPAN = 8


@triton.jit
def _ceil_pos_to_i64(x):
    xi = x.to(tl.int64)
    return xi + tl.where(xi.to(tl.float32) < x, 1, 0)


@triton.jit
def _x2_nchw_kernel(
    grad_out,
    grad_in,
    total: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    iw = offsets % IW
    ih = (offsets // IW) % IH
    nc = offsets // (IH * IW)

    base = nc.to(tl.int64) * OH * OW + (ih * 2).to(tl.int64) * OW + iw * 2
    g00 = tl.load(grad_out + base, mask=mask, other=0.0)
    g01 = tl.load(grad_out + base + 1, mask=mask, other=0.0)
    g10 = tl.load(grad_out + base + OW, mask=mask, other=0.0)
    g11 = tl.load(grad_out + base + OW + 1, mask=mask, other=0.0)
    acc = (
        g00.to(tl.float32)
        + g01.to(tl.float32)
        + g10.to(tl.float32)
        + g11.to(tl.float32)
    )

    tl.store(grad_in + offsets, acc.to(grad_in.dtype.element_ty), mask=mask)


@triton.jit
def _range_nchw_kernel(
    grad_out,
    grad_in,
    total: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    scale_h,
    scale_w,
    MAX_SPAN_H: tl.constexpr,
    MAX_SPAN_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    iw = offsets % IW
    ih = (offsets // IW) % IH
    nc = offsets // (IH * IW)

    if USE_INT32_IDX:
        oh_start = tl.ceil(ih.to(tl.float32) * scale_h).to(tl.int32)
        oh_end = tl.ceil((ih.to(tl.float32) + 1.0) * scale_h).to(tl.int32)
        ow_start = tl.ceil(iw.to(tl.float32) * scale_w).to(tl.int32)
        ow_end = tl.ceil((iw.to(tl.float32) + 1.0) * scale_w).to(tl.int32)
    else:
        oh_start = _ceil_pos_to_i64(ih.to(tl.float32) * scale_h)
        oh_end = _ceil_pos_to_i64((ih.to(tl.float32) + 1.0) * scale_h)
        ow_start = _ceil_pos_to_i64(iw.to(tl.float32) * scale_w)
        ow_end = _ceil_pos_to_i64((iw.to(tl.float32) + 1.0) * scale_w)

    oh_start = tl.minimum(tl.maximum(oh_start, 0), OH)
    oh_end = tl.minimum(tl.maximum(oh_end, 0), OH)
    ow_start = tl.minimum(tl.maximum(ow_start, 0), OW)
    ow_end = tl.minimum(tl.maximum(ow_end, 0), OW)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    base = nc.to(tl.int64) * OH * OW
    for dh in tl.static_range(MAX_SPAN_H):
        oh = oh_start + dh
        h_valid = oh < oh_end
        for dw in tl.static_range(MAX_SPAN_W):
            ow = ow_start + dw
            valid = mask & h_valid & (ow < ow_end)
            go = tl.load(
                grad_out + base + oh.to(tl.int64) * OW + ow.to(tl.int64),
                mask=valid,
                other=0.0,
            )
            acc += go.to(tl.float32)

    tl.store(grad_in + offsets, acc.to(grad_in.dtype.element_ty), mask=mask)


@triton.jit
def _range_nchw_strided_kernel(
    grad_out,
    grad_in,
    total: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    go_stride_n: tl.constexpr,
    go_stride_c: tl.constexpr,
    go_stride_h: tl.constexpr,
    go_stride_w: tl.constexpr,
    gi_stride_n: tl.constexpr,
    gi_stride_c: tl.constexpr,
    gi_stride_h: tl.constexpr,
    gi_stride_w: tl.constexpr,
    scale_h,
    scale_w,
    MAX_SPAN_H: tl.constexpr,
    MAX_SPAN_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    iw = offsets % IW
    ih = (offsets // IW) % IH
    c = (offsets // (IH * IW)) % C
    n = offsets // (C * IH * IW)

    if USE_INT32_IDX:
        oh_start = tl.ceil(ih.to(tl.float32) * scale_h).to(tl.int32)
        oh_end = tl.ceil((ih.to(tl.float32) + 1.0) * scale_h).to(tl.int32)
        ow_start = tl.ceil(iw.to(tl.float32) * scale_w).to(tl.int32)
        ow_end = tl.ceil((iw.to(tl.float32) + 1.0) * scale_w).to(tl.int32)
    else:
        oh_start = _ceil_pos_to_i64(ih.to(tl.float32) * scale_h)
        oh_end = _ceil_pos_to_i64((ih.to(tl.float32) + 1.0) * scale_h)
        ow_start = _ceil_pos_to_i64(iw.to(tl.float32) * scale_w)
        ow_end = _ceil_pos_to_i64((iw.to(tl.float32) + 1.0) * scale_w)

    oh_start = tl.minimum(tl.maximum(oh_start, 0), OH)
    oh_end = tl.minimum(tl.maximum(oh_end, 0), OH)
    ow_start = tl.minimum(tl.maximum(ow_start, 0), OW)
    ow_end = tl.minimum(tl.maximum(ow_end, 0), OW)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    go_base = n.to(tl.int64) * go_stride_n + c.to(tl.int64) * go_stride_c
    for dh in tl.static_range(MAX_SPAN_H):
        oh = oh_start + dh
        h_valid = oh < oh_end
        for dw in tl.static_range(MAX_SPAN_W):
            ow = ow_start + dw
            valid = mask & h_valid & (ow < ow_end)
            go = tl.load(
                grad_out
                + go_base
                + oh.to(tl.int64) * go_stride_h
                + ow.to(tl.int64) * go_stride_w,
                mask=valid,
                other=0.0,
            )
            acc += go.to(tl.float32)

    gi_offset = (
        n.to(tl.int64) * gi_stride_n
        + c.to(tl.int64) * gi_stride_c
        + ih.to(tl.int64) * gi_stride_h
        + iw.to(tl.int64) * gi_stride_w
    )
    tl.store(grad_in + gi_offset, acc.to(grad_in.dtype.element_ty), mask=mask)


@triton.jit
def _span1_nchw_kernel(
    grad_out,
    grad_in,
    total: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    scale_h,
    scale_w,
    USE_INT32_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    iw = offsets % IW
    ih = (offsets // IW) % IH
    nc = offsets // (IH * IW)

    if USE_INT32_IDX:
        oh = tl.ceil(ih.to(tl.float32) * scale_h).to(tl.int32)
        oh_end = tl.ceil((ih.to(tl.float32) + 1.0) * scale_h).to(tl.int32)
        ow = tl.ceil(iw.to(tl.float32) * scale_w).to(tl.int32)
        ow_end = tl.ceil((iw.to(tl.float32) + 1.0) * scale_w).to(tl.int32)
    else:
        oh = _ceil_pos_to_i64(ih.to(tl.float32) * scale_h)
        oh_end = _ceil_pos_to_i64((ih.to(tl.float32) + 1.0) * scale_h)
        ow = _ceil_pos_to_i64(iw.to(tl.float32) * scale_w)
        ow_end = _ceil_pos_to_i64((iw.to(tl.float32) + 1.0) * scale_w)

    valid = mask & (oh < oh_end) & (ow < ow_end) & (oh < OH) & (ow < OW)
    go = tl.load(
        grad_out + nc.to(tl.int64) * OH * OW + oh.to(tl.int64) * OW + ow.to(tl.int64),
        mask=valid,
        other=0.0,
    )
    tl.store(grad_in + offsets, go, mask=mask)


@triton.jit
def _downsample_nchw_scatter_kernel(
    grad_out,
    grad_in,
    total: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    oh = (offsets // OW) % OH
    nc = offsets // (OH * OW)

    ih = tl.minimum((oh.to(tl.float32) * reciprocal_scale_h).to(tl.int64), IH - 1)
    iw = tl.minimum((ow.to(tl.float32) * reciprocal_scale_w).to(tl.int64), IW - 1)
    go = tl.load(grad_out + offsets, mask=mask, other=0.0)
    tl.store(grad_in + nc.to(tl.int64) * IH * IW + ih * IW + iw, go, mask=mask)


@triton.jit
def _x2_nhwc_kernel(
    grad_out,
    grad_in,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    hw_offsets = tl.program_id(0) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_total: tl.constexpr = N * IH * IW

    n = hw_offsets // (IH * IW)
    rem = hw_offsets % (IH * IW)
    ih = rem // IW
    iw = rem % IW

    c_mask = c_offsets < C
    hw_mask = hw_offsets < hw_total
    mask = hw_mask[:, None] & c_mask[None, :]

    c = c_offsets[None, :]
    go_base = (
        n[:, None].to(tl.int64) * OH * OW * C
        + (ih[:, None] * 2).to(tl.int64) * OW * C
        + (iw[:, None] * 2).to(tl.int64) * C
        + c
    )
    g00 = tl.load(grad_out + go_base, mask=mask, other=0.0)
    g01 = tl.load(grad_out + go_base + C, mask=mask, other=0.0)
    g10 = tl.load(grad_out + go_base + OW * C, mask=mask, other=0.0)
    g11 = tl.load(grad_out + go_base + OW * C + C, mask=mask, other=0.0)
    acc = (
        g00.to(tl.float32)
        + g01.to(tl.float32)
        + g10.to(tl.float32)
        + g11.to(tl.float32)
    )

    gi_base = (
        n[:, None].to(tl.int64) * IH * IW * C
        + ih[:, None].to(tl.int64) * IW * C
        + iw[:, None].to(tl.int64) * C
        + c
    )
    tl.store(grad_in + gi_base, acc.to(grad_in.dtype.element_ty), mask=mask)


@triton.jit
def _range_nhwc_kernel(
    grad_out,
    grad_in,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    scale_h,
    scale_w,
    MAX_SPAN_H: tl.constexpr,
    MAX_SPAN_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    hw_offsets = tl.program_id(0) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_total: tl.constexpr = N * IH * IW

    n = hw_offsets // (IH * IW)
    rem = hw_offsets % (IH * IW)
    ih = rem // IW
    iw = rem % IW

    c_mask = c_offsets < C
    hw_mask = hw_offsets < hw_total
    mask = hw_mask[:, None] & c_mask[None, :]
    c = c_offsets[None, :]

    if USE_INT32_IDX:
        oh_start = tl.ceil(ih.to(tl.float32) * scale_h).to(tl.int32)
        oh_end = tl.ceil((ih.to(tl.float32) + 1.0) * scale_h).to(tl.int32)
        ow_start = tl.ceil(iw.to(tl.float32) * scale_w).to(tl.int32)
        ow_end = tl.ceil((iw.to(tl.float32) + 1.0) * scale_w).to(tl.int32)
    else:
        oh_start = _ceil_pos_to_i64(ih.to(tl.float32) * scale_h)
        oh_end = _ceil_pos_to_i64((ih.to(tl.float32) + 1.0) * scale_h)
        ow_start = _ceil_pos_to_i64(iw.to(tl.float32) * scale_w)
        ow_end = _ceil_pos_to_i64((iw.to(tl.float32) + 1.0) * scale_w)

    oh_start = tl.minimum(tl.maximum(oh_start, 0), OH)
    oh_end = tl.minimum(tl.maximum(oh_end, 0), OH)
    ow_start = tl.minimum(tl.maximum(ow_start, 0), OW)
    ow_end = tl.minimum(tl.maximum(ow_end, 0), OW)

    acc = tl.zeros([BLOCK_HW, BLOCK_C], dtype=tl.float32)
    go_n_base = n[:, None].to(tl.int64) * OH * OW * C
    for dh in tl.static_range(MAX_SPAN_H):
        oh = oh_start + dh
        h_valid = oh < oh_end
        for dw in tl.static_range(MAX_SPAN_W):
            ow = ow_start + dw
            valid = mask & h_valid[:, None] & (ow < ow_end)[:, None]
            go = tl.load(
                grad_out
                + go_n_base
                + oh[:, None].to(tl.int64) * OW * C
                + ow[:, None].to(tl.int64) * C
                + c,
                mask=valid,
                other=0.0,
            )
            acc += go.to(tl.float32)

    gi_base = (
        n[:, None].to(tl.int64) * IH * IW * C
        + ih[:, None].to(tl.int64) * IW * C
        + iw[:, None].to(tl.int64) * C
        + c
    )
    tl.store(grad_in + gi_base, acc.to(grad_in.dtype.element_ty), mask=mask)


def _round_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _nearest_reciprocal_scale(
    input_size: int, output_size: int, scale: Optional[float]
) -> float:
    if scale is not None:
        scale_value = float(scale)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            return math.nan
        return _round_float32(1.0 / scale_value)
    if output_size <= 0:
        return math.nan
    return _round_float32(float(input_size) / float(output_size))


def _nearest_scale(input_size: int, output_size: int, scale: Optional[float]) -> float:
    if scale is not None:
        scale_value = float(scale)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            return math.nan
        return _round_float32(scale_value)
    if input_size <= 0:
        return math.nan
    return _round_float32(float(output_size) / float(input_size))


def _max_range_span(scale: float) -> int:
    if not math.isfinite(scale) or scale <= 0.0:
        return _MAX_RANGE_SPAN + 1
    if scale <= 1.0:
        return 1
    return max(1, int(math.ceil(scale)))


def _is_x2(IH: int, IW: int, OH: int, OW: int, scale_h: float, scale_w: float) -> bool:
    return (
        OH == IH * 2
        and OW == IW * 2
        and scale_h == _round_float32(0.5)
        and scale_w == _round_float32(0.5)
    )


def _x2_nchw_launch_config(dtype: torch.dtype, total: int) -> Tuple[int, int]:
    if dtype in (torch.float16, torch.bfloat16) and total > 1_000_000:
        return 256, 8
    if dtype == torch.float32:
        if 1_000_000 < total <= 2_500_000:
            return 128, 8
        if 2_500_000 < total <= 8_000_000:
            return 512, 8
        if total >= 268_000_000:
            return 256, 8
    return 256, 4


def _range_launch_config(
    dtype: torch.dtype, span_h: int, span_w: int
) -> Tuple[int, int]:
    span_product = span_h * span_w
    if span_product <= 6 and max(span_h, span_w) == 3:
        if dtype == torch.float32:
            return 256, 8
        return 1024, 4
    return 256, 4


def _contiguous_range_launch_config(
    dtype: torch.dtype, max_h_span: int, max_w_span: int
) -> Tuple[int, int]:
    return _range_launch_config(dtype, max_h_span, max_w_span)


def _use_int32_idx(*sizes: int) -> bool:
    return all(0 <= int(size) <= torch.iinfo(torch.int32).max for size in sizes)


def _nhwc_channel_block(C: int) -> int:
    if C <= 8:
        return 8
    if C <= 16:
        return 16
    return 32


def _nhwc_dense_launch_config(dtype: torch.dtype, C: int) -> Tuple[int, int, int]:
    block_c = _nhwc_channel_block(C)
    if dtype in (torch.bfloat16, torch.float32):
        block_c = max(block_c, 32)
    if dtype == torch.float32:
        return 128, block_c, 4
    return 64, block_c, 4


def _empty_grad_input(
    grad_output: torch.Tensor, input_size: Tuple[int, int, int, int]
) -> torch.Tensor:
    if grad_output.is_contiguous():
        return torch.empty(
            input_size, device=grad_output.device, dtype=grad_output.dtype
        )
    if grad_output.is_contiguous(memory_format=torch.channels_last):
        return torch.empty(
            input_size,
            device=grad_output.device,
            dtype=grad_output.dtype,
            memory_format=torch.channels_last,
        )
    return torch.empty(input_size, device=grad_output.device, dtype=grad_output.dtype)


def _native_fallback(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    return torch.ops.aten.upsample_nearest2d_backward.default.redispatch(
        _FALLBACK_KEYSET,
        grad_output,
        output_size,
        input_size,
        scales_h,
        scales_w,
    )


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    N, C, IH, IW = (int(dim) for dim in input_size)
    OH, OW = (int(dim) for dim in output_size)
    expected = (N, C, OH, OW)
    assert (
        tuple(grad_output.shape) == expected
    ), f"grad_output shape {tuple(grad_output.shape)} != expected {expected}"

    if grad_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return _native_fallback(
            grad_output, output_size, input_size, scales_h, scales_w
        )
    if N == 0 or C == 0 or IH == 0 or IW == 0:
        return _empty_grad_input(grad_output, (N, C, IH, IW))
    if OH <= 0 or OW <= 0 or IH < 0 or IW < 0:
        return _native_fallback(
            grad_output, output_size, input_size, scales_h, scales_w
        )

    scale_h = _nearest_scale(IH, OH, scales_h)
    scale_w = _nearest_scale(IW, OW, scales_w)
    span_h = _max_range_span(scale_h)
    span_w = _max_range_span(scale_w)
    if span_h > _MAX_RANGE_SPAN or span_w > _MAX_RANGE_SPAN:
        return _native_fallback(
            grad_output, output_size, input_size, scales_h, scales_w
        )

    grad_input = _empty_grad_input(grad_output, (N, C, IH, IW))
    total = grad_input.numel()
    if total == 0:
        return grad_input
    use_int32_idx = _use_int32_idx(N, C, IH, IW, OH, OW, total)

    is_channels_last_only = (
        grad_output.is_contiguous(memory_format=torch.channels_last)
        and not grad_output.is_contiguous()
    )
    reciprocal_scale_h = _nearest_reciprocal_scale(IH, OH, scales_h)
    reciprocal_scale_w = _nearest_reciprocal_scale(IW, OW, scales_w)
    use_x2 = _is_x2(IH, IW, OH, OW, reciprocal_scale_h, reciprocal_scale_w)

    with torch_device_fn.device(grad_output.device):
        if is_channels_last_only:
            if use_x2:
                block_hw, block_c, num_warps = _nhwc_dense_launch_config(
                    grad_output.dtype, C
                )
                grid = (triton.cdiv(N * IH * IW, block_hw), triton.cdiv(C, block_c))
                _x2_nhwc_kernel[grid](
                    grad_output,
                    grad_input,
                    N,
                    C,
                    IH,
                    IW,
                    OH,
                    OW,
                    BLOCK_HW=block_hw,
                    BLOCK_C=block_c,
                    num_warps=num_warps,
                )
            else:
                block_hw, num_warps = _range_launch_config(
                    grad_output.dtype, span_h, span_w
                )
                block_c = _nhwc_channel_block(C)
                if C >= 64 and span_h * span_w > 4:
                    block_hw, block_c, num_warps = _nhwc_dense_launch_config(
                        grad_output.dtype, C
                    )
                grid = (triton.cdiv(N * IH * IW, block_hw), triton.cdiv(C, block_c))
                _range_nhwc_kernel[grid](
                    grad_output,
                    grad_input,
                    N,
                    C,
                    IH,
                    IW,
                    OH,
                    OW,
                    scale_h,
                    scale_w,
                    MAX_SPAN_H=span_h,
                    MAX_SPAN_W=span_w,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_HW=block_hw,
                    BLOCK_C=block_c,
                    num_warps=num_warps,
                )
            return grad_input

        if not grad_output.is_contiguous():
            block_size, num_warps = _range_launch_config(
                grad_output.dtype, span_h, span_w
            )
            grid = (triton.cdiv(total, block_size),)
            _range_nchw_strided_kernel[grid](
                grad_output,
                grad_input,
                total,
                C,
                IH,
                IW,
                OH,
                OW,
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                grad_output.stride(3),
                grad_input.stride(0),
                grad_input.stride(1),
                grad_input.stride(2),
                grad_input.stride(3),
                scale_h,
                scale_w,
                MAX_SPAN_H=span_h,
                MAX_SPAN_W=span_w,
                USE_INT32_IDX=use_int32_idx,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
            return grad_input

        grad_out_contig = grad_output
        if use_x2:
            block_size, num_warps = _x2_nchw_launch_config(grad_output.dtype, total)
            grid = (triton.cdiv(total, block_size),)
            _x2_nchw_kernel[grid](
                grad_out_contig,
                grad_input,
                total,
                IH,
                IW,
                OH,
                OW,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        else:
            block_size, num_warps = _contiguous_range_launch_config(
                grad_output.dtype, span_h, span_w
            )
            if scale_h <= 1.0 and scale_w <= 1.0:
                grid = (triton.cdiv(total, block_size),)
                _span1_nchw_kernel[grid](
                    grad_out_contig,
                    grad_input,
                    total,
                    IH,
                    IW,
                    OH,
                    OW,
                    scale_h,
                    scale_w,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            else:
                grid = (triton.cdiv(total, block_size),)
                _range_nchw_kernel[grid](
                    grad_out_contig,
                    grad_input,
                    total,
                    IH,
                    IW,
                    OH,
                    OW,
                    scale_h,
                    scale_w,
                    MAX_SPAN_H=span_h,
                    MAX_SPAN_W=span_w,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
    return grad_input
