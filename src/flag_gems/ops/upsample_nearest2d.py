import logging
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


@triton.jit
def upsample_nearest2d_contiguous_flat_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    total = N * C * OH * OW
    mask = idx < total
    ow = idx % OW
    oh = (idx // OW) % OH
    nc = idx // (OH * OW)

    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum((oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32), IW - 1)

    offset_i = nc * IH * IW + ih * IW + iw
    data = tl.load(ptr_i + offset_i, mask=mask)
    tl.store(ptr_o + idx, data, mask=mask)


@triton.jit
def nearest2d_contiguous_spatial_tiles_kernel(
    out_ptr,
    in_ptr,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        tile = tl.program_id(axis=0)
    else:
        tile = tl.program_id(axis=0).to(tl.int64)

    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        offsets = offsets.to(tl.int64)

    output_plane = OH * OW
    inside_plane = offsets < output_plane
    out_w = offsets % OW
    out_h = offsets // OW

    if SAME_H:
        source_h = out_h
    else:
        source_h = tl.minimum(
            (out_h.to(tl.float32) * reciprocal_scale_h).to(tl.int32), IH - 1
        )
    if SAME_W:
        source_w = out_w
    else:
        source_w = tl.minimum(
            (out_w.to(tl.float32) * reciprocal_scale_w).to(tl.int32), IW - 1
        )

    plane_owner = tl.program_id(axis=1)
    owner_step = tl.num_programs(axis=1)
    input_offset = plane_owner * IH * IW + source_h * IW + source_w
    output_offset = plane_owner * output_plane + offsets
    input_plane_step = owner_step * IH * IW
    output_plane_step = owner_step * output_plane
    while plane_owner < N * C:
        data = tl.load(in_ptr + input_offset, mask=inside_plane)
        tl.store(out_ptr + output_offset, data, mask=inside_plane)
        input_offset += input_plane_step
        output_offset += output_plane_step
        plane_owner += owner_step


@triton.jit
def nearest2d_2x_nchw_source_kernel(
    out_ptr,
    in_ptr,
    batch_count,
    channel_count,
    input_h,
    input_w,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        tile_id = tl.program_id(axis=0)
    else:
        tile_id = tl.program_id(axis=0).to(tl.int64)

    source_offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        source_offset = source_offset.to(tl.int64)

    input_plane = input_h * input_w
    element_count = batch_count * channel_count * input_plane
    valid = source_offset < element_count
    source_pixel = source_offset % input_plane
    source_x = source_pixel % input_w
    source_y = source_pixel // input_w
    batch_channel = source_offset // input_plane

    value = tl.load(in_ptr + source_offset, mask=valid)
    output_w = input_w + input_w
    output_plane = (input_h + input_h) * output_w
    output_row = source_y + source_y
    output_col = source_x + source_x
    top_left = batch_channel * output_plane + output_row * output_w + output_col
    next_row = top_left + output_w
    tl.store(out_ptr + top_left, value, mask=valid)
    tl.store(out_ptr + top_left + 1, value, mask=valid)
    tl.store(out_ptr + next_row, value, mask=valid)
    tl.store(out_ptr + next_row + 1, value, mask=valid)


@triton.jit
def nearest2d_2x_nchw_output_kernel(
    out_ptr,
    in_ptr,
    batch_count,
    channel_count,
    input_h,
    input_w,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        tile_id = tl.program_id(axis=0)
    else:
        tile_id = tl.program_id(axis=0).to(tl.int64)

    output_offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        output_offset = output_offset.to(tl.int64)

    output_h = input_h + input_h
    output_w = input_w + input_w
    output_plane = output_h * output_w
    valid = output_offset < batch_count * channel_count * output_plane
    output_x = output_offset % output_w
    output_y = (output_offset // output_w) % output_h
    batch_channel = output_offset // output_plane
    source_y = output_y // 2
    source_x = output_x // 2

    input_offset = batch_channel * input_h * input_w + source_y * input_w + source_x
    value = tl.load(in_ptr + input_offset, mask=valid)
    tl.store(out_ptr + output_offset, value, mask=valid)


@triton.jit
def nearest2d_2x_nhwc_source_kernel(
    out_ptr,
    in_ptr,
    batch_count,
    channel_count,
    input_h,
    input_w,
    input_stride_n,
    input_stride_h,
    input_stride_w,
    output_stride_n,
    output_stride_h,
    output_stride_w,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        tile_id = tl.program_id(axis=0)
    else:
        tile_id = tl.program_id(axis=0).to(tl.int64)

    lane = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        lane = lane.to(tl.int64)

    input_pixels = input_h * input_w
    lane_count = batch_count * input_pixels * channel_count
    valid = lane < lane_count
    channel = lane % channel_count
    pixel_id = lane // channel_count
    source_x = pixel_id % input_w
    source_y = (pixel_id // input_w) % input_h
    batch = pixel_id // input_pixels

    source_addr = (
        batch * input_stride_n
        + source_y * input_stride_h
        + source_x * input_stride_w
        + channel
    )
    output_y = source_y + source_y
    output_x = source_x + source_x
    dest_pixel = (
        batch * output_stride_n
        + output_y * output_stride_h
        + output_x * output_stride_w
    )
    dest_left = dest_pixel + channel
    dest_right = dest_left + output_stride_w
    dest_bottom = dest_left + output_stride_h
    value = tl.load(in_ptr + source_addr, mask=valid)
    tl.store(out_ptr + dest_left, value, mask=valid)
    tl.store(out_ptr + dest_right, value, mask=valid)
    tl.store(out_ptr + dest_bottom, value, mask=valid)
    tl.store(out_ptr + dest_bottom + output_stride_w, value, mask=valid)


@triton.jit
def upsample_nearest2d_spatial_strided_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    spatial = OH * OW
    mask = idx < spatial
    ow = idx % OW
    oh = idx // OW
    nc = tl.program_id(axis=1)
    c = nc % C
    n = nc // C

    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum((oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32), IH - 1)
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum((ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32), IW - 1)

    input_offset = (
        n * input_stride_n
        + c * input_stride_c
        + ih * input_stride_h
        + iw * input_stride_w
    )
    output_offset = nc * OH * OW + idx
    data = tl.load(ptr_i + input_offset, mask=mask)
    tl.store(ptr_o + output_offset, data, mask=mask)


@triton.jit
def nearest2d_strided_elementwise_kernel(
    out_ptr,
    in_ptr,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    output_stride_n,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    BLOCK_SIZE: tl.constexpr,
    SAME_H: tl.constexpr,
    SAME_W: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        tile = tl.program_id(axis=0)
    else:
        tile = tl.program_id(axis=0).to(tl.int64)

    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        offsets = offsets.to(tl.int64)

    output_plane = OH * OW
    batch_span = C * output_plane
    element_count = N * batch_span
    in_bounds = offsets < element_count
    out_w = offsets % OW
    row_major = offsets // OW
    out_h = row_major % OH
    channel = (offsets // output_plane) % C
    batch = offsets // batch_span

    if SAME_H:
        source_h = out_h
    else:
        source_h = tl.minimum(
            (out_h.to(tl.float32) * reciprocal_scale_h).to(tl.int32), IH - 1
        )
    if SAME_W:
        source_w = out_w
    else:
        source_w = tl.minimum(
            (out_w.to(tl.float32) * reciprocal_scale_w).to(tl.int32), IW - 1
        )

    input_offset = (
        batch * input_stride_n
        + channel * input_stride_c
        + source_h * input_stride_h
        + source_w * input_stride_w
    )
    output_offset = (
        batch * output_stride_n
        + channel * output_stride_c
        + out_h * output_stride_h
        + out_w * output_stride_w
    )
    data = tl.load(in_ptr + input_offset, mask=in_bounds)
    tl.store(out_ptr + output_offset, data, mask=in_bounds)


def _as_float32_scalar(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _nearest_reciprocal_float32(
    input_size: int, output_size: int, scale: Optional[float]
) -> float:
    if scale is not None:
        return _as_float32_scalar(1.0 / scale)
    return _as_float32_scalar(input_size / output_size)


def _has_distinct_channels_last_strides(input: torch.Tensor) -> bool:
    return input.is_contiguous(memory_format=torch.channels_last) and (
        not input.is_contiguous()
    )


def _is_exact_2x_resize(
    IH: int,
    IW: int,
    OH: int,
    OW: int,
    scales_h: Optional[float],
    scales_w: Optional[float],
) -> bool:
    height_is_doubled = OH == IH + IH
    width_is_doubled = OW == IW + IW
    height_scale_allows_2x = scales_h is None or scales_h == 2.0
    width_scale_allows_2x = scales_w is None or scales_w == 2.0
    return (
        height_is_doubled
        and width_is_doubled
        and height_scale_allows_2x
        and width_scale_allows_2x
    )


def _native_layout_copy(input: torch.Tensor, channels_last: bool) -> torch.Tensor:
    if channels_last:
        output = torch.empty_like(input, memory_format=torch.channels_last)
    elif input.is_contiguous():
        output = torch.empty_like(input, memory_format=torch.contiguous_format)
    else:
        output = torch.empty_like(input, memory_format=torch.contiguous_format)
    torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, output, input, False)
    return output


def _nhwc_2x_source_config(dtype: torch.dtype) -> tuple[int, int]:
    if dtype is torch.float16:
        return 512, 8
    if dtype is torch.bfloat16:
        return 256, 4
    return 512, 4


def _nchw_2x_output_config(
    dtype: torch.dtype, channel_count: int, input_total: int
) -> Optional[tuple[int, int]]:
    if dtype in (torch.float16, torch.bfloat16):
        if input_total <= 4 * 1024 * 1024:
            if dtype == torch.bfloat16 and channel_count > 3:
                return 1024, 4
            return 2048, 4
        return 2048, 8
    if dtype == torch.float32 and channel_count > 3 and input_total <= 4 * 1024 * 1024:
        return 512, 4
    return None


def _nchw_flat_config(dtype: torch.dtype, is_downsample: bool) -> tuple[int, int]:
    if is_downsample:
        return 256, 8 if dtype == torch.float32 else 4
    if dtype == torch.float32:
        return 2048, 8
    if dtype == torch.bfloat16:
        return 1024, 4
    return 1024, 8


def _nchw_spatial_config(
    dtype: torch.dtype, has_explicit_scale: bool
) -> Optional[tuple[int, int, int]]:
    if has_explicit_scale:
        if dtype == torch.bfloat16:
            return 512, 4, 1
        if dtype == torch.float32:
            return 2048, 4, 1
        return 1024, 4, 1
    if dtype == torch.bfloat16:
        return 512, 4, 1
    if dtype == torch.float16:
        return 2048, 4, 2
    return None


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = _nearest_reciprocal_float32(IH, OH, scales_h)
    reciprocal_scale_w = _nearest_reciprocal_float32(IW, OW, scales_w)
    channels_last = _has_distinct_channels_last_strides(input)

    if OH == IH and OW == IW:
        return _native_layout_copy(input, channels_last)

    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    output = torch.empty(
        (N, C, OH, OW),
        device=input.device,
        dtype=input.dtype,
        memory_format=memory_format,
    )
    output_total = N * C * OH * OW
    if output_total == 0:
        return output

    same_h = OH == IH
    same_w = OW == IW
    exact_2x_resize = _is_exact_2x_resize(IH, IW, OH, OW, scales_h, scales_w)
    use_int32_idx = max(output_total, N * C * IH * IW) <= (2**31 - 1)
    is_downsample = OH < IH or OW < IW

    if channels_last and exact_2x_resize:
        block_size, num_warps = _nhwc_2x_source_config(input.dtype)

        def grid(meta):
            return (triton.cdiv(N * IH * IW * C, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(input.device):
            nearest2d_2x_nhwc_source_kernel[grid](
                output,
                input,
                N,
                C,
                IH,
                IW,
                input.stride(0),
                input.stride(2),
                input.stride(3),
                output.stride(0),
                output.stride(2),
                output.stride(3),
                USE_INT32_IDX=use_int32_idx,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        return output

    if input.is_contiguous() and output.is_contiguous():
        input_total = N * C * IH * IW
        nchw_output_2x_config = (
            _nchw_2x_output_config(input.dtype, C, input_total)
            if exact_2x_resize
            else None
        )
        if nchw_output_2x_config is not None:
            block_size, num_warps = nchw_output_2x_config

            def grid(meta):
                return (triton.cdiv(output_total, meta["BLOCK_SIZE"]),)

            with torch_device_fn.device(input.device):
                nearest2d_2x_nchw_output_kernel[grid](
                    output,
                    input,
                    N,
                    C,
                    IH,
                    IW,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            return output

        if exact_2x_resize:
            block_size = 256 if C <= 3 else 1024
            num_warps = 8 if C <= 3 else 4

            def grid(meta):
                return (triton.cdiv(input_total, meta["BLOCK_SIZE"]),)

            with torch_device_fn.device(input.device):
                nearest2d_2x_nchw_source_kernel[grid](
                    output,
                    input,
                    N,
                    C,
                    IH,
                    IW,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            return output

        spatial_config = (
            _nchw_spatial_config(
                input.dtype, scales_h is not None or scales_w is not None
            )
            if (not is_downsample and OH >= IH and OW >= IW)
            else None
        )
        if spatial_config is not None:
            block_size, num_warps, nc_group = spatial_config

            def grid(meta):
                return (
                    triton.cdiv(OH * OW, meta["BLOCK_SIZE"]),
                    triton.cdiv(N * C, nc_group),
                )

            with torch_device_fn.device(input.device):
                nearest2d_contiguous_spatial_tiles_kernel[grid](
                    output,
                    input,
                    N,
                    C,
                    OH,
                    OW,
                    IH,
                    IW,
                    reciprocal_scale_h,
                    reciprocal_scale_w,
                    SAME_H=same_h,
                    SAME_W=same_w,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                )
            return output

        block_size, num_warps = _nchw_flat_config(input.dtype, is_downsample)

        def grid(meta):
            return (triton.cdiv(output_total, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(input.device):
            upsample_nearest2d_contiguous_flat_kernel[grid](
                output,
                input,
                N,
                C,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_h,
                reciprocal_scale_w,
                SAME_H=same_h,
                SAME_W=same_w,
                USE_INT32_IDX=use_int32_idx,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
        return output

    if output.is_contiguous():

        def grid(meta):
            return (triton.cdiv(OH * OW, meta["BLOCK_SIZE"]), N * C)

        with torch_device_fn.device(input.device):
            upsample_nearest2d_spatial_strided_kernel[grid](
                output,
                input,
                N,
                C,
                OH,
                OW,
                IH,
                IW,
                reciprocal_scale_h,
                reciprocal_scale_w,
                input.stride(0),
                input.stride(1),
                input.stride(2),
                input.stride(3),
                SAME_H=same_h,
                SAME_W=same_w,
                USE_INT32_IDX=use_int32_idx,
                BLOCK_SIZE=2048,
                num_warps=8 if input.dtype == torch.float32 else 4,
            )
        return output

    def grid(meta):
        return (triton.cdiv(output_total, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        nearest2d_strided_elementwise_kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            SAME_H=same_h,
            SAME_W=same_w,
            USE_INT32_IDX=use_int32_idx,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
    return output
