import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@triton.jit
def upsample_nearest2d_x2_channels_last_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    IH,
    IW,
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
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    total = N * IH * IW * C
    mask = idx < total
    c = idx % C
    pixel = idx // C
    iw = pixel % IW
    ih = (pixel // IW) % IH
    n = pixel // (IH * IW)

    input_offset = n * input_stride_n + ih * input_stride_h + iw * input_stride_w + c
    output_base = (
        n * output_stride_n + (ih * 2) * output_stride_h + (iw * 2) * output_stride_w
    )
    data = tl.load(ptr_i + input_offset, mask=mask)
    tl.store(ptr_o + output_base + c, data, mask=mask)
    tl.store(ptr_o + output_base + output_stride_w + c, data, mask=mask)
    tl.store(ptr_o + output_base + output_stride_h + c, data, mask=mask)
    tl.store(
        ptr_o + output_base + output_stride_h + output_stride_w + c, data, mask=mask
    )


@triton.jit
def upsample_nearest2d_x2_contiguous_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    IH,
    IW,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tl.program_id(axis=0)
    else:
        pid = tl.program_id(axis=0).to(tl.int64)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if not USE_INT32_IDX:
        idx = idx.to(tl.int64)

    total = N * C * IH * IW
    mask = idx < total
    iw = idx % IW
    ih = (idx // IW) % IH
    nc = idx // (IH * IW)

    data = tl.load(ptr_i + idx, mask=mask)
    ow = iw * 2
    oh = ih * 2
    OW = IW * 2
    output_base = nc * (IH * 2 * OW) + oh * OW + ow
    tl.store(ptr_o + output_base, data, mask=mask)
    tl.store(ptr_o + output_base + 1, data, mask=mask)
    tl.store(ptr_o + output_base + OW, data, mask=mask)
    tl.store(ptr_o + output_base + OW + 1, data, mask=mask)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
)
@triton.jit
def upsample_nearest2d_contiguous_kernel(
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

    spatial = OH * OW
    mask = idx < spatial
    ow = idx % OW
    oh = idx // OW

    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor(oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32),
            IH - 1,
        )
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor(ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32),
            IW - 1,
        )

    nc_iter = tl.program_id(axis=1)
    nc_stride = tl.num_programs(axis=1)
    offset_i = nc_iter * IH * IW + ih * IW + iw
    offset_o = nc_iter * OH * OW + idx
    src_index_stride = nc_stride * IH * IW
    dst_index_stride = nc_stride * OH * OW
    while nc_iter < N * C:
        data = tl.load(ptr_i + offset_i, mask=mask)
        tl.store(ptr_o + offset_o, data, mask=mask)
        offset_i += src_index_stride
        offset_o += dst_index_stride
        nc_iter += nc_stride


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_nearest2d"), key=["N", "C", "OH", "OW"]
)
@triton.jit
def upsample_nearest2d_kernel(
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
    c = (idx // (OH * OW)) % C
    n = idx // (C * OH * OW)

    if SAME_H:
        ih = oh
    else:
        ih = tl.minimum(
            tl.math.floor(oh.to(tl.float32) * reciprocal_scale_h).to(tl.int32),
            IH - 1,
        )
    if SAME_W:
        iw = ow
    else:
        iw = tl.minimum(
            tl.math.floor(ow.to(tl.float32) * reciprocal_scale_w).to(tl.int32),
            IW - 1,
        )

    offset_i = (
        n * input_stride_n
        + c * input_stride_c
        + ih * input_stride_h
        + iw * input_stride_w
    )
    offset_o = (
        n * output_stride_n
        + c * output_stride_c
        + oh * output_stride_h
        + ow * output_stride_w
    )

    data = tl.load(ptr_i + offset_i, mask=mask)
    tl.store(ptr_o + offset_o, data, mask=mask)


def _compute_reciprocal_scale(
    input_size: int, output_size: int, scale: Optional[float]
) -> float:
    if scale is not None:
        return float(torch.tensor(1.0 / scale, dtype=torch.float32).item())
    return float(
        (
            torch.tensor(input_size, dtype=torch.float32)
            / torch.tensor(output_size, dtype=torch.float32)
        ).item()
    )


def _use_channels_last_output(input: torch.Tensor) -> bool:
    return input.is_contiguous(memory_format=torch.channels_last) and input.stride(1) == 1


def _identity_copy(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_strided(
        input.size(), input.stride(), dtype=input.dtype, device=input.device
    )
    torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, output, input, False)
    return output


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

    reciprocal_scale_h = _compute_reciprocal_scale(IH, OH, scales_h)
    reciprocal_scale_w = _compute_reciprocal_scale(IW, OW, scales_w)

    if (OH == IH and OW == IW) and (
        input.is_contiguous() or _use_channels_last_output(input)
    ):
        return _identity_copy(input)

    memory_format = (
        torch.channels_last
        if _use_channels_last_output(input)
        else torch.contiguous_format
    )
    output = torch.empty(
        (N, C, OH, OW),
        device=input.device,
        dtype=input.dtype,
        memory_format=memory_format,
    )
    total_threads = N * C * OH * OW
    if total_threads == 0:
        return output

    use_int32_idx = total_threads <= (2**31 - 1)
    same_h = OH == IH
    same_w = OW == IW
    if _use_channels_last_output(input) and OH == IH * 2 and OW == IW * 2:
        def grid(meta):
            return (triton.cdiv(N * IH * IW * C, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(input.device):
            upsample_nearest2d_x2_channels_last_kernel[grid](
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
                BLOCK_SIZE=256,
            )
        return output

    if input.is_contiguous() and output.is_contiguous() and OH >= IH and OW >= IW:
        if OH == IH * 2 and OW == IW * 2:
            def grid(meta):
                return (triton.cdiv(N * C * IH * IW, meta["BLOCK_SIZE"]),)

            with torch_device_fn.device(input.device):
                upsample_nearest2d_x2_contiguous_kernel[grid](
                    output,
                    input,
                    N,
                    C,
                    IH,
                    IW,
                    USE_INT32_IDX=use_int32_idx,
                    BLOCK_SIZE=1024,
                )
            return output

        total_spatial_threads = OH * OW

        def grid(meta):
            return (
                triton.cdiv(total_spatial_threads, meta["BLOCK_SIZE"]),
                triton.cdiv(N * C, 4),
            )

        with torch_device_fn.device(input.device):
            upsample_nearest2d_contiguous_kernel[grid](
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
            )
        return output

    def grid(meta):
        return (triton.cdiv(total_threads, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        upsample_nearest2d_kernel[grid](
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
        )
    return output
