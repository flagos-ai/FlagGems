import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool = False,
) -> int:
    numerator = in_size + 2 * padding - kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.jit
def avg_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    in_c: tl.constexpr,
    in_d: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    CEIL_MODE: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    ow = offsets % out_w
    oh = (offsets // out_w) % out_h
    od = (offsets // (out_h * out_w)) % out_d
    c = (offsets // (out_d * out_h * out_w)) % in_c
    n = offsets // (in_c * out_d * out_h * out_w)

    id_start = od * stride_d - padding_d
    ih_start = oh * stride_h - padding_h
    iw_start = ow * stride_w - padding_w

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for kd in tl.static_range(0, kernel_d):
        id_in = id_start + kd
        d_valid = (id_in >= 0) & (id_in < in_d)
        for kh in tl.static_range(0, kernel_h):
            ih_in = ih_start + kh
            dh_valid = d_valid & (ih_in >= 0) & (ih_in < in_h)
            for kw in tl.static_range(0, kernel_w):
                iw_in = iw_start + kw
                in_mask = mask & dh_valid & (iw_in >= 0) & (iw_in < in_w)
                input_offsets = (
                    ((n * in_c + c) * in_d + id_in) * in_h + ih_in
                ) * in_w + iw_in
                vals = tl.load(input_ptr + input_offsets, mask=in_mask, other=0.0)
                acc += tl.where(in_mask, vals, 0.0)
                count += in_mask.to(tl.int32)

    if divisor_override != 0:
        divisor = tl.full((BLOCK_SIZE,), divisor_override, dtype=tl.float32)
    elif COUNT_INCLUDE_PAD:
        if CEIL_MODE:
            d_count = tl.minimum(id_start + kernel_d, in_d + padding_d) - tl.maximum(
                id_start, -padding_d
            )
            h_count = tl.minimum(ih_start + kernel_h, in_h + padding_h) - tl.maximum(
                ih_start, -padding_h
            )
            w_count = tl.minimum(iw_start + kernel_w, in_w + padding_w) - tl.maximum(
                iw_start, -padding_w
            )
            d_count = tl.maximum(d_count, 0)
            h_count = tl.maximum(h_count, 0)
            w_count = tl.maximum(w_count, 0)
            divisor = (d_count * h_count * w_count).to(tl.float32)
        else:
            divisor = tl.full(
                (BLOCK_SIZE,), kernel_d * kernel_h * kernel_w, dtype=tl.float32
            )
    else:
        divisor = count.to(tl.float32)

    output = tl.where(divisor != 0, acc / divisor, 0.0)
    tl.store(output_ptr + offsets, output.to(output_ptr.type.element_ty), mask=mask)


def _triple(value, name):
    if isinstance(value, int):
        return value, value, value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(value)
    raise ValueError(f"{name} must be an int or a sequence of three ints")


def _parse_pool_params(kernel_size, stride, padding):
    kernel_d, kernel_h, kernel_w = _triple(kernel_size, "kernel_size")

    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride_d, stride_h, stride_w = kernel_d, kernel_h, kernel_w
    else:
        stride_d, stride_h, stride_w = _triple(stride, "stride")

    padding_d, padding_h, padding_w = _triple(padding, "padding")

    if kernel_d <= 0 or kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("kernel_size must be greater than zero")

    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride must be greater than zero")

    if padding_d < 0 or padding_h < 0 or padding_w < 0:
        raise ValueError("padding must be non-negative")

    if (
        padding_d > kernel_d // 2
        or padding_h > kernel_h // 2
        or padding_w > kernel_w // 2
    ):
        raise ValueError("pad should be smaller than or equal to half of kernel size")

    return (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    )


def avg_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    logger.debug("GEMS AVG_POOL3D FORWARD")

    if input.dim() not in (4, 5):
        raise ValueError("avg_pool3d expects 4D or 5D input")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    ) = _parse_pool_params(kernel_size, stride, padding)

    squeeze_batch = input.dim() == 4
    if squeeze_batch:
        input = input.unsqueeze(0)

    input = input.contiguous()
    in_n, in_c, in_d, in_h, in_w = input.shape

    if in_d <= 0 or in_h <= 0 or in_w <= 0:
        raise ValueError("input non-batch dimensions must have positive length")

    out_d = pool3d_output_size(in_d, kernel_d, stride_d, padding_d, ceil_mode)
    out_h = pool3d_output_size(in_h, kernel_h, stride_h, padding_h, ceil_mode)
    out_w = pool3d_output_size(in_w, kernel_w, stride_w, padding_w, ceil_mode)

    if out_d <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("calculated output size is too small")

    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output.squeeze(0) if squeeze_batch else output

    block_size = 64 if input.dtype in (torch.float32, torch.float64) else 256
    grid = (triton.cdiv(output.numel(), block_size),)

    avg_pool3d_forward_kernel[grid](
        input,
        output,
        output.numel(),
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        CEIL_MODE=ceil_mode,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0,
        BLOCK_SIZE=block_size,
    )

    return output.squeeze(0) if squeeze_batch else output
