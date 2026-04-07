import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# avg_pool3d: 3D average pooling over an input signal (N, C, D, H, W).
# Iterates over the 3D kernel window, accumulates valid elements,
# and divides by the count (or divisor_override).
def pool3d_output_size(in_size, kernel_size, stride, padding, ceil_mode=False):
    numerator = in_size + 2 * padding - kernel_size
    if ceil_mode:
        out_size = (numerator + stride - 1) // stride + 1
        if (out_size - 1) * stride >= in_size + padding:
            out_size -= 1
    else:
        out_size = numerator // stride + 1
    return out_size


@libentry()
@triton.jit
def avg_pool3d_kernel(
    input_ptr,
    output_ptr,
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
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode linear index to (n, c, od, oh, ow)
    ow = offsets % out_w
    tmp = offsets // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    nc = tmp // out_d

    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    nc_offset = nc * in_d * in_h * in_w

    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                id = od * stride_d - padding_d + kd
                ih = oh * stride_h - padding_h + kh
                iw = ow * stride_w - padding_w + kw

                valid = (
                    (id >= 0)
                    & (id < in_d)
                    & (ih >= 0)
                    & (ih < in_h)
                    & (iw >= 0)
                    & (iw < in_w)
                )
                in_idx = nc_offset + id * in_h * in_w + ih * in_w + iw

                val = tl.load(input_ptr + in_idx, mask=mask & valid, other=0.0)
                sum_val += val.to(tl.float32)

                if COUNT_INCLUDE_PAD:
                    count += 1
                else:
                    count += tl.where(valid, 1, 0)

    if divisor_override > 0:
        result = sum_val / divisor_override
    else:
        result = sum_val / count.to(tl.float32)

    tl.store(output_ptr + offsets, result, mask=mask)


def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    logger.debug("GEMS AVG_POOL3D")

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)

    assert input.ndim == 5, "Input must be 5D (N, C, D, H, W)"
    N, C, D, H, W = input.shape

    out_d = pool3d_output_size(D, kernel_size[0], stride[0], padding[0], ceil_mode)
    out_h = pool3d_output_size(H, kernel_size[1], stride[1], padding[1], ceil_mode)
    out_w = pool3d_output_size(W, kernel_size[2], stride[2], padding[2], ceil_mode)

    input = input.contiguous()
    output = torch.empty(
        (N, C, out_d, out_h, out_w), dtype=input.dtype, device=input.device
    )

    n_elements = output.numel()
    if n_elements == 0:
        return output

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    div_override = divisor_override if divisor_override is not None else 0

    with torch_device_fn.device(input.device):
        avg_pool3d_kernel[grid](
            input,
            output,
            D,
            H,
            W,
            out_d,
            out_h,
            out_w,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            stride[0],
            stride[1],
            stride[2],
            padding[0],
            padding[1],
            padding[2],
            count_include_pad,
            div_override,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output
