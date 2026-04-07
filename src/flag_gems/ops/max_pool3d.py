import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


# max_pool3d: 3D max pooling over an input signal (N, C, D, H, W).
# Returns the maximum value (and optionally indices) within each pooling window.
# Supports dilation, ceil_mode, and return_indices.
def pool3d_output_size(
    in_size, kernel_size, stride, padding, dilation, ceil_mode=False
):
    effective_ks = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_ks
    if ceil_mode:
        out_size = (numerator + stride - 1) // stride + 1
        if (out_size - 1) * stride >= in_size + padding:
            out_size -= 1
    else:
        out_size = numerator // stride + 1
    return out_size


@libentry()
@triton.jit
def max_pool3d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
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
    dilation_d,
    dilation_h,
    dilation_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode linear index to (nc, od, oh, ow)
    ow = offsets % out_w
    tmp = offsets // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    nc = tmp // out_d

    nc_offset = nc * in_d * in_h * in_w

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val = tl.full((BLOCK_SIZE,), min_val, dtype=dtype)
    max_idx = tl.full((BLOCK_SIZE,), -1, dtype=tl.int64)

    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                id_val = od * stride_d - padding_d + kd * dilation_d
                ih_val = oh * stride_h - padding_h + kh * dilation_h
                iw_val = ow * stride_w - padding_w + kw * dilation_w

                valid = (
                    (id_val >= 0)
                    & (id_val < in_d)
                    & (ih_val >= 0)
                    & (ih_val < in_h)
                    & (iw_val >= 0)
                    & (iw_val < in_w)
                )

                in_idx = nc_offset + id_val * in_h * in_w + ih_val * in_w + iw_val
                val = tl.load(input_ptr + in_idx, mask=mask & valid, other=min_val)

                is_new_max = val > max_val
                flat_idx = id_val * in_h * in_w + ih_val * in_w + iw_val
                max_val = tl.where(is_new_max & valid, val, max_val)
                max_idx = tl.where(is_new_max & valid, flat_idx, max_idx)

    tl.store(output_ptr + offsets, max_val, mask=mask)
    tl.store(indices_ptr + offsets, max_idx, mask=mask)


def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    logger.debug("GEMS MAX_POOL3D")

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    assert input.ndim == 5, "Input must be 5D (N, C, D, H, W)"
    N, C, D, H, W = input.shape

    out_d = pool3d_output_size(
        D, kernel_size[0], stride[0], padding[0], dilation[0], ceil_mode
    )
    out_h = pool3d_output_size(
        H, kernel_size[1], stride[1], padding[1], dilation[1], ceil_mode
    )
    out_w = pool3d_output_size(
        W, kernel_size[2], stride[2], padding[2], dilation[2], ceil_mode
    )

    input = input.contiguous()
    output = torch.empty(
        (N, C, out_d, out_h, out_w), dtype=input.dtype, device=input.device
    )
    indices = torch.empty(
        (N, C, out_d, out_h, out_w), dtype=torch.int64, device=input.device
    )

    n_elements = output.numel()
    if n_elements == 0:
        if return_indices:
            return output, indices
        return output

    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        max_pool3d_kernel[grid](
            input,
            output,
            indices,
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
            dilation[0],
            dilation[1],
            dilation[2],
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    if return_indices:
        return output, indices
    return output
