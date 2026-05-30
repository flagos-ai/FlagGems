"""
avg_pool3d operator implementation in Triton.
Applies 3D average pooling over input tensor.
"""

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.jit
def avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    batch,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    channels,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Triton kernel for 3D average pooling.
    Each block processes a (depth, height, width) region of output.
    """
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_b = tl.program_id(3)

    d_start = pid_d * BLOCK_SIZE_D
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)

    d_mask = d_offsets < out_d
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w

    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for kd in range(kernel_d):
        in_d_pos = d_offsets[:, None, None] * stride_d + kd - padding_d
        valid_d = (in_d_pos >= 0) & (in_d_pos < in_d)

        for kh in range(kernel_h):
            in_h_pos = h_offsets[None, :, None] * stride_h + kh - padding_h
            valid_h = (in_h_pos >= 0) & (in_h_pos < in_h)

            for kw in range(kernel_w):
                in_w_pos = w_offsets[None, None, :] * stride_w + kw - padding_w
                valid_w = (in_w_pos >= 0) & (in_w_pos < in_w)

                valid = valid_d & valid_h & valid_w

                in_offset = (
                    pid_b * channels * in_d * in_h * in_w
                    + in_d_pos * in_h * in_w
                    + in_h_pos * in_w
                    + in_w_pos
                )

                val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)

                acc += val
                count += valid.to(tl.float32)

    avg = acc / tl.where(count > 0, count, 1.0)

    out_offset = (
        pid_b * channels * out_d * out_h * out_w
        + d_offsets[:, None, None] * out_h * out_w
        + h_offsets[None, :, None] * out_w
        + w_offsets[None, None, :]
    )

    out_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    tl.store(output_ptr + out_offset, avg, mask=out_mask)


def avg_pool3d(
    input: torch.Tensor,
    kernel_size: tuple,
    stride: tuple = None,
    padding: tuple = (0, 0, 0),
) -> torch.Tensor:
    """
    PyTorch-compatible avg_pool3d implementation.
    """
    if input.numel() == 0:
        return torch.empty_like(input)

    batch, channels, in_d, in_h, in_w = input.shape

    if stride is None:
        stride = kernel_size

    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding

    out_d = (in_d + 2 * padding_d - kernel_d) // stride_d + 1
    out_h = (in_h + 2 * padding_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * padding_w - kernel_w) // stride_w + 1

    output = torch.zeros(
        batch, channels, out_d, out_h, out_w, dtype=input.dtype, device=input.device
    )

    BLOCK_SIZE_D = min(4, out_d)
    BLOCK_SIZE_H = min(8, out_h)
    BLOCK_SIZE_W = min(8, out_w)

    grid = (
        triton.cdiv(out_d, BLOCK_SIZE_D),
        triton.cdiv(out_h, BLOCK_SIZE_H),
        triton.cdiv(out_w, BLOCK_SIZE_W),
        batch,
    )

    avg_pool3d_kernel[grid](
        input,
        output,
        batch,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        channels,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )

    return output
