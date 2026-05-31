import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def reflection_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    B,
    H_in,
    W_in,
    pad_left,
    pad_top,
    H_out,
    W_out,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_idx = offs_n // W_out
    w_idx = offs_n % W_out
    mask_out = (offs_n < H_out * W_out) & (pid_b < B)

    base_out = pid_b * (H_out * W_out)
    base_in = pid_b * (H_in * W_in)

    # Load gradient from output and cast to float32 for accumulation
    grad = tl.load(grad_output_ptr + base_out + offs_n, mask=mask_out, other=0.0)
    grad_f32 = grad.to(tl.float32)

    # Compute reflected index for height
    y = h_idx.to(tl.int32) - pad_top
    Hm1 = H_in - 1
    pH = 2 * Hm1
    t_h = tl.abs(y)
    m_h = t_h % pH
    ih = tl.where(m_h < H_in, m_h, pH - m_h)

    # Compute reflected index for width
    x = w_idx.to(tl.int32) - pad_left
    Wm1 = W_in - 1
    pW = 2 * Wm1
    t_w = tl.abs(x)
    m_w = t_w % pW
    iw = tl.where(m_w < W_in, m_w, pW - m_w)

    # Atomically accumulate gradient to input positions (float32 for atomic safety)
    grad_input_offset = base_in + ih * W_in + iw
    tl.atomic_add(grad_input_ptr + grad_input_offset, grad_f32, mask=mask_out)


def _launch_reflection_pad2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    padding,
):
    if not isinstance(padding, (list, tuple)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 2:
        raise ValueError("input must have at least 2 dimensions")

    grad_output = grad_output.contiguous()
    x = input.contiguous()

    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    if H_in <= 0 or W_in <= 0:
        raise ValueError("spatial dimensions must be > 0")

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    # Initialize grad_input as float32 for safe accumulation
    grad_input = torch.zeros_like(x, dtype=torch.float32)

    # No padding case - just copy
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        if H_out != H_in or W_out != W_in:
            raise RuntimeError(
                "Internal error: H_out/W_out should equal H_in/W_in when no padding"
            )
        BLOCK_HW = 256
        grid = (B, triton.cdiv(H_in * W_in, BLOCK_HW))
        with torch_device_fn.device(x.device):
            _copy_tensor_kernel_f32[grid](grad_output, grad_input, B, H_in, W_in, BLOCK_HW=BLOCK_HW)
        if grad_input.dtype == x.dtype:
            return grad_input
        result = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            _copy_tensor_kernel[grid](grad_input, result, B, H_in, W_in, BLOCK_HW=BLOCK_HW)
        return result

    # Validate input dimensions
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions for reflection padding"
        )

    BLOCK_HW = 256
    grid = (B, triton.cdiv(H_out * W_out, BLOCK_HW))
    with torch_device_fn.device(x.device):
        reflection_pad2d_backward_kernel[grid](
            grad_output, grad_input, B, H_in, W_in, pad_left, pad_top, H_out, W_out, BLOCK_HW=BLOCK_HW
        )
    if grad_input.dtype == x.dtype:
        return grad_input
    result = torch.empty_like(x)
    cast_grid = (B, triton.cdiv(H_in * W_in, BLOCK_HW))
    with torch_device_fn.device(x.device):
        _copy_tensor_kernel[cast_grid](grad_input, result, B, H_in, W_in, BLOCK_HW=BLOCK_HW)
    return result


@triton.jit
def _copy_tensor_kernel_f32(in_ptr, out_ptr, B, H, W, BLOCK_HW: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_idx = offs_n // W
    w_idx = offs_n % W
    mask = (offs_n < H * W) & (pid_b < B)

    base = pid_b * (H * W)
    vals = tl.load(in_ptr + base + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offs_n, vals, mask=mask)


@triton.jit
def _copy_tensor_kernel(in_ptr, out_ptr, B, H, W, BLOCK_HW: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_idx = offs_n // W
    w_idx = offs_n % W
    mask = (offs_n < H * W) & (pid_b < B)

    base = pid_b * (H * W)
    vals = tl.load(in_ptr + base + offs_n, mask=mask, other=0)
    tl.store(out_ptr + base + offs_n, vals, mask=mask)


def reflection_pad2d_backward(grad_output: torch.Tensor, input: torch.Tensor, padding):
    logger.debug("GEMS REFLECTION_PAD2D_BACKWARD")
    return _launch_reflection_pad2d_backward(grad_output, input, padding)
