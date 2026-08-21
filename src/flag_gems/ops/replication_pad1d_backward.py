import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def replication_pad1d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    B,
    W_in,
    pad_left,
    W_out,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_out = offs_w < W_out

    base_out = pid_b * W_out
    base_in = pid_b * W_in

    grad = tl.load(grad_output_ptr + base_out + offs_w, mask=mask_out, other=0.0)
    grad_f32 = grad.to(tl.float32)

    w_in = offs_w.to(tl.int32) - pad_left
    w_in = tl.maximum(w_in, 0)
    w_in = tl.minimum(w_in, W_in - 1)

    grad_input_offset = base_in + w_in
    tl.atomic_add(grad_input_ptr + grad_input_offset, grad_f32, mask=mask_out)


@triton.jit
def _copy_rows_kernel(in_ptr, out_ptr, B, W, BLOCK_W: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_w < W) & (pid_b < B)

    base = pid_b * W
    vals = tl.load(in_ptr + base + offs_w, mask=mask, other=0)
    tl.store(out_ptr + base + offs_w, vals, mask=mask)


def _launch_replication_pad1d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    padding,
):
    if not isinstance(padding, (list, tuple)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of length 2: (pad_left, pad_right)"
        )
    pad_left, pad_right = int(padding[0]), int(padding[1])
    if pad_left < 0 or pad_right < 0:
        raise ValueError("padding values must be >= 0")
    if input.dim() < 1:
        raise ValueError("input must have at least 1 dimension")

    grad_output = grad_output.contiguous()
    x = input.contiguous()

    W_in = int(x.shape[-1])
    if W_in <= 0:
        raise ValueError("last dimension (width) must be > 0")

    W_out = W_in + pad_left + pad_right
    leading_shape = x.shape[:-1]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    grad_input = torch.zeros_like(x, dtype=torch.float32)

    if pad_left == 0 and pad_right == 0:
        grid = (B, triton.cdiv(W_in, 256))
        with torch_device_fn.device(x.device):
            _copy_rows_kernel[grid](grad_output, grad_input, B, W_in, BLOCK_W=256)
        if grad_input.dtype == x.dtype:
            return grad_input
        result = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            _copy_rows_kernel[grid](grad_input, result, B, W_in, BLOCK_W=256)
        return result

    grid = (B, triton.cdiv(W_out, 256))
    with torch_device_fn.device(x.device):
        replication_pad1d_backward_kernel[grid](
            grad_output, grad_input, B, W_in, pad_left, W_out, BLOCK_W=256
        )
    if grad_input.dtype == x.dtype:
        return grad_input
    result = torch.empty_like(x)
    cast_grid = (B, triton.cdiv(W_in, 256))
    with torch_device_fn.device(x.device):
        _copy_rows_kernel[cast_grid](grad_input, result, B, W_in, BLOCK_W=256)
    return result


def replication_pad1d_backward(grad_output: torch.Tensor, input: torch.Tensor, padding):
    logger.debug("GEMS REPLICATION_PAD1D_BACKWARD")
    return _launch_replication_pad1d_backward(grad_output, input, padding)
