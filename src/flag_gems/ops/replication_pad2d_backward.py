import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def replication_pad2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    H_in,
    W_in,
    H_out,
    W_out,
    pad_l,
    pad_t,
    stride_gn,
    stride_gc,
    stride_gh,
    stride_gw,
    stride_in,
    stride_ic,
    stride_ih,
    stride_iw,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    c_idx = pid_nc % C
    n_idx = pid_nc // C

    grad_output_base = grad_output_ptr + n_idx * stride_gn + c_idx * stride_gc
    grad_input_base = grad_input_ptr + n_idx * stride_in + c_idx * stride_ic

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # For each output position, find which input position it came from
    # and accumulate the gradient
    oy = offs_h
    ox = offs_w

    # Compute corresponding input indices (clamped)
    iy = oy - pad_t
    iy = tl.where(iy < 0, 0, iy)
    iy = tl.where(iy > H_in - 1, H_in - 1, iy)

    ix = ox - pad_l
    ix = tl.where(ix < 0, 0, ix)
    ix = tl.where(ix > W_in - 1, W_in - 1, ix)

    # Check if this output position contributes to the current block
    mask_in = (iy[:, None] < H_in) & (ix[None, :] < W_in)
    mask_out = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)

    # Load gradient from output positions
    grad_output_offset = oy[:, None] * stride_gh + ox[None, :] * stride_gw
    grads = tl.load(grad_output_base + grad_output_offset, mask=mask_out, other=0.0)

    # Only accumulate gradients for output positions that map to valid input positions
    mask = mask_out & mask_in
    grads = tl.where(mask, grads, 0.0)

    # Find unique input positions and accumulate
    # For efficiency, we use atomic_add on the input gradient positions
    grad_input_offset = iy[:, None] * stride_ih + ix[None, :] * stride_iw

    # Convert to float32 for safe atomic accumulation
    grads_f32 = grads.to(tl.float32)

    tl.atomic_add(grad_input_base + grad_input_offset, grads_f32, mask=mask)


@triton.jit
def zero_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(ptr + offs, 0.0, mask=mask)


def replication_pad2d_backward(grad_output, input, padding):
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD")
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = padding
    else:
        pad_l, pad_r, pad_t, pad_b = padding

    N, C, H_in, W_in = input.shape
    H_out = H_in + pad_t + pad_b
    W_out = W_in + pad_l + pad_r

    grad_output = grad_output.contiguous()

    # Initialize grad_input as float32 for safe accumulation
    grad_input = torch.zeros((N, C, H_in, W_in), device=input.device, dtype=torch.float32)

    # No padding case - just copy
    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
        if H_out != H_in or W_out != W_in:
            raise RuntimeError("Internal error: output dimensions should match input when no padding")
        n_elements = N * C * H_in * W_in
        grid = (triton.cdiv(n_elements, 256),)
        with torch_device_fn.device(input.device):
            zero_kernel[grid](grad_input, n_elements, BLOCK_SIZE=256)

        # Copy gradients
        grad_input_view = grad_input.view(-1)
        grad_output_view = grad_output.view(-1)
        for i in range(grad_output_view.numel()):
            grad_input_view[i] = grad_output_view[i].to(torch.float32)

        return grad_input.to(input.dtype)

    grid = lambda META: (
        triton.cdiv(W_out, META["BLOCK_W"]),
        triton.cdiv(H_out, META["BLOCK_H"]),
        N * C,
    )

    with torch_device_fn.device(input.device):
        replication_pad2d_backward_kernel[grid](
            grad_output,
            grad_input,
            H_in,
            W_in,
            H_out,
            W_out,
            pad_l,
            pad_t,
            *grad_output.stride(),
            *grad_input.stride(),
            C,
        )

    return grad_input.to(input.dtype)
