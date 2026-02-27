import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
)
@triton.jit
def _adaptive_avg_pool2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides for grad_output
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Strides for grad_input
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Backward kernel for adaptive average pooling 2D.

    For each input position (ih, iw), we need to find all output positions (oh, ow)
    whose pooling window includes (ih, iw), and accumulate:
        grad_input[ih, iw] += grad_output[oh, ow] / window_size

    The adaptive pooling window boundaries are computed as:
        start_h(oh) = (oh * in_h) // out_h
        end_h(oh) = ((oh + 1) * in_h + out_h - 1) // out_h
    """
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)

    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Input positions for this block
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Compute base pointers
    grad_output_base = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    grad_input_base = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Accumulator for gradients
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Cast dimensions to int64 for safe integer arithmetic
    in_h_i64 = tl.full((), in_h, dtype=tl.int64)
    in_w_i64 = tl.full((), in_w, dtype=tl.int64)
    out_h_i64 = tl.full((), out_h, dtype=tl.int64)
    out_w_i64 = tl.full((), out_w, dtype=tl.int64)

    h_in_i64 = h_in_offsets.to(tl.int64)
    w_in_i64 = w_in_offsets.to(tl.int64)

    # For each output position, check if it includes our input position
    for oh in range(out_h):
        oh_i64 = tl.full((), oh, dtype=tl.int64)

        # Compute height window boundaries for this output row
        # start_h = (oh * in_h) // out_h
        # end_h = ((oh + 1) * in_h + out_h - 1) // out_h
        start_h = (oh_i64 * in_h_i64) // out_h_i64
        end_h = ((oh_i64 + 1) * in_h_i64 + out_h_i64 - 1) // out_h_i64

        # Check if input height positions are in this window
        h_in_window = (h_in_i64[:, None] >= start_h) & (h_in_i64[:, None] < end_h)

        for ow in range(out_w):
            ow_i64 = tl.full((), ow, dtype=tl.int64)

            # Compute width window boundaries for this output column
            start_w = (ow_i64 * in_w_i64) // out_w_i64
            end_w = ((ow_i64 + 1) * in_w_i64 + out_w_i64 - 1) // out_w_i64

            # Check if input width positions are in this window
            w_in_window = (w_in_i64[None, :] >= start_w) & (w_in_i64[None, :] < end_w)

            # Combined mask: input position is in this output's window
            in_window = h_in_window & w_in_window

            # Compute window size for this output position
            window_h = end_h - start_h
            window_w = end_w - start_w
            window_size = (window_h * window_w).to(tl.float32)

            # Load gradient from output
            grad_out_ptr = (
                grad_output_base + oh * out_stride_h + ow * out_stride_w
            )
            grad_out_val = tl.load(grad_out_ptr)
            grad_out_val = grad_out_val.to(tl.float32)

            # Add contribution: grad_output / window_size for positions in window
            grad_contribution = grad_out_val / window_size
            grad_acc += tl.where(in_window, grad_contribution, 0.0)

    # Store accumulated gradients
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    grad_input_store_ptr = (
        grad_input_base
        + h_in_offsets[:, None] * in_stride_h
        + w_in_offsets[None, :] * in_stride_w
    )
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=in_write_mask,
    )


def _adaptive_avg_pool2d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the backward pass for adaptive average pooling 2D.

    Args:
        grad_output: Gradient with respect to the output of adaptive_avg_pool2d.
                    Shape: (N, C, out_H, out_W) or (C, out_H, out_W)
        self: The input tensor from the forward pass.
              Shape: (N, C, in_H, in_W) or (C, in_H, in_W)

    Returns:
        grad_input: Gradient with respect to the input.
                   Shape: same as self
    """
    logger.debug("GEMS _ADAPTIVE_AVG_POOL2D_BACKWARD")

    # Handle 3D input (C, H, W) by adding batch dimension
    input_is_3d = self.dim() == 3
    if input_is_3d:
        grad_output = grad_output.unsqueeze(0)
        self = self.unsqueeze(0)

    grad_output = grad_output.contiguous()

    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    # Create output tensor (same shape as input)
    grad_input = torch.empty_like(self, dtype=self.dtype)

    if grad_output.numel() == 0 or self.numel() == 0:
        if input_is_3d:
            return grad_input.squeeze(0)
        return grad_input

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_h, meta["BLOCK_H"]) * triton.cdiv(in_w, meta["BLOCK_W"]),
    )

    _adaptive_avg_pool2d_backward_kernel[grid](
        grad_output,
        grad_input,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
    )

    if input_is_3d:
        return grad_input.squeeze(0)

    return grad_input
