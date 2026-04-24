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
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "in_h", "in_w"],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Adaptive pooling: compute adaptive windows
    # i_start = floor(i * in_h / out_h)
    # i_end = floor((i + 1) * in_h / out_h)
    h_start = h_out_offsets[:, None] * in_h // out_h
    h_end = ((h_out_offsets[:, None] + 1) * in_h) // out_h

    w_start = w_out_offsets[None, :] * in_w // out_w
    w_end = ((w_out_offsets[None, :] + 1) * in_w) // out_w

    # For the last element, extend to the end of input
    h_end = tl.where(h_out_offsets[:, None] == out_h - 1, in_h, h_end)
    w_end = tl.where(w_out_offsets[None, :] == out_w - 1, in_w, w_end)

    # Compute window size for each output position
    window_size = (h_end - h_start) * (w_end - w_start)
    window_size = tl.maximum(window_size, 1)

    h_start = tl.maximum(h_start, 0)
    w_start = tl.maximum(w_start, 0)

    # Accumulator
    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Find the maximum window size
    max_win_h = (in_h + out_h - 1) // out_h
    max_win_w = (in_w + out_w - 1) // out_w

    # Loop over all possible positions in the max window
    for kh in range(64):
        h_in = h_start + kh
        h_valid = (h_in >= 0) & (h_in < h_end) & (h_in < in_h)
        h_valid_kw = kh < max_win_h

        for kw in range(64):
            w_in = w_start + kw
            w_valid = (w_in >= 0) & (w_in < w_end) & (w_in < in_w)
            w_valid_kw = kw < max_win_w

            # Combine all validity checks - ensure 2D shape
            h_valid_2d = h_valid & h_valid_kw
            w_valid_2d = w_valid & w_valid_kw
            in_mask = h_valid_2d & w_valid_2d

            # Compute input offset - h_in and w_in need to be 2D
            # Use tl.reshape to ensure 2D
            h_in_2d = tl.reshape(h_in, (BLOCK_H, 1))
            w_in_2d = tl.reshape(w_in, (1, BLOCK_W))

            input_offset = h_in_2d * in_stride_h + w_in_2d * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=0.0
            )

            sum_val = tl.where(in_mask, current_val, 0.0)

            # Ensure sum_val is 2D before adding
            sum_acc = sum_acc + tl.reshape(sum_val, (BLOCK_H, BLOCK_W))

    output_vals = sum_acc / window_size.to(tl.float32)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(
        output_block_ptr, output_vals.to(output_ptr.type.element_ty), mask=out_mask
    )


def adaptive_avg_pool2d(input: torch.Tensor, output_size):
    logger.debug("GEMS ADAPTIVE_AVG_POOL2D")

    input = input.contiguous()

    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    out_h, out_w = output_size
    in_n, in_c, in_h, in_w = input.shape

    if out_h == 0 or out_w == 0 or in_h == 0 or in_w == 0:
        return torch.empty(
            (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
        )

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    adaptive_avg_pool2d_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
    )

    return output
