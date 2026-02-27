import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["out_numel"],
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def adaptive_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_h,
    in_w,
    out_h,
    out_w,
    out_numel,
    in_numel_per_nc,
    # Strides for grad_output/indices (contiguous, so (N*C, out_h*out_w))
    out_stride_nc,
    # Tiling
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a block of output elements.
    For each output element, scatter its grad_output value to the position
    indicated by the corresponding index in grad_input.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_numel

    # Load grad_output and indices
    grad_out_val = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    indices_val = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # Calculate which (n*c) batch this output element belongs to
    nc_idx = offsets // out_stride_nc

    # Calculate the position in grad_input
    # indices_val is the flat index within (in_h * in_w) spatial plane
    grad_input_offset = nc_idx * in_numel_per_nc + indices_val

    # Use atomic add since multiple output elements could map to the same input
    # (though for adaptive_max_pool2d this shouldn't happen, but using atomic
    # for correctness)
    tl.atomic_add(grad_input_ptr + grad_input_offset, grad_out_val, mask=mask)


def adaptive_max_pool2d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for adaptive_max_pool2d.

    Args:
        grad_output: Gradient from the output, shape (N, C, out_H, out_W)
        self: The original input tensor, shape (N, C, in_H, in_W)
        indices: Indices of max values from forward pass, shape (N, C, out_H, out_W)

    Returns:
        grad_input: Gradient with respect to input, shape (N, C, in_H, in_W)
    """
    logger.debug("GEMS ADAPTIVE_MAX_POOL2D_BACKWARD")

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    # Get shapes
    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    # Initialize grad_input with zeros
    grad_input = torch.zeros(
        (in_n, in_c, in_h, in_w),
        device=grad_output.device,
        dtype=torch.float32,
    )

    out_numel = grad_output.numel()
    if out_numel == 0:
        return grad_input.to(grad_output.dtype)

    out_stride_nc = out_h * out_w
    in_numel_per_nc = in_h * in_w

    grid = lambda meta: (triton.cdiv(out_numel, meta["BLOCK_SIZE"]),)

    adaptive_max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        out_numel,
        in_numel_per_nc,
        out_stride_nc,
    )

    return grad_input.to(grad_output.dtype)
