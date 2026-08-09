import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _reflection_pad1d_backward_kernel(
    grad_output,
    grad_input,
    input_width,
    pad_left,
    output_width,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    mask = offsets < input_width

    output_base = row * output_width
    input_base = row * input_width
    center = pad_left + offsets
    value = tl.load(grad_output + output_base + center, mask=mask, other=0.0).to(
        tl.float32
    )

    left_mask = mask & (offsets > 0) & (offsets <= pad_left)
    left = tl.where(left_mask, pad_left - offsets, 0)
    left_value = tl.load(grad_output + output_base + left).to(tl.float32)
    value += tl.where(left_mask, left_value, 0.0)

    pad_right = output_width - input_width - pad_left
    right_start = input_width - pad_right - 1
    right_mask = mask & (offsets < input_width - 1) & (offsets >= right_start)
    right = tl.where(right_mask, pad_left + 2 * (input_width - 1) - offsets, 0)
    right_value = tl.load(grad_output + output_base + right).to(tl.float32)
    value += tl.where(right_mask, right_value, 0.0)

    tl.store(grad_input + input_base + offsets, value, mask=mask)


def reflection_pad1d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_BACKWARD")

    if isinstance(padding, int):
        pad_left = pad_right = padding
    else:
        pad_left, pad_right = padding

    if self.dim() not in (2, 3):
        raise ValueError("input must be a 2D or 3D tensor")

    input_width = self.shape[-1]
    output_width = input_width + pad_left + pad_right
    if grad_output.shape[-1] != output_width:
        raise ValueError(
            f"grad_output last dim {grad_output.shape[-1]}, expected {output_width}"
        )

    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(self, memory_format=torch.contiguous_format)
    rows = grad_input.numel() // input_width
    grid = (rows, triton.cdiv(input_width, 256))
    with torch_device_fn.device(self.device):
        _reflection_pad1d_backward_kernel[grid](
            grad_output,
            grad_input,
            input_width,
            pad_left,
            output_width,
            BLOCK_WIDTH=256,
        )
    return grad_input
