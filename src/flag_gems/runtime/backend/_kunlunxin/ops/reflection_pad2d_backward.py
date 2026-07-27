import logging

import torch

logger = logging.getLogger(__name__)


def _reflect_fold(g: torch.Tensor, dim: int, pad_before: int, pad_after: int) -> torch.Tensor:
    input_size = g.shape[dim] - pad_before - pad_after
    grad_input = g.narrow(dim, pad_before, input_size).clone()

    if pad_before:
        grad_input.narrow(dim, 1, pad_before).add_(
            g.narrow(dim, 0, pad_before).flip(dim)
        )
    if pad_after:
        grad_input.narrow(dim, input_size - 1 - pad_after, pad_after).add_(
            g.narrow(dim, pad_before + input_size, pad_after).flip(dim)
        )

    return grad_input


def reflection_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_BACKWARD")

    if len(padding) != 4:
        raise ValueError("padding must be a sequence of 4 elements")
    pad_left, pad_right, pad_top, pad_bottom = (int(pad) for pad in padding)

    if self.dim() not in (3, 4):
        raise ValueError("input must be a 3D or 4D tensor")

    input_height, input_width = self.shape[-2:]
    output_height = input_height + pad_top + pad_bottom
    output_width = input_width + pad_left + pad_right
    if tuple(grad_output.shape[-2:]) != (output_height, output_width):
        raise ValueError(
            "grad_output spatial shape "
            f"{tuple(grad_output.shape[-2:])}, expected {(output_height, output_width)}"
        )

    if not any((pad_left, pad_right, pad_top, pad_bottom)):
        return grad_output.clone()

    accumulation_dtype = (
        torch.float32
        if grad_output.dtype in (torch.float16, torch.bfloat16)
        else grad_output.dtype
    )
    grad_input = grad_output.contiguous().to(accumulation_dtype)
    grad_input = _reflect_fold(grad_input, grad_input.dim() - 1, pad_left, pad_right)
    grad_input = _reflect_fold(grad_input, grad_input.dim() - 2, pad_top, pad_bottom)
    return grad_input.contiguous().to(self.dtype)
