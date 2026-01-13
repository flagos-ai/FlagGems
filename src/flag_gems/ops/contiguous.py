import logging

import torch

from flag_gems.ops.copy import copy

logger = logging.getLogger(__name__)

ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


@torch.library.custom_op("flag_gems::contiguous_forward", mutates_args=())
def contiguous_forward(inp: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS CONTIGUOUS_FORWARD")
    memory_format = torch.contiguous_format
    if inp.is_contiguous(memory_format=memory_format):
        out = inp.clone()
    else:
        if inp.dtype in ALL_FLOAT_DTYPES:
            output = torch.empty_like(inp, memory_format=memory_format)
        else:
            output = torch.empty_like(inp, memory_format=memory_format)
        out = copy(inp, out0=output)
    return out


@contiguous_forward.register_fake
def _(inp):
    return torch.empty_like(inp)


@torch.library.custom_op("flag_gems::contiguous_backward", mutates_args=())
def contiguous_backward(
    grad_input: torch.Tensor, grad_out: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS CONTIGUOUS_BACKWARD")
    output = torch.empty_like(grad_input)
    return copy(grad_out, out0=output)


@contiguous_backward.register_fake
def _(grad_out):
    return torch.empty_like(grad_out)


def backward(ctx, grad_out):
    grad_input = grad_out.new_zeros(ctx.inp_shape)
    grad_input = contiguous_backward(grad_input, grad_out)
    return grad_input, None


def setup_context(ctx, inputs, output):
    (inp,) = inputs
    ctx.inp_shape = inp.shape


contiguous_forward.register_autograd(backward, setup_context=setup_context)

# torch.library.register_autograd(
#     "aten::contiguous", backward, setup_context=setup_context
# )


def contiguous(inp, *, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    return contiguous_forward(inp)
