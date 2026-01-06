import os
from pathlib import Path

import torch

# set FLAGGEMS_SOURCE_DIR to be used for c_operators
os.environ["FLAGGEMS_SOURCE_DIR"] = str(Path(__file__).parent.absolute())

from flag_gems import c_operators  # noqa: F401, E402


@torch.library.register_fake("flag_gems::add_tensor")
def _(a, b):
    return a + b


@torch.library.custom_op("my::maybe_reduce", device_types=("cuda",), mutates_args=())
def maybe_reduce(grad: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    reduce_axes = []
    diff_rank = grad.ndim - input.ndim
    for i in range(grad.ndim):
        if i < diff_rank:
            reduce_axes.append(i)
        elif grad.shape[i] > 1 and input.shape[i - diff_rank] == 1:
            reduce_axes.append(i)
    # empty reduce axes would make a full reduce
    result = torch.sum(grad, reduce_axes) if reduce_axes else grad.clone()
    result = result.view(input.shape)
    return result


@torch.library.register_fake("my::maybe_reduce")
def maybe_reduce_meta(a, b):
    return torch.empty_like(b)


def _setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_a = a
    if ctx.needs_input_grad[1]:
        saved_b = b
    ctx.save_for_backward(saved_a, saved_b)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = maybe_reduce(grad, a)
    if ctx.needs_input_grad[1]:
        grad_b = maybe_reduce(grad, b)
    return grad_a, grad_b


# This code adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "flag_gems::add_tensor", _backward, setup_context=_setup_context
)

# Now torch.ops.flag_gems.add_tensor is pt2_compliant
