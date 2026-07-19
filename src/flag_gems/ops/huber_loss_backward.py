"""Triton implementation of ``aten::huber_loss_backward``.

PyTorch reference: for ``L(x, y; delta) = 0.5 * (x - y) ** 2`` when
``|x - y| <= delta`` and ``delta * (|x - y| - 0.5 * delta)`` otherwise, the
gradient of the loss w.r.t. ``input`` (a.k.a. ``self``) is

    dL / d_input = grad_output * clip(input - target, -delta, delta)

scaled by ``1 / N`` if ``reduction == 'mean'`` (``reduction == 1``) and
unscaled for ``'none' (0)`` / ``'sum' (2)``.

aten schema::

    huber_loss_backward(
        Tensor grad_output, Tensor self, Tensor target,
        int reduction, float delta,
    ) -> Tensor
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.jit
def _huber_loss_backward_kernel(
    grad_output,
    inp,
    target,
    out,
    n_elements,
    reduction_elements,
    delta: tl.constexpr,
    reduction: tl.constexpr,
    GRAD_OUTPUT_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-element gradient:

        d_input = grad_output * clip(input - target, -delta, delta) / scale

    where ``scale = N`` if ``reduction == 1 (mean)`` else 1.  The
    ``GRAD_OUTPUT_SCALAR`` constexpr lets us specialise the kernel for the
    common (and natural) case where ``grad_output`` is a 0-D tensor returned
    by autograd of a reduced loss.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp_val = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = inp_val - target_val
    # clip(diff, -delta, delta)
    grad_per_element = tl.where(
        diff > delta, delta, tl.where(diff < -delta, -delta, diff)
    )

    if GRAD_OUTPUT_SCALAR:
        grad_out = tl.load(grad_output).to(tl.float32)
        if reduction == 1:
            grad_out = grad_out * (1.0 / reduction_elements)
    else:
        grad_out = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        if reduction == 1:
            grad_out = grad_out * (1.0 / reduction_elements)

    tl.store(out + offsets, grad_per_element * grad_out, mask=mask)


def _normalize_reduction(reduction):
    """Coerce a reduction passed as a string or as an int into the canonical
    ``aten`` integer convention: 0='none', 1='mean', 2='sum'."""
    if isinstance(reduction, str):
        if reduction == "none":
            return 0
        if reduction == "mean":
            return 1
        if reduction == "sum":
            return 2
    elif isinstance(reduction, int) and reduction in (0, 1, 2):
        return reduction
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


def _check_backward_input(grad_output, input, target, delta):
    if delta < 0:
        raise RuntimeError("huber_loss does not support negative values for delta.")
    if input.device.type != device or target.device.type != device:
        raise AssertionError(
            "huber_loss_backward: input and target must be CUDA tensors."
        )
    if input.device != target.device:
        raise AssertionError(
            "huber_loss_backward: input and target must be on the same device."
        )
    if grad_output.device.type != device:
        raise AssertionError("huber_loss_backward: grad_output must be a CUDA tensor.")
    if grad_output.device != input.device:
        raise AssertionError(
            "huber_loss_backward: grad_output must be on the same device."
        )
    # PyTorch's `aten::huber_loss_backward` scales the mean-reduction
    # gradient by 1 / self.numel() (the ORIGINAL input numel, not the
    # broadcast loss numel) -- this is the convention smooth_l1_loss_backward
    # follows in FlagGems too.  Capture it before broadcasting input/target.
    reduction_elements = input.numel()
    orig_input_shape = tuple(input.shape)
    input, target = torch.broadcast_tensors(input, target)
    input = input.contiguous()
    target = target.contiguous()
    if grad_output.numel() != 1:
        grad_output = torch.broadcast_to(grad_output, input.shape)
    return (
        grad_output.contiguous(),
        input,
        target,
        reduction_elements,
        orig_input_shape,
    )


def huber_loss_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction,
    delta: float,
) -> torch.Tensor:
    logger.debug("GEMS HUBER_LOSS BACKWARD")
    reduction = _normalize_reduction(reduction)
    grad_output, input, target, reduction_elements, _ = _check_backward_input(
        grad_output, input, target, float(delta)
    )
    out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        _huber_loss_backward_kernel[grid](
            grad_output,
            input,
            target,
            out,
            n_elements,
            reduction_elements,
            delta=float(delta),
            reduction=reduction,
            GRAD_OUTPUT_SCALAR=grad_output.numel() == 1,
            BLOCK_SIZE=1024,
        )
    return out


def huber_loss_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    reduction,
    delta: float,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    """``huber_loss_backward.out`` variant: writes the result into the
    pre-allocated ``grad_input`` tensor (PyTorch's *out=* convention).
    """
    logger.debug("GEMS HUBER_LOSS BACKWARD OUT")
    reduction = _normalize_reduction(reduction)
    grad_output, input, target, reduction_elements, _ = _check_backward_input(
        grad_output, input, target, float(delta)
    )
    if grad_input.device != input.device:
        raise AssertionError(
            "huber_loss_backward.out: grad_input must be on the same device."
        )
    # `aten::huber_loss_backward.out` returns a tensor whose shape matches
    # the broadcast of `input` and `target` (the loss tensor shape), so we
    # resize `grad_input` to that.
    if tuple(grad_input.shape) != tuple(input.shape):
        grad_input.resize_(input.shape)
    out_contiguous = (
        grad_input if grad_input.is_contiguous() else torch.empty_like(input)
    )

    n_elements = input.numel()
    if n_elements > 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(input.device):
            _huber_loss_backward_kernel[grid](
                grad_output,
                input,
                target,
                out_contiguous,
                n_elements,
                reduction_elements,
                delta=float(delta),
                reduction=reduction,
                GRAD_OUTPUT_SCALAR=grad_output.numel() == 1,
                BLOCK_SIZE=1024,
            )
    if out_contiguous is not grad_input:
        grad_input.copy_(out_contiguous)
    return grad_input
