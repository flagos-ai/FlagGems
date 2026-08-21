import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def prelu_backward_kernel(
    grad_output_ptr,
    x_ptr,
    weight_ptr,
    grad_input_ptr,
    n_elements,
    S,
    C,
    w_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    if w_is_scalar:
        alpha = tl.load(weight_ptr)
    else:
        c = (offsets // S) % C
        alpha = tl.load(weight_ptr + c, mask=mask)

    grad_input = tl.where(x >= 0, grad_output, grad_output * alpha)
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


@triton.jit
def prelu_weight_grad_kernel(
    grad_output_ptr,
    x_ptr,
    grad_weight_ptr,
    n_elements,
    S,
    C,
    w_is_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    if w_is_scalar:
        # Scalar weight: accumulate all negative regions
        neg_grad = tl.where(x < 0, grad_output * x, 0.0)
        grad_w = tl.sum(neg_grad.to(tl.float32))
        tl.atomic_add(grad_weight_ptr, grad_w)
    else:
        # Per-channel weight: accumulate per channel
        c = (offsets // S) % C
        neg_grad = tl.where(x < 0, grad_output * x, 0.0).to(tl.float32)
        tl.atomic_add(grad_weight_ptr + c, neg_grad, mask=mask)


def prelu_backward(grad_output, x, weight):
    logger.debug("GEMS PRELU_BACKWARD")
    if x.device.type != flag_gems.device or weight.device.type != flag_gems.device:
        raise AssertionError(f"Tensors must be {flag_gems.device} tensors.")

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    grad_input = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return grad_input, torch.zeros_like(weight)

    # Determine channel count C and spatial size S
    ndim = x.dim()
    if weight.numel() == 1:
        C = 1
        S = 1
        w_is_scalar = True
    else:
        if ndim == 1:
            C = x.shape[0]
            S = 1
        else:
            C = x.shape[1]
            S = 1
            if ndim > 2:
                for d in x.shape[2:]:
                    S *= d
        if weight.numel() != C:
            raise AssertionError(
                f"Weight numel ({weight.numel()}) must equal channel dimension size ({C})."
            )
        w_is_scalar = False

    C = max(int(C), 1)
    S = max(int(S), 1)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Compute grad_input
    prelu_backward_kernel[grid](
        grad_output,
        x,
        weight,
        grad_input,
        n_elements,
        S,
        C,
        w_is_scalar=w_is_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute grad_weight
    grad_weight = torch.zeros_like(weight)
    prelu_weight_grad_kernel[grid](
        grad_output,
        x,
        grad_weight,
        n_elements,
        S,
        C,
        w_is_scalar=w_is_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_input, grad_weight
