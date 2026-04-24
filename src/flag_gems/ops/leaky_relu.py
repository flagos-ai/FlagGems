import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DEFAULT_BLOCK_SIZE = 2048
_HALF_FORWARD_BLOCK_SIZE = 4096


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], is_tensor=[True, False])
@triton.jit
def leaky_relu_func(x, negative_slope):
    x_f = x.to(tl.float32)
    return tl.where(x_f >= 0, x_f, x_f * negative_slope)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], is_tensor=[True, True, False])
@triton.jit
def leaky_relu_backward_func(grad_output, x, negative_slope):
    x_f = x.to(tl.float32)
    grad_f = grad_output.to(tl.float32)
    return tl.where(x_f > 0, grad_f, grad_f * negative_slope)


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x >= 0.0, x, x * negative_slope)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def leaky_relu_inplace_kernel(
    x_ptr, n_elements, negative_slope: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x >= 0.0, x, x * negative_slope)
    tl.store(x_ptr + offsets, y, mask=mask)


def _can_use_fast_kernel(A):
    return (
        isinstance(A, torch.Tensor)
        and A.is_cuda
        and A.is_contiguous()
        and A.dtype in _FAST_DTYPES
    )


def _get_block_size(A, is_inplace):
    if not is_inplace and A.dtype in (torch.float16, torch.bfloat16):
        return _HALF_FORWARD_BLOCK_SIZE
    return _DEFAULT_BLOCK_SIZE


def leaky_relu(A, negative_slope: float = 0.01):
    logger.debug("GEMS LEAKY_RELU")
    if _can_use_fast_kernel(A):
        out = torch.empty_like(A)
        n_elements = A.numel()
        if n_elements == 0:
            return out
        block_size = _get_block_size(A, is_inplace=False)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(A.device):
            leaky_relu_kernel[grid](
                A, out, n_elements, float(negative_slope), BLOCK_SIZE=block_size
            )
        return out
    return leaky_relu_func(A, negative_slope)


def leaky_relu_(A, negative_slope: float = 0.01):
    logger.debug("GEMS LEAKY_RELU_")
    if _can_use_fast_kernel(A):
        n_elements = A.numel()
        if n_elements == 0:
            return A
        block_size = _get_block_size(A, is_inplace=True)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(A.device):
            leaky_relu_inplace_kernel[grid](
                A, n_elements, float(negative_slope), BLOCK_SIZE=block_size
            )
        return A
    leaky_relu_func(A, negative_slope, out0=A)
    return A


def leaky_relu_backward(
    grad_output, A, negative_slope: float = 0.01, self_is_result: bool = False
):
    logger.debug("GEMS LEAKY_RELU BACKWARD")
    if self_is_result and negative_slope < 0:
        raise RuntimeError(
            "In-place leakyReLu backward calculation is triggered with a negative "
            "slope which is not supported."
        )
    return leaky_relu_backward_func(grad_output, A, negative_slope)
