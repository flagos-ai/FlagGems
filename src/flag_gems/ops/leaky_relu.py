import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _leaky_relu_autotune_configs():
    return [
        # Tiny tensors (n <= 32K): small blocks
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        # Small-medium tensors (n ~ 64K-4M): 1024-element blocks
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=4),
        # Medium-large tensors (n ~ 4M-16M): 2048-element blocks
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=5),
        # Large tensors (n >= 16M): 4096-element blocks for max bandwidth
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=5),
    ]


@libentry()
@triton.autotune(configs=_leaky_relu_autotune_configs(), key=["n_elements"])
@triton.jit(do_not_specialize=["negative_slope"])
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(x >= 0, x, x * negative_slope)
    tl.store(output_ptr + offsets, output, mask=mask)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU")
    if not A.is_contiguous():
        A = A.contiguous()
    output = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, output, n_elements, negative_slope)
    return output


@libentry()
@triton.autotune(configs=_leaky_relu_autotune_configs(), key=["n_elements"])
@triton.jit(do_not_specialize=["negative_slope"])
def _leaky_relu_backward_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(input_ptr + offsets, mask=mask)
    grad_input = tl.where(x >= 0, grad_output, grad_output * negative_slope)
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def leaky_relu_backward(grad_output, self, negative_slope=0.01, self_is_result=False):
    logger.debug("GEMS LEAKY_RELU BACKWARD")
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not self.is_contiguous():
        self = self.contiguous()
    grad_input = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return grad_input
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(self.device.index):
        _leaky_relu_backward_kernel[grid](
            grad_output, self, grad_input, n_elements, negative_slope
        )
    return grad_input


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU_")
    if not A.is_contiguous():
        raise RuntimeError(
            "leaky_relu_ requires a contiguous tensor for in-place operation"
        )
    n_elements = A.numel()
    if n_elements == 0:
        return A
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, A, n_elements, negative_slope)
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu(A, negative_slope)
    if not A.is_contiguous():
        A = A.contiguous()
    n_elements = A.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, out, n_elements, negative_slope)
    return out
