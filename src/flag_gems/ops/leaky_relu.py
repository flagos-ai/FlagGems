<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
LeakyReLU operator implementation for FlagGems.

This module provides the LeakyReLU activation function using native Triton kernels.
LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def leaky_relu_func(x, negative_slope):
    y = tl.where(x >= 0, x, x * negative_slope)
    return y


def leaky_relu(A, negative_slope=0.01):
    """
    Apply LeakyReLU activation function element-wise.

    Args:
        A (Tensor): Input tensor
        negative_slope (float): Slope for negative values (default: 0.01)

    Returns:
        Tensor: Output tensor with LeakyReLU applied
    """
    logger.debug("GEMS LEAKY_RELU")
    return leaky_relu_func(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    """
    In-place LeakyReLU activation function.

    Args:
        A (Tensor): Input tensor (modified in-place)
        negative_slope (float): Slope for negative values (default: 0.01)

    Returns:
        Tensor: The same tensor A containing LeakyReLU applied
    """
    logger.debug("GEMS LEAKY_RELU_")
    leaky_relu_func(A, negative_slope, out0=A)
    return A
=======
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
>>>>>>> master
