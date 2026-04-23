import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def pow_scalar_kernel(
    output,
    input_exponent,
    sbase,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for pow_scalar: output = sbase ** input_exponent
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load exponent values
    exponent = tl.load(input_exponent + offsets, mask=mask, other=0.0)

    # Compute sbase ** exponent
    result = tl.pow(sbase.to(tl.float32), exponent.to(tl.float32))

    # Store result
    tl.store(output + offsets, result, mask=mask)


@libentry()
@triton.jit
def pow_scalar_inplace_kernel(
    input_output,
    sbase,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for in-place pow_scalar_: input_output = sbase ** input_output
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load exponent values
    exponent = tl.load(input_output + offsets, mask=mask, other=0.0)

    # Compute sbase ** exponent
    result = tl.pow(sbase.to(tl.float32), exponent.to(tl.float32))

    # Store result in-place
    tl.store(input_output + offsets, result, mask=mask)


def pow_scalar(A, exponent):
    """
    Computes base^exponent where base is a scalar and exponent is a tensor.

    Optimized Triton kernel for Iluvatar platform with BLOCK_SIZE=2048.

    Args:
        A: Scalar base value
        exponent: Exponent tensor

    Returns:
        Output tensor with same shape as exponent
    """
    logger.debug("GEMS_ILUVATAR POW_SCALAR")

    # Handle empty tensor
    if volume(exponent.shape) == 0:
        return torch.empty_like(exponent)

    output = torch.empty_like(exponent)
    n_elements = volume(exponent.shape)

    # Convert scalar base to float32 for computation
    sbase = float(A)

    # Grid size
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    pow_scalar_kernel[grid](output, exponent, sbase, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output


def pow_scalar_(A, exponent):
    """
    In-place version of pow_scalar.

    Args:
        A: Scalar base value
        exponent: Exponent tensor (modified in-place)

    Returns:
        The modified exponent tensor
    """
    logger.debug("GEMS_ILUVATAR POW_SCALAR_")

    # Handle empty tensor
    if volume(exponent.shape) == 0:
        return exponent

    n_elements = volume(exponent.shape)

    # Convert scalar base to float32 for computation
    sbase = float(A)

    # Grid size
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    pow_scalar_inplace_kernel[grid](exponent, sbase, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return exponent
