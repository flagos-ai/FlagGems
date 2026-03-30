import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def clamp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val,
    max_val,
    has_min: tl.constexpr,
    has_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles UNROLL * BLOCK_SIZE elements
    base = pid * (BLOCK_SIZE * UNROLL)
    for i in tl.static_range(UNROLL):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        if has_min:
            x = tl.maximum(x, min_val)
        if has_max:
            x = tl.minimum(x, max_val)
        tl.store(output_ptr + offsets, x, mask=mask)


@libentry()
@triton.jit
def clamp_min_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK_SIZE * UNROLL)
    for i in tl.static_range(UNROLL):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        x = tl.maximum(x, min_val)
        tl.store(output_ptr + offsets, x, mask=mask)


@libentry()
@triton.jit
def clamp_max_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    max_val,
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK_SIZE * UNROLL)
    for i in tl.static_range(UNROLL):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        x = tl.minimum(x, max_val)
        tl.store(output_ptr + offsets, x, mask=mask)


def clamp(A, mini=None, maxi=None):
    """Clamp all elements in input into the range [mini, maxi].

    Args:
        A: The input tensor.
        mini: The lower-bound of the range to be clamped to.
        maxi: The upper-bound of the range to be clamped to.

    Returns:
        A tensor where each element is clamped to [mini, maxi].
    """
    logger.debug("GEMS_ILUVATAR CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")

    A = A.contiguous()
    output = torch.empty_like(A)
    n_elements = output.numel()

    # Empty tensor protection
    if n_elements == 0:
        return output

    BLOCK_SIZE = 1024
    UNROLL = 8
    elements_per_program = BLOCK_SIZE * UNROLL
    grid = ((n_elements + elements_per_program - 1) // elements_per_program,)

    has_min = mini is not None
    has_max = maxi is not None

    if has_min and has_max:
        clamp_kernel[grid](
            A,
            output,
            n_elements,
            float(mini),
            float(maxi),
            has_min=True,
            has_max=True,
            BLOCK_SIZE=BLOCK_SIZE,
            UNROLL=UNROLL,
            num_warps=4,
            num_stages=4,
        )
    elif has_min:
        clamp_min_kernel[grid](
            A,
            output,
            n_elements,
            float(mini),
            BLOCK_SIZE=BLOCK_SIZE,
            UNROLL=UNROLL,
            num_warps=4,
            num_stages=4,
        )
    else:  # has_max only
        clamp_max_kernel[grid](
            A,
            output,
            n_elements,
            float(maxi),
            BLOCK_SIZE=BLOCK_SIZE,
            UNROLL=UNROLL,
            num_warps=4,
            num_stages=4,
        )

    return output


def clamp_(A, mini=None, maxi=None):
    """In-place version of clamp."""
    logger.debug("GEMS_ILUVATAR CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")

    A = A.contiguous()
    n_elements = A.numel()

    # Empty tensor protection
    if n_elements == 0:
        return A

    BLOCK_SIZE = 1024
    UNROLL = 8
    elements_per_program = BLOCK_SIZE * UNROLL
    grid = ((n_elements + elements_per_program - 1) // elements_per_program,)

    has_min = mini is not None
    has_max = maxi is not None

    if has_min and has_max:
        clamp_kernel[grid](
            A,
            A,
            n_elements,
            float(mini),
            float(maxi),
            has_min=True,
            has_max=True,
            BLOCK_SIZE=BLOCK_SIZE,
            UNROLL=UNROLL,
            num_warps=4,
            num_stages=4,
        )
    elif has_min:
        clamp_min_kernel[grid](
            A,
            A,
            n_elements,
            float(mini),
            BLOCK_SIZE=BLOCK_SIZE,
            UNROLL=UNROLL,
            num_warps=4,
            num_stages=4,
        )
    else:  # has_max only
        clamp_max_kernel[grid](
            A,
            A,
            n_elements,
            float(maxi),
            BLOCK_SIZE=BLOCK_SIZE,
            UNROLL=UNROLL,
            num_warps=4,
            num_stages=4,
        )

    return A


def clamp_min(A, mini):
    """Clamp all elements in input to be larger than mini.

    Args:
        A: The input tensor.
        mini: The lower-bound of the range to be clamped to.

    Returns:
        A tensor where each element is at least mini.
    """
    logger.debug("GEMS_ILUVATAR CLAMP_MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp(A, mini=mini, maxi=None)


def clamp_min_(A, mini):
    """In-place version of clamp_min."""
    logger.debug("GEMS_ILUVATAR CLAMP_MIN_")
    if mini is None:
        raise ValueError("Mini must not be None")
    return clamp_(A, mini=mini, maxi=None)


def clamp_max(A, maxi):
    """Clamp all elements in input to be smaller than maxi.

    Args:
        A: The input tensor.
        maxi: The upper-bound of the range to be clamped to.

    Returns:
        A tensor where each element is at most maxi.
    """
    logger.debug("GEMS_ILUVATAR CLAMP_MAX")
    if maxi is None:
        raise ValueError("Maxi must not be None")
    return clamp(A, mini=None, maxi=maxi)


def clamp_max_(A, maxi):
    """In-place version of clamp_max."""
    logger.debug("GEMS_ILUVATAR CLAMP_MAX_")
    if maxi is None:
        raise ValueError("Maxi must not be None")
    return clamp_(A, mini=None, maxi=maxi)
