import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def conj_physical__kernel(ptr, n_real_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_real_elements
    x = tl.load(ptr + offsets, mask=mask)
    # even offsets are real parts (kept), odd offsets are imag parts (negated)
    y = tl.where((offsets % 2) == 1, -x, x)
    tl.store(ptr + offsets, y, mask=mask)


def conj_physical_(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS CONJ_PHYSICAL_")
    if not input.is_complex():
        return input

    if not input.is_contiguous():
        raise RuntimeError(
            "conj_physical_ only supports contiguous tensors. "
            "Please call .contiguous() before this operation."
        )

    # view complex64/128 as a flat float32/64 buffer of 2 * numel elements
    flat = torch.view_as_real(input).view(-1)
    n = flat.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    conj_physical__kernel[grid](flat, n, BLOCK_SIZE=BLOCK_SIZE)

    return input
