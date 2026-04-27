import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger("flag_gems.runtime.backend._nvidia.hopper.ops.sqrt")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sqrt"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    output = tl.sqrt(x_fp32)
    output = output.to(output_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, output, mask=mask)


def sqrt(A):
    logger.debug("GEMS_HOPPER SQRT")
    output = torch.empty_like(A)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sqrt_kernel[grid](A, output, n_elements)
    return output


def sqrt_(A):
    logger.debug("GEMS_HOPPER SQRT_")
    output = torch.empty_like(A)
    n_elements = A.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sqrt_kernel[grid](A, output, n_elements)
    A.copy_(output)
    return A
