import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Bandwidth-bound element-wise op. Explicit autotune sweep so the chosen
# tile size adapts to small vs large tensors instead of relying on a
# single template default.
_AUTOTUNE_CFGS = [
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
]


@libentry()
@triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
@triton.jit
def _abs_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.abs(x), mask=mask)


def _launch(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = inp.numel()
    if n == 0:
        return out
    if not inp.is_contiguous():
        inp = inp.contiguous()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _abs_kernel[grid](inp, out, n)
    return out


def abs(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ABS")
    return _launch(A, torch.empty_like(A))


def abs_(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ABS_")
    return _launch(A, A)
