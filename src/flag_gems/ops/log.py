import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

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
def _log_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    # Promote fp16/bf16 to fp32 to match torch.log's accumulator behaviour.
    y = tl.log(x.to(tl.float32)).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def _launch(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = inp.numel()
    if n == 0:
        return out
    if not inp.is_contiguous():
        inp = inp.contiguous()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _log_kernel[grid](inp, out, n)
    return out


def log(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS LOG")
    if not A.is_floating_point():
        A = A.to(torch.float32)
    return _launch(A, torch.empty_like(A))
