import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Bandwidth-bound element-wise op. Sweep block size + warps so we can stay
# close to peak DRAM bandwidth across small / medium / large tensors.
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
def _log10_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    # log10(x) = ln(x) * (1 / ln 10). Promote fp16/bf16 to fp32 to keep
    # numerics aligned with torch.log10 (which also accumulates in fp32).
    y = (tl.log(x.to(tl.float32)) * 0.4342944819032518).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


@libentry()
@triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
@triton.jit
def _log10_kernel_f64(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # fp64 path: do not down-cast the accumulator.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    y = tl.log(x) * 0.4342944819032518
    tl.store(y_ptr + offs, y, mask=mask)


def _launch(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = inp.numel()
    if n == 0:
        return out
    if not inp.is_contiguous():
        inp = inp.contiguous()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    if inp.dtype == torch.float64:
        _log10_kernel_f64[grid](inp, out, n)
    else:
        _log10_kernel[grid](inp, out, n)
    return out


def log10(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS LOG10")
    if not A.is_floating_point():
        # Match torch.log10 integer promotion to fp32.
        A = A.to(torch.float32)
    return _launch(A, torch.empty_like(A))


def log10_(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS LOG10_")
    if not A.is_floating_point():
        raise RuntimeError("log10_: in-place op requires a floating-point tensor")
    return _launch(A, A)


def log10_out(A: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS LOG10_OUT")
    if not A.is_floating_point():
        A = A.to(out.dtype)
    return _launch(A, out)
