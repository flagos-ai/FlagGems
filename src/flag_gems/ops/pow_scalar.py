import torch
import triton
import triton.language as tl
import logging
from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=["M"],
)
@triton.jit
def pow_scalar_kernel(
    X_val,
    exponent,
    Y,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    offset = pid + tl.arange(0, M_BLOCK_SIZE)
    mask = offset < M
    exp_ptrs = exponent + offset
    Y_ptrs = Y + offset
    exp_val = tl.load(exp_ptrs, mask=mask, other=0.0)
    Y_val = tl.math.pow(X_val, exp_val)
    tl.store(Y_ptrs, Y_val, mask=mask)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    exponent = exponent.contiguous()
    O = torch.empty_like(exponent)
    M = exponent.numel()
    grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
    pow_scalar_kernel[grid_fn](A, exponent, O, M)
    return O
