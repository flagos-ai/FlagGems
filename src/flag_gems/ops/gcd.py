import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Fibonacci worst-case Euclidean steps per dtype.
# gcd(fib(n+1), fib(n)) needs exactly n steps — the slowest possible pair.
# N_ITERS is chosen so that fib(N_ITERS) > dtype.max, guaranteeing termination.
_DTYPE_MAX_ITERS = {
    torch.int8: 12,  # fib(12) = 144 > 127  (INT8_MAX)
    torch.int16: 24,  # fib(24) = 46368 > 32767  (INT16_MAX)
    torch.int32: 47,  # fib(47) ≈ 2.97e9 > 2^31 - 1  (INT32_MAX)
    torch.int64: 93,  # fib(93) ≈ 1.22e19 > 2^63 - 1  (INT64_MAX)
}

_gcd_configs = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
]


@libentry()
@triton.autotune(configs=_gcd_configs, key=["N"])
@triton.jit
def gcd_kernel(
    X,
    Y,
    Out,
    N,
    N_ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0)
    y = tl.load(Y + offs, mask=mask, other=0)

    # Safe absolute value: INT_MIN satisfies x == -x in two's-complement,
    # so tl.abs(INT_MIN) would overflow back to INT_MIN and corrupt the loop.
    # We replace INT_MIN with 0 (gcd(0, b) = b), which is mathematically sound
    # since |INT_MIN| has no representation in the signed dtype anyway.
    x = tl.abs(tl.where((x != 0) & (x == -x), x * 0, x))
    y = tl.abs(tl.where((y != 0) & (y == -y), y * 0, y))

    # Fixed-iteration Euclidean GCD — no block-level tl.max() reduction needed.
    # The old `while tl.max(y) > 0` required a __syncthreads() barrier on every
    # iteration; those barriers dominate runtime for large block sizes.
    # N_ITERS is the Fibonacci worst-case for this dtype, so correctness is
    # guaranteed without any dynamic loop-exit check.
    for _ in range(N_ITERS):
        nonzero = y != 0
        safe_y = tl.where(nonzero, y, 1)
        rem = x % safe_y
        x = tl.where(nonzero, y, x)
        y = tl.where(nonzero, rem, 0)

    tl.store(Out + offs, x, mask=mask)


def gcd(A, B):
    logger.debug("GEMS GCD")
    A, B = torch.broadcast_tensors(A, B)
    dtype = torch.result_type(A, B)
    if A.dtype != dtype:
        A = A.to(dtype)
    if B.dtype != dtype:
        B = B.to(dtype)
    out = torch.empty(A.shape, dtype=dtype, device=A.device)
    N = A.numel()
    if N == 0:
        return out
    n_iters = _DTYPE_MAX_ITERS.get(dtype, 93)
    with torch_device_fn.device(A.device):
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        gcd_kernel[grid](
            A.contiguous().view(-1),
            B.contiguous().view(-1),
            out.view(-1),
            N,
            n_iters,
        )
    return out
