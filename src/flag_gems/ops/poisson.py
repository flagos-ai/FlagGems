import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@triton.jit
def poisson_small_lambda(lam, seed, c0, c1, z, MAX_ITERS: tl.constexpr):
    """
    Knuth's algorithm for Poisson sampling with small lambda.
    Returns the count of exponential inter-arrival times that sum to <= 1.
    Uses inverse transform: -log(U) / lam for exponential samples.
    """
    # L = exp(-lambda)
    L = tl.exp(-lam)
    k = (lam * 0).to(tl.int32)  # Initialize counter to 0
    p = lam * 0.0 + 1.0  # Initialize p to 1.0

    # We need to iterate. Each iteration we multiply p by a uniform random.
    # Continue while p > L.
    for _ in range(MAX_ITERS):
        # Generate uniform random
        r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
        u = uint_to_uniform_float(r0)
        # Ensure u is not 0 to avoid issues
        u = tl.maximum(u, 1e-10)
        p = p * u
        # Increment counter where p > L
        k = tl.where(p > L, k + 1, k)
        # Update counter for next iteration
        c0 = c0 + 1

    return k.to(tl.float32)


@triton.jit
def poisson_large_lambda(lam, seed, c0, c1, z):
    """
    Normal approximation for Poisson with large lambda.
    Poisson(lambda) ~ N(lambda, lambda) for large lambda.
    Uses Box-Muller transform.
    """
    # Generate two uniform random numbers for Box-Muller
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
    u1 = uint_to_uniform_float(r0)
    u2 = uint_to_uniform_float(r1)

    # Avoid log(0)
    u1 = tl.maximum(u1, 1e-10)

    # Box-Muller transform for standard normal
    two_pi = 6.283185307179586
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = two_pi * u2
    normal_sample = r * tl.cos(theta)

    # Transform to Poisson approximation: mean=lam, std=sqrt(lam)
    result = lam + tl.sqrt(lam) * normal_sample

    # Poisson must be non-negative integer
    result = tl.maximum(result, 0.0)
    result = tl.floor(result + 0.5)  # Round to nearest integer

    return result


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel(
    inp_ptr,
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    LAMBDA_THRESHOLD: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    """
    Poisson sampling kernel.
    For each input lambda:
    - If lambda < LAMBDA_THRESHOLD: use Knuth's algorithm
    - Otherwise: use normal approximation
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0_base = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load input lambda values
    lam = tl.load(inp_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Clamp lambda to non-negative
    lam = tl.maximum(lam, 0.0)

    # Use different algorithms based on lambda size
    use_small = lam < LAMBDA_THRESHOLD

    # For small lambda: Knuth's algorithm
    # Each thread needs its own random state offset based on position and iteration count
    c0_small = c0_base + offs.to(tl.uint32) * MAX_ITERS
    z = c0_small * 0
    small_result = poisson_small_lambda(lam, philox_seed, c0_small, c1, z, MAX_ITERS)

    # For large lambda: normal approximation
    c0_large = c0_base + offs.to(tl.uint32)
    z_large = c0_large * 0
    large_result = poisson_large_lambda(lam, philox_seed, c0_large, c1, z_large)

    # Select result based on lambda size
    result = tl.where(use_small, small_result, large_result)

    tl.store(out_ptr + offs, result, mask=mask)


def poisson(input, generator=None):
    """
    Returns a tensor of the same size as input with each element sampled
    from a Poisson distribution with rate parameter given by the corresponding
    element in input.

    Args:
        input (Tensor): the input tensor containing the rates of the Poisson distribution
        generator (torch.Generator, optional): a pseudorandom number generator for sampling

    Returns:
        Tensor: output tensor with Poisson samples
    """
    logger.debug("GEMS POISSON")

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"Unsupported dtype: {input.dtype}"

    # Ensure input is contiguous
    inp = input.contiguous()
    N = volume(inp.shape)

    # Create output tensor with same shape and dtype as input
    out = torch.empty_like(inp)

    if N == 0:
        return out

    # Parameters for the algorithm
    LAMBDA_THRESHOLD = 30  # Threshold for switching between algorithms
    MAX_ITERS = 64  # Maximum iterations for Knuth's algorithm

    # Calculate grid
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)

    # Get random seed and offset
    # Each element may need up to MAX_ITERS random numbers for small lambda case
    increment = triton.cdiv(N * MAX_ITERS, 4)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    with torch_device_fn.device(inp.device):
        poisson_kernel[grid](
            inp,
            out,
            N,
            philox_seed,
            philox_offset,
            LAMBDA_THRESHOLD=LAMBDA_THRESHOLD,
            MAX_ITERS=MAX_ITERS,
        )

    return out
