import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


# SELU constants for alpha dropout
LAMBDA: tl.constexpr = 1.0507010293579108
ALPHA: tl.constexpr = 1.6732632423543772


def compute_alpha_dropout_scales(p):
    """
    Compute the scaling factors for alpha dropout.

    The scaling factors are computed such that:
    - E[output] = 0 for standardized input
    - Var[output] = 1 for standardized input

    Using the formula:
    - keep_scale = LAMBDA * sqrt(p / (1-p))
    - drop_scale = -LAMBDA * sqrt(p * (1-p))

    This is derived from the self-normalizing property of alpha dropout.
    """
    keep_scale = LAMBDA * triton.math.sqrt(p / (1 - p))
    drop_scale = -LAMBDA * triton.math.sqrt(p * (1 - p))
    return keep_scale, drop_scale


@triton.heuristics(runtime.get_heuristic_config("dropout"))
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def alpha_dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    # Generate mask: keep when random > p
    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p

    # Compute alpha dropout scaling factors
    # keep_scale = LAMBDA * sqrt(p / (1-p))
    # drop_scale = -LAMBDA * sqrt(p * (1-p))
    keep_scale = LAMBDA * tl.sqrt(p / (1.0 - p))
    drop_scale = -LAMBDA * tl.sqrt(p * (1.0 - p))

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    # Apply alpha dropout: keep_scale for kept elements, drop_scale for dropped
    y0 = tl.where(mask0, x0 * keep_scale, x0 * drop_scale)
    y1 = tl.where(mask1, x1 * keep_scale, x1 * drop_scale)
    y2 = tl.where(mask2, x2 * keep_scale, x2 * drop_scale)
    y3 = tl.where(mask3, x3 * keep_scale, x3 * drop_scale)

    tl.store(Y + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(Y + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(Y + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(Y + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4


def alpha_dropout(input, p=0.5, train=True):
    logger.debug("GEMS ALPHA_DROPOUT FORWARD")
    if not train or p == 0:
        return input.clone()
    if p == 1:
        return torch.zeros_like(input)

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    N = input.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        alpha_dropout_forward_kernel[grid_fn](
            input, out, N, p, philox_seed, philox_offset
        )
    return out
