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

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.heuristics(runtime.get_heuristic_config("dropout"))
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    dropout_mask,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 8
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    # First set of 4 random numbers
    i4_0 = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK)
    c0_0 = c0 + i4_0
    _O = c0_0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0_0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    # Second set of 4 random numbers
    i4_1 = tl.program_id(0) * BLOCK * 2 + BLOCK + tl.arange(0, BLOCK)
    c0_1 = c0 + i4_1
    _O1 = c0_1 * 0
    r4, r5, r6, r7 = tl.philox(philox_seed, c0_1, c1, _O1, _O1)
    r4 = uint_to_uniform_float(r4)
    r5 = uint_to_uniform_float(r5)
    r6 = uint_to_uniform_float(r6)
    r7 = uint_to_uniform_float(r7)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p
    mask4 = r4 > p
    mask5 = r5 > p
    mask6 = r6 > p
    mask7 = r7 > p
    scale = 1.0 / (1.0 - p)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0)
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0)
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0)
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0)
    x4 = tl.load(X + off_4, mask=off_4 < N, other=0.0)
    x5 = tl.load(X + off_5, mask=off_5 < N, other=0.0)
    x6 = tl.load(X + off_6, mask=off_6 < N, other=0.0)
    x7 = tl.load(X + off_7, mask=off_7 < N, other=0.0)

    y0 = tl.where(mask0, x0 * scale, 0.0)
    y1 = tl.where(mask1, x1 * scale, 0.0)
    y2 = tl.where(mask2, x2 * scale, 0.0)
    y3 = tl.where(mask3, x3 * scale, 0.0)
    y4 = tl.where(mask4, x4 * scale, 0.0)
    y5 = tl.where(mask5, x5 * scale, 0.0)
    y6 = tl.where(mask6, x6 * scale, 0.0)
    y7 = tl.where(mask7, x7 * scale, 0.0)

    tl.store(Y + off_0, y0, mask=off_0 < N)
    tl.store(Y + off_1, y1, mask=off_1 < N)
    tl.store(Y + off_2, y2, mask=off_2 < N)
    tl.store(Y + off_3, y3, mask=off_3 < N)
    tl.store(Y + off_4, y4, mask=off_4 < N)
    tl.store(Y + off_5, y5, mask=off_5 < N)
    tl.store(Y + off_6, y6, mask=off_6 < N)
    tl.store(Y + off_7, y7, mask=off_7 < N)
    tl.store(dropout_mask + off_0, mask0, mask=off_0 < N)
    tl.store(dropout_mask + off_1, mask1, mask=off_1 < N)
    tl.store(dropout_mask + off_2, mask2, mask=off_2 < N)
    tl.store(dropout_mask + off_3, mask3, mask=off_3 < N)
    tl.store(dropout_mask + off_4, mask4, mask=off_4 < N)
    tl.store(dropout_mask + off_5, mask5, mask=off_5 < N)
    tl.store(dropout_mask + off_6, mask6, mask=off_6 < N)
    tl.store(dropout_mask + off_7, mask7, mask=off_7 < N)


@triton.heuristics(runtime.get_heuristic_config("dropout"))
@triton.jit(do_not_specialize=["scale"])
def dropout_backward_kernel(
    DY,
    DX,
    dropout_mask,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    m = tl.load(dropout_mask + offset, mask=mask, other=0)
    dy = tl.load(DY + offset, mask=mask, other=0)
    dx = dy * m * scale
    tl.store(DX + offset, dx, mask=mask)


UNROLL = 8


def dropout(input, p, train=True):
    logger.debug("GEMS NATIVE DROPOUT FORWARD")
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    mask = torch.empty_like(input, dtype=torch.bool)
    N = input.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        dropout_forward_kernel[grid_fn](
            input, out, mask, N, p, philox_seed, philox_offset
        )
    return out, mask


def dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS NATIVE DROPOUT BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    with torch_device_fn.device(grad_output.device):
        dropout_backward_kernel[grid_fn](grad_output, grad_input, mask, N, scale)
    return grad_input
