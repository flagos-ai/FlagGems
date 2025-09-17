import logging

import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import philox as _philox

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

from ..utils import TOTAL_CORE_NUM

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
    UNROLL: tl.constexpr = 4  # philox generate 128 random bits at a time
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    i4_start = pid * BLOCK
    block_start = pid * UNROLL * BLOCK
    step = num_jobs * BLOCK * UNROLL
    mp = 1.0 / (1.0 - p)

    for block_offset in range(block_start, N, step):
        sl = (philox_seed & 0xFFFFFFFF).to(tl.uint32)
        sh = ((philox_seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        r = _philox(BLOCK, sl, sh, c0 + i4_start, c1, 0, 0, 10)
        r = uint_to_uniform_float(r)

        mask = r > p

        off = block_offset + tl.arange(0, UNROLL * BLOCK)
        x = tl.load(X + off, mask=off < N, other=0.0)
        y = (
            x * mp * tl.reshape(mask, [UNROLL * BLOCK], can_reorder=True)
        )  # tl.where(mask0, x0 * p, 0.0)
        mask_reshaped = tl.reshape(mask, [UNROLL * BLOCK], can_reorder=True)
        tl.store(dropout_mask + off, mask_reshaped, mask=off < N)
        tl.store(Y + off, y, mask=off < N)
        i4_start += num_jobs * BLOCK


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
    UNROLL: tl.constexpr = 4

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_start = pid * UNROLL * BLOCK
    step = num_programs * UNROLL * BLOCK
    for block_offset in range(block_start, N, step):
        off = block_offset + tl.arange(0, UNROLL * BLOCK)
        mask = tl.load(
            dropout_mask + off, mask=off < N, other=0, eviction_policy="evict_first"
        )
        dy = tl.load(DY + off, mask=off < N, other=0.0, eviction_policy="evict_first")
        dx = dy * mask * scale

        tl.store(DX + off, dx, mask=off < N, eviction_policy="evict_first")


UNROLL = 4


def dropout(input, p, train=True):
    logger.debug("GEMS_CAMBRICON NATIVE DROPOUT FORWARD")
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
    # TODO: remove contiguous enforcement
    input = input.contiguous()
    out = torch.empty_like(input)
    mask = torch.empty_like(input, dtype=torch.bool)
    N = input.numel()
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
    # hence we cannot obtain the per thread offset as in Pytorch.
    increment = triton.cdiv(N, UNROLL)
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        dropout_forward_kernel[grid_fn](
            input,
            out,
            mask,
            N,
            p,
            philox_seed,
            philox_offset,
            num_warps=1,
            num_stages=3,
        )
    return out, mask


def dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS_CAMBRICON NATIVE DROPOUT BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    grid_fn = lambda meta: (
        min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),
    )
    with torch_device_fn.device(grad_output.device):
        dropout_backward_kernel[grid_fn](
            grad_output,
            grad_input,
            mask,
            N,
            scale,
            num_stages=3,
            num_warps=1,
        )
    return grad_input
