import logging

import torch
import triton
import triton.language as tl

from ..utils.random_utils import philox_cuda_seed_offset, uint_to_uniform_float

UNROLL = 4


@triton.heuristics(
    values={
        "BLOCK": lambda args: 512 if args["N"] <= 512 else 1024,
        "num_warps": lambda args: 4 if args["N"] <= 512 else 8 if args["N"] <= 1024 else 16,  # fmt: skip
    }
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
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

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p
    p = 1.0 / (1.0 - p)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    y0 = x0 * p * mask0  # tl.where(mask0, x0 * p, 0.0)
    y1 = x1 * p * mask1  # tl.where(mask1, x1 * p, 0.0)
    y2 = x2 * p * mask2  # tl.where(mask2, x2 * p, 0.0)
    y3 = x3 * p * mask3  # tl.where(mask3, x3 * p, 0.0)

    tl.store(Y + off_0, y0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(Y + off_1, y1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(Y + off_2, y2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(Y + off_3, y3, mask=off_3 < N, eviction_policy="evict_first")


@triton.heuristics(
    values={
        "BLOCK": lambda args: 512 if args["N"] <= 512 else 1024,
        "num_warps": lambda args: 4 if args["N"] <= 512 else 8 if args["N"] <= 1024 else 16,  # fmt: skip
    }
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_backward_kernel(
    DY,
    DX,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL = 4
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

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p
    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    dy_0 = tl.load(DY + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    dy_1 = tl.load(DY + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    dy_2 = tl.load(DY + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    dy_3 = tl.load(DY + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    p = 1.0 / (1.0 - p)
    dx_0 = p * dy_0 * mask0
    dx_1 = p * dy_1 * mask1
    dx_2 = p * dy_2 * mask2
    dx_3 = p * dy_3 * mask3

    tl.store(DX + off_0, dx_0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(DX + off_1, dx_1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(DX + off_2, dx_2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(DX + off_3, dx_3, mask=off_3 < N, eviction_policy="evict_first")


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        logging.debug("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        device = x.device
        x = x.contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
        # hence we cannot obtain the per thread offset as in Pytorch.
        increment = triton.cdiv(N, UNROLL)
        with torch.cuda.device(device):
            philox_seed, philox_offset = philox_cuda_seed_offset(increment)
            dropout_forward_kernel[grid_fn](x, out, N, p, philox_seed, philox_offset)
        ctx.p = p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return out, None

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        logging.debug("GEMS NATIVE DROPOUT BACKWARD")
        device = grad_outputs.device
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        N = grad_outputs.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        with torch.cuda.device(device):
            dropout_backward_kernel[grid_fn](
                grad_outputs, grad_inputs, N, ctx.p, ctx.philox_seed, ctx.philox_offset
            )
        return grad_inputs, None, None


def native_dropout(x, p=0.5, train=True):
    return NativeDropout.apply(x, p, train)
