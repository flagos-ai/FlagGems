import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from ..utils.random_utils import philox_cuda_seed_offset

try:
    tl_rand_dtype = tl.int64

    @triton.jit
    def _rand(seed, offset):
        offset = offset.to(tl_rand_dtype)

    _grid = (1,)
    _seed, _offset = philox_cuda_seed_offset(0)
    _rand[_grid](_seed, _offset)
except Exception:
    tl_rand_dtype = tl.int32

del _grid
del _seed
del _offset


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=[
        "N",
    ],
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    N_BLOCK_SIZE: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl_rand_dtype)
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    X_ptr = X + offset
    Y_ptr = Y + offset
    inp = tl.load(X_ptr, mask=mask, other=0.0)
    philox_offset = philox_offset + offset
    pmask = tl.rand(philox_seed, philox_offset, n_rounds=6) > p
    p = 1.0 / (1.0 - p)
    out = tl.where(pmask, inp * p, 0.0)
    tl.store(Y_ptr, out.to(inp.dtype), mask=mask)


@libentry()
@triton.autotune(
    [
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=[
        "N",
    ],
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_backward_kernel(
    DY,
    DX,
    N,
    p,
    philox_seed,
    philox_offset,
    N_BLOCK_SIZE: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl_rand_dtype)
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    DY_ptr = DY + offset
    DX_ptr = DX + offset
    philox_offset = philox_offset + offset
    pmask = tl.rand(philox_seed, philox_offset, n_rounds=6) > p
    dy = tl.load(DY_ptr, mask=mask, other=0.0)

    output = dy * pmask
    p = 1.0 / (1.0 - p)
    output *= p
    tl.store(DX_ptr, output.to(dy.dtype), mask=mask)


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        logging.debug("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        x = x.contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
        # hence we cannot obtain the per thread offset as in Pytorch.
        increment = N
        philox_seed, philox_offset = philox_cuda_seed_offset(increment)
        dropout_forward_kernel[grid_fn](x, out, N, p, philox_seed, philox_offset)
        ctx.p = p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return out, None

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        logging.debug("GEMS NATIVE DROPOUT BACKWARD")
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        N = grad_outputs.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_backward_kernel[grid_fn](
            grad_outputs, grad_inputs, N, ctx.p, ctx.philox_seed, ctx.philox_offset
        )
        return grad_inputs, None, None


def native_dropout(x, p=0.5, train=True):
    return NativeDropout.apply(x, p, train)
