import triton
import triton.language as tl
import torch
import logging
from ..utils.random_utils import philox_cuda_seed_offset
from ..utils import libentry


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
@triton.jit
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    X_ptr = X + offset
    Y_ptr = Y + offset
    inp = tl.load(X_ptr, mask=mask, other=0.0)
    philox_offset = philox_offset + offset.to(tl.uint64)
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
@triton.jit
def dropout_backward_kernel(
    DY,
    DX,
    N,
    p,
    philox_seed,
    philox_offset,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    DY_ptr = DY + offset
    DX_ptr = DX + offset

    philox_offset = philox_offset + offset.to(tl.uint64)
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
        O = torch.empty_like(x)
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
        # hence we cannot obtain the per thread offset as in Pytorch.
        increment = N
        philox_seed, philox_offset = philox_cuda_seed_offset(increment)
        dropout_forward_kernel[grid_fn](x, O, N, p, philox_seed, philox_offset)
        ctx.p = p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return O, None

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
