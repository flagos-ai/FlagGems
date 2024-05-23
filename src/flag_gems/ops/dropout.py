import triton
import triton.language as tl
import torch
import logging
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
    Mask,
    N,
    p,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    X_ptr = X + offset
    Y_ptr = Y + offset
    Mask_ptr = Mask + offset

    inp = tl.load(X_ptr, mask=mask, other=0.0)
    # random seed (lucky number)
    seed = 7
    pmask = tl.rand(seed, offset) > p
    output = tl.where(pmask, inp, 0.0)
    output = output * (1.0 / (1.0 - p))
    tl.store(Y_ptr, output.to(inp.dtype), mask=mask)
    tl.store(Mask_ptr, pmask.to(tl.int8), mask=mask)


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
    MASK,
    DX,
    N,
    scale,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    DY_ptr = DY + offset
    MASK_ptr = MASK + offset
    DX_ptr = DX + offset

    dy = tl.load(DY_ptr, mask=mask, other=0.0)
    Mask = tl.load(MASK_ptr, mask=mask, other=0.0)
    output = dy * Mask
    output = output * scale
    tl.store(DX_ptr, output.to(dy.dtype), mask=mask)


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        logging.debug("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        x = x.contiguous()
        O = torch.empty_like(x)
        Mask = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_forward_kernel[grid_fn](x, O, Mask, N, p)
        ctx.save_for_backward(Mask)
        ctx.p = p
        return O, Mask

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        logging.debug("GEMS NATIVE DROPOUT BACKWARD")
        (Mask,) = ctx.saved_tensors
        scale = 1.0 / (1.0 - ctx.p)
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        N = grad_outputs.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_backward_kernel[grid_fn](grad_outputs, Mask, grad_inputs, N, scale)
        return grad_inputs, None, None


def native_dropout(x, p=0.5, train=True):
    return NativeDropout.apply(x, p, train)
