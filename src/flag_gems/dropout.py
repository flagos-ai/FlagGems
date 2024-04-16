import triton
import triton.language as tl
import torch
from .__libentry__ import libentry


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
    n_offset = tl.program_id(0) * N_BLOCK_SIZE
    X_ptr = tl.make_block_ptr(
        X,
        shape=(N,),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    Y_ptr = tl.make_block_ptr(
        Y,
        shape=(N,),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    Mask_ptr = tl.make_block_ptr(
        Mask,
        shape=(N,),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    inp = tl.load(X_ptr)
    # random seed (lucky number)
    seed = 7
    pmask = tl.rand(seed, n_offset + tl.arange(0, N_BLOCK_SIZE)) > p
    output = tl.where(pmask, inp, 0.0)
    output = output * (1.0 / (1.0 - p))
    tl.store(Y_ptr, output.to(inp.dtype))
    tl.store(Mask_ptr, pmask.to(tl.int8))


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
    n_offset = tl.program_id(0) * N_BLOCK_SIZE
    DY_ptr = tl.make_block_ptr(
        DY,
        shape=(N,),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    MASK_ptr = tl.make_block_ptr(
        MASK,
        shape=(N,),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    DX_ptr = tl.make_block_ptr(
        DX,
        shape=(N,),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    dy = tl.load(DY_ptr)
    mask = tl.load(MASK_ptr)
    output = dy * mask
    output = output * scale
    tl.store(DX_ptr, output.to(dy.dtype))


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        if __debug__:
            print("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        x = x.contiguous()
        O = torch.empty_like(x)
        Mask = torch.empty(x.shape, dtype=torch.bool, device="cuda")
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_forward_kernel[grid_fn](x, O, Mask, N, p)
        ctx.save_for_backward(Mask)
        ctx.p = p
        return O, Mask

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        if __debug__:
            print("GEMS NATIVE DROPOUT BACKWARD")
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
