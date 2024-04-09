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
def dropout_kernel(
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
    p,
    N_BLOCK_SIZE: tl.constexpr,
):
    n_offset = tl.program_id(0) * N_BLOCK_SIZE
    DY_ptr = tl.make_block_ptr(
        DY,
        shape=(N, ),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE, ),
        order=(0,),
    )
    MASK_ptr = tl.make_block_ptr(
        MASK,
        shape=(N, ),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE, ),
        order=(0,),
    )
    DX_ptr = tl.make_block_ptr(
        DX,
        shape=(N, ),
        strides=(1,),
        offsets=(n_offset,),
        block_shape=(N_BLOCK_SIZE, ),
        order=(0,),
    )
    dy = tl.load(DY_ptr)
    mask = tl.load(MASK_ptr)
    output = dy * mask
    output = output * (1.0 / (1.0 - p))
    tl.store(DX_ptr, output.to(dy.dtype))


def native_dropout(A, p=0.5, train=False):
    if __debug__:
        print("GEMS NATIVE DROPOUT")
    if not train:
        return A
    assert p >= 0.0 and p < 1.0, "p must be in [0, 1)"
    A = A.contiguous()
    O = torch.empty_like(A)
    Mask = torch.empty(A.shape, dtype=torch.int8, device="cuda")
    N = A.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
    dropout_kernel[grid_fn](A, O, Mask, N, p)
    return O, Mask


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        if __debug__:
            print("GEMS DROPOUT FORWARD")
        if not train:
            return x
        assert p >= 0.0 and p < 1.0, "p must be in [0, 1)"
        x = x.contiguous()
        O = torch.empty_like(x)
        Mask = torch.empty(x.shape, dtype=torch.int8, device="cuda")
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_kernel[grid_fn](x, O, Mask, N, p)
        ctx.save_for_backward(Mask)
        ctx.p = p
        return O


    @staticmethod
    def backward(ctx, grad_outputs):
        if __debug__:
            print("GEMS DROPOUT BACKWARD")
        Mask, = ctx.saved_tensors
        p = ctx.p
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        N = grad_outputs.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_backward_kernel[grid_fn](grad_outputs, Mask, grad_inputs, N, p)
        return grad_inputs, None, None


def dropout(x, p=0.5, train=False):
    return Dropout.apply(x, p, train)

