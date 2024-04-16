import torch
import triton
import triton.language as tl
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
    key=["N"],
)
@triton.jit
def silu_forward_kernel(
    X,
    Y,
    N,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    Y_ptrs = tl.make_block_ptr(
        Y,
        shape=(N,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    X_ptrs = tl.make_block_ptr(
        X,
        shape=(N,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    X_val = tl.load(X_ptrs)
    X_val_fp32 = X_val.to(tl.float32)
    Y_val = tl.fdiv(X_val_fp32, (1.0 + tl.exp(-X_val_fp32)))
    tl.store(Y_ptrs, Y_val.to(X_val.dtype))


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
    key=["N"],
)
@triton.jit
def silu_backward_kernel(
    DY,
    X,
    DX,
    N,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    dY_ptrs = tl.make_block_ptr(
        DY,
        shape=(N,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    X_ptrs = tl.make_block_ptr(
        X,
        shape=(N,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    dX_ptrs = tl.make_block_ptr(
        DX,
        shape=(N,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    dY_val = tl.load(dY_ptrs)
    X_val = tl.load(X_ptrs)
    dY_val_fp32 = dY_val.to(tl.float32)
    X_val_fp32 = X_val.to(tl.float32)
    sigmoid_val = tl.math.div_rn(1.0, 1.0 + tl.exp(-X_val_fp32))
    dX_val = dY_val_fp32 * sigmoid_val * (1.0 + X_val_fp32 * (1.0 - sigmoid_val))
    tl.store(dX_ptrs, dX_val.to(dY_val.dtype))


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        if __debug__:
            print("GEMS SILU FORWARD")
        A = A.contiguous()
        O = torch.empty_like(A)
        N = A.numel()

        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        silu_forward_kernel[grid_fn](A, O, N)
        ctx.save_for_backward(A)
        return O

    @staticmethod
    def backward(ctx, out_grad):
        if __debug__:
            print("GEMS SILU BACKWARD")
        (inp,) = ctx.saved_tensors
        N = inp.numel()
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out_grad)
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        silu_backward_kernel[grid_fn](out_grad, inp, in_grad, N)
        return in_grad


def silu(A):
    return Silu.apply(A)
