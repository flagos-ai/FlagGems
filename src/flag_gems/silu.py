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
def silu_kernel(
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


def silu(A, *, out=None):
    if __debug__:
        print("FLAG SILU")
    if out == None:
        O = torch.empty_like(A)
    else:
        O = out
    A = A.contiguous()
    M = A.numel()

    grid_fn = lambda meta: (triton.cdiv(M, meta["N_BLOCK_SIZE"]),)
    silu_kernel[grid_fn](A, O, M)
    return O
