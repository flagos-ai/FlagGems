import torch
import triton
import triton.language as tl
from .__libentry__ import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=["M"],
)
@triton.jit
def rsqrt_kernel(
    X,
    Y,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    Y_ptrs = tl.make_block_ptr(
        Y,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    X_ptrs = tl.make_block_ptr(
        X,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    X_val = tl.load(X_ptrs)
    Y_val = 1.0 / tl.sqrt(X_val.to(tl.float32))
    tl.store(Y_ptrs, Y_val.to(X_val.dtype))


def rsqrt(A):
    if __debug__:
        print("GEMS RSQRT")
    A = A.contiguous()
    O = torch.empty_like(A)
    M = A.numel()
    grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
    rsqrt_kernel[grid_fn](A, O, M)
    return O
