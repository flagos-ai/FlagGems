import torch
import triton
import triton.language as tl
from .libentry import libentry


@libentry()
@triton.autotune(configs=[
    triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
    triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
    triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
    triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
    triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
    triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
    triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
    triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=["N"]
)
@triton.jit
def selu_kernel(
    X,
    Y,
    N,
    DTYPE,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    Y_ptrs = tl.make_block_ptr(
        Y,
        shape=(N, ),
        strides=(1, ),
        offsets=(pid, ),
        block_shape=(N_BLOCK_SIZE, ),
        order=(0, ),
    )
    X_ptrs = tl.make_block_ptr(
        X,
        shape=(N, ),
        strides=(1, ),
        offsets=(pid, ),
        block_shape=(N_BLOCK_SIZE, ),
        order=(0, ),
    )
    X_val = tl.load(X_ptrs)
    Y_val = X_val / (1.0 + tl.exp(-X_val.to(tl.float32)))
    tl.store(Y_ptrs, Y_val.to(input.dtype))
    

def selu(A, *, out=None):
    print("FLAG SELU")
    M, N = A.shape
    if out == None:
        O = torch.empty_like(A)
    else:
        O = out
    
    grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]), )
    selu_kernel[grid_fn](A, O, M*N)
    # reshape input data into 2D tensor
    O.reshape(M, N)
    return O