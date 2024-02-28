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
def gelu_kernel(
    X,
    Y,
    N,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * N_BLOCK_SIZE
    X_ptrs = tl.make_block_ptr(X,
                               shape=(N, ),
                               strides=(1, ),
                               offsets=(pid, ),
                               block_shape=(N_BLOCK_SIZE, ),
                               order=(0, ))
    Y_ptrs = tl.make_block_ptr(Y,
                               shape=(N, ),
                               strides=(1, ),
                               offsets=(pid, ),
                               block_shape=(N_BLOCK_SIZE, ),
                               order=(0, ))
    input = tl.load(X_ptrs)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x * x)))
    # sqrt(2/pi) = 0.79788456
    output = 0.5 * input * (
        1 + tl.math.tanh(input * 0.79788456 *
                         (1 + 0.044715 * tl.math.pow(input.to(tl.float32), 2))))
    tl.store(Y_ptrs, output.to(input.dtype))
    

def gelu(A, *, out=None):
    print("FLAG GELU")
    M, N = A.shape
    if out == None:
        O = torch.empty_like(A)
    else:
        O = out
    
    grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]), )
    A = A.view(-1)
    gelu_kernel[grid_fn](A, O, M*N)
    # reshape input data into 2D tensor
    O = O.reshape(M, N)
    return O