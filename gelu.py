import torch
import triton
import triton.language as tl
from .libentry import libentry
import math

@libentry()
@triton.autotune(configs=[
    triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
    triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
    triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
    triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=["M"]
)
@triton.jit
def gelu_none_kernel(
    X,
    Y,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    X_ptrs = tl.make_block_ptr(X,
                               shape=(M, ),
                               strides=(1, ),
                               offsets=(pid, ),
                               block_shape=(M_BLOCK_SIZE, ),
                               order=(0, ))
    Y_ptrs = tl.make_block_ptr(Y,
                               shape=(M, ),
                               strides=(1, ),
                               offsets=(pid, ),
                               block_shape=(M_BLOCK_SIZE, ),
                               order=(0, ))
    input = tl.load(X_ptrs)
    output = output
    tl.store(Y_ptrs, output.to(input.dtype))


@libentry()
@triton.autotune(configs=[
    triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
    triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
    triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
    triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
    triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=["M"]
)
@triton.jit
def gelu_tanh_kernel(
    X,
    Y,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    X_ptrs = tl.make_block_ptr(X,
                               shape=(M, ),
                               strides=(1, ),
                               offsets=(pid, ),
                               block_shape=(M_BLOCK_SIZE, ),
                               order=(0, ))
    Y_ptrs = tl.make_block_ptr(Y,
                               shape=(M, ),
                               strides=(1, ),
                               offsets=(pid, ),
                               block_shape=(M_BLOCK_SIZE, ),
                               order=(0, ))
    input = tl.load(X_ptrs)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x * x)))
    # sqrt(2/pi) = 0.79788456
    output = 0.5 * input * (
        1 + tl.math.tanh(input * 0.79788456 *
                         (1 + 0.044715 * tl.math.pow(input.to(tl.float32), 2))))
    tl.store(Y_ptrs, output.to(input.dtype))


def gelu(A, approximate='none', out=None):
    print("FLAG GELU")
    A = A.contiguous()
    M = math.prod(A.shape)
    if out == None:
        O = torch.empty_like(A)
    else:
        O = out

    grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]), )
    if approximate == 'none':
        print("NONE GELU")
        print("ERF VERSION NOT IMPLEMENT SO WE USED TAHN VERSION")
        gelu_tanh_kernel[grid_fn](A, O, M)
    elif approximate == 'tanh':
        print("TANH GELU")
        gelu_tanh_kernel[grid_fn](A, O, M)
    else:
        torch.error("approximation type not supported")
    
    return O