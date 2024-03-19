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
    inp = tl.load(X_ptr)
    # random seed (lucky number)
    seed = 7
    pmask = tl.rand(seed, n_offset + tl.arange(0, N_BLOCK_SIZE)) > p
    output = tl.where(pmask, inp, 0.0)
    output = output * (1.0 / (1.0 - p))
    tl.store(Y_ptr, output.to(inp.dtype))


def dropout(A, p=0.5, train=False):
    if __debug__:
        print("FLAG DROPOUT")
    if not train:
        return A
    assert p >= 0.0 and p < 1.0, "p must be in [0, 1)"
    # training not supported
    O = torch.empty_like(A)
    A = A.contiguous()
    N = A.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
    dropout_kernel[grid_fn](A, O, N, p)
    return O
