import torch
import triton
import triton.language as tl
import logging
from ..utils import libentry, pointwise_dynamic


@pointwise_dynamic
@triton.jit
def mul_func(x, y):
    return x * y


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
def mul_scalar_kernel(
    X,
    Y_scalar,
    O,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    X_ptrs = tl.make_block_ptr(
        X,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    O_ptrs = tl.make_block_ptr(
        O,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    X_val = tl.load(X_ptrs)
    O_val = X_val * Y_scalar
    tl.store(O_ptrs, O_val.to(X_val.dtype))


def mul(A, B):
    logging.debug("GEMS MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        O = mul_func(A, B)
        return O
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        O = torch.empty_like(A)
        M = A.numel()
        grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        mul_scalar_kernel[grid_fn](A, B, O, M)
        return O
    elif isinstance(B, torch.Tensor):
        B = B.contiguous()
        O = torch.empty_like(B)
        M = B.numel()
        grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        mul_scalar_kernel[grid_fn](B, A, O, M)
        return O
    else:
        # Both scalar
        return A * B
