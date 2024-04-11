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
def mul_kernel(
    X,
    Y,
    O,
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
    O_ptrs = tl.make_block_ptr(
        O,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    X_val = tl.load(X_ptrs)
    Y_val = tl.load(Y_ptrs)
    O_val = X_val * Y_val
    tl.store(O_ptrs, O_val.to(X_val.dtype))


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
    if __debug__:
        print("GEMS MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A = A.contiguous()
        O = torch.empty_like(A)
        try:
            A, B = torch.broadcast_tensors(A, B)
        except RuntimeError as e:
            print(
                f"Mul: Tensor shape {A.shape} and tensor shape {B.shape} cannot broadcast to each other."
            )
        B = B.contiguous()
        M = A.numel()
        grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        mul_kernel[grid_fn](A, B, O, M)
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
