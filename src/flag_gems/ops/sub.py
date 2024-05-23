import torch
import triton
import triton.language as tl
import logging
from ..utils import libentry


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
def sub_kernel(
    X,
    Y,
    alpha,
    O,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    offset = pid + tl.arange(0, M_BLOCK_SIZE)
    mask = offset < M
    X_ptr = X + offset
    Y_ptr = Y + offset
    O_ptr = O + offset
    X_val = tl.load(X_ptr, mask=mask, other=0.0)
    Y_val = tl.load(Y_ptr, mask=mask, other=0.0)
    O_val = X_val - Y_val * alpha
    tl.store(O_ptr, O_val, mask=mask)


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
def sub_tensor_scalar_kernel(
    X,
    Y_scalar,
    O,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    offset = pid + tl.arange(0, M_BLOCK_SIZE)
    mask = offset < M
    X_ptr = X + offset
    O_ptr = O + offset
    X_val = tl.load(X_ptr, mask=mask, other=0.0)
    O_val = X_val - Y_scalar
    tl.store(O_ptr, O_val, mask=mask)


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
def sub_scalar_tensor_kernel(
    X_scalar,
    Y,
    alpha,
    O,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    offset = pid + tl.arange(0, M_BLOCK_SIZE)
    mask = offset < M
    Y_ptr = Y + offset
    O_ptr = O + offset
    Y_val = tl.load(Y_ptr, mask=mask, other=0.0)
    O_val = X_scalar - Y_val * alpha
    tl.store(O_ptr, O_val, mask=mask)


def sub(A, B, *, alpha=1):
    logging.debug("GEMS SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        try:
            A, B = torch.broadcast_tensors(A, B)
        except RuntimeError as e:
            logging.error(
                f"Sub: Tensor shape {A.shape} and tensor shape {B.shape} cannot broadcast to each other."
            )
        A = A.contiguous()
        B = B.contiguous()
        O = torch.empty_like(A)
        M = A.numel()
        grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        sub_kernel[grid_fn](A, B, alpha, O, M)
        return O
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        O = torch.empty_like(A)
        M = A.numel()
        grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        sub_tensor_scalar_kernel[grid_fn](A, B * alpha, O, M)
        return O
    elif isinstance(B, torch.Tensor):
        B = B.contiguous()
        O = torch.empty_like(B)
        M = B.numel()
        grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        sub_scalar_tensor_kernel[grid_fn](A, B, alpha, O, M)
        return O
    else:
        # Both scalar
        return A - B
