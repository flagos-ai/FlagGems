import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("div_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def div_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """tensor / tensor, using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        b_blk = tl.make_block_ptr(
            base=B_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        a_dtype = a.dtype
        a = a.to(tl.float32)
        b = tl.load(b_blk, boundary_check=(0,)).to(tl.float32)
        out = a / b
        tl.store(out_blk, out.to(a_dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("div_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def div_kernel_ts(
    A_ptr,
    Out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """tensor / scalar, using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        a_dtype = a.dtype
        a = a.to(tl.float32)
        out = a / scalar
        tl.store(out_blk, out.to(a_dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("div_scalar_tensor"),
    key=["n_elements"],
)
@triton.jit
def div_kernel_st(
    B_ptr,
    Out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """scalar / tensor, using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        b_blk = tl.make_block_ptr(
            base=B_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        b = tl.load(b_blk, boundary_check=(0,))
        b_dtype = b.dtype
        b = b.to(tl.float32)
        out = scalar / b
        tl.store(out_blk, out.to(b_dtype), boundary_check=(0,))


def true_divide(A, B):
    logger.debug("GEMS_SPACEMIT TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A, B = A.contiguous(), B.contiguous()
        out_dtype = A.dtype if A.is_floating_point() else torch.float32
        out = torch.empty_like(A, dtype=out_dtype)
        n = A.numel()
        div_kernel_tt[(NUM_CTAS,)](A, B, out, n)
        return out
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        out_dtype = A.dtype if A.is_floating_point() else torch.float32
        out = torch.empty_like(A, dtype=out_dtype)
        n = A.numel()
        div_kernel_ts[(NUM_CTAS,)](A, out, float(B), n)
        return out
    elif isinstance(B, torch.Tensor):
        B = B.contiguous()
        out_dtype = B.dtype if B.is_floating_point() else torch.float32
        out = torch.empty_like(B, dtype=out_dtype)
        n = B.numel()
        div_kernel_st[(NUM_CTAS,)](B, out, float(A), n)
        return out
    else:
        return torch.tensor(A / B)


def true_divide_(A, B):
    logger.debug("GEMS_SPACEMIT TRUE_DIVIDE_")
    A_contig = A.contiguous()
    n = A_contig.numel()
    if isinstance(B, torch.Tensor):
        B = B.contiguous()
        div_kernel_tt[(NUM_CTAS,)](A_contig, B, A_contig, n)
    else:
        div_kernel_ts[(NUM_CTAS,)](A_contig, A_contig, float(B), n)
    if not A.is_contiguous():
        A.copy_(A_contig)
    return A


def floor_divide(A, B):
    logger.debug("GEMS_SPACEMIT FLOOR_DIVIDE")
    result = true_divide(A, B)
    if isinstance(A, torch.Tensor):
        out_dtype = A.dtype
    elif isinstance(B, torch.Tensor):
        out_dtype = B.dtype
    else:
        out_dtype = torch.float32 if isinstance(A, float) or isinstance(B, float) else torch.int64
    return result.floor_().to(out_dtype)


def trunc_divide(A, B):
    logger.debug("GEMS_SPACEMIT TRUNC_DIVIDE")
    result = true_divide(A, B)
    if isinstance(A, torch.Tensor):
        out_dtype = A.dtype
    elif isinstance(B, torch.Tensor):
        out_dtype = B.dtype
    else:
        out_dtype = torch.float32 if isinstance(A, float) or isinstance(B, float) else torch.int64
    return result.trunc_().to(out_dtype)


def div_mode(A, B, rounding_mode=None):
    logger.debug("GEMS_SPACEMIT DIV_MODE")
    if rounding_mode is None:
        return true_divide(A, B)
    if rounding_mode == "trunc":
        return trunc_divide(A, B)
    if rounding_mode == "floor":
        return floor_divide(A, B)
    msg = (
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        f"but found {rounding_mode}."
    )
    raise ValueError(msg)
