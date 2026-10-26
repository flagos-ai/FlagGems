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
    configs=runtime.get_tuned_config("sub_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def sub_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """tensor - tensor * alpha, using make_block_ptr + boundary_check."""
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
        b = tl.load(b_blk, boundary_check=(0,))
        out = a - b * alpha
        tl.store(out_blk, out.to(a.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sub_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def sub_kernel_ts(
    A_ptr,
    Out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """tensor - scalar * alpha."""
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
        out = a - scalar * alpha
        tl.store(out_blk, out.to(a.dtype), boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sub_scalar_tensor"),
    key=["n_elements"],
)
@triton.jit
def sub_kernel_st(
    B_ptr,
    Out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """scalar - tensor * alpha."""
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
        out = scalar - b * alpha
        tl.store(out_blk, out.to(b.dtype), boundary_check=(0,))


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_SPACEMIT SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A = A.contiguous()
        B = B.contiguous()
        out = torch.empty_like(A)
        n_elements = A.numel()
        sub_kernel_tt[(NUM_CTAS,)](A, B, out, alpha, n_elements)
        return out
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        out = torch.empty_like(A)
        n_elements = A.numel()
        sub_kernel_ts[(NUM_CTAS,)](A, out, B, alpha, n_elements)
        return out
    elif isinstance(B, torch.Tensor):
        B = B.contiguous()
        out = torch.empty_like(B)
        n_elements = B.numel()
        sub_kernel_st[(NUM_CTAS,)](B, out, A, alpha, n_elements)
        return out
    else:
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_SPACEMIT SUB_")
    if isinstance(B, torch.Tensor):
        A_contig = A.contiguous()
        B = B.contiguous()
        n_elements = A_contig.numel()
        sub_kernel_tt[(NUM_CTAS,)](A_contig, B, A_contig, alpha, n_elements)
        if not A.is_contiguous():
            A.copy_(A_contig)
    else:
        A_contig = A.contiguous()
        n_elements = A_contig.numel()
        sub_kernel_ts[(NUM_CTAS,)](A_contig, A_contig, B, alpha, n_elements)
        if not A.is_contiguous():
            A.copy_(A_contig)
    return A
