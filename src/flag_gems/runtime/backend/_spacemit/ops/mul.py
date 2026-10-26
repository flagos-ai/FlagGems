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
    configs=runtime.get_tuned_config("mul_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def mul_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """tensor * tensor, using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        b_blk = tl.make_block_ptr(
            base=B_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        b = tl.load(b_blk, boundary_check=(0,))
        tl.store(out_blk, a * b, boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mul_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def mul_kernel_ts(
    A_ptr,
    Out_ptr,
    scalar_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """tensor * scalar, using make_block_ptr + boundary_check."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        a_blk = tl.make_block_ptr(
            base=A_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )
        out_blk = tl.make_block_ptr(
            base=Out_ptr, shape=(n_elements,), strides=(1,),
            offsets=(block_start,), block_shape=(BLOCK_SIZE,), order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        out = a * scalar_val
        tl.store(out_blk, out.to(a.dtype), boundary_check=(0,))


def mul(A, B):
    logger.debug("GEMS_SPACEMIT MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        A, B = torch.broadcast_tensors(A, B)
        A = A.contiguous()
        B = B.contiguous()
        out = torch.empty_like(A)
        n = A.numel()
        mul_kernel_tt[(NUM_CTAS,)](A, B, out, n)
        return out
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        out = torch.empty_like(A)
        n = A.numel()
        mul_kernel_ts[(NUM_CTAS,)](A, out, float(B), n)
        return out
    elif isinstance(B, torch.Tensor):
        B = B.contiguous()
        out = torch.empty_like(B)
        n = B.numel()
        mul_kernel_ts[(NUM_CTAS,)](B, out, float(A), n)
        return out
    else:
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_SPACEMIT MUL_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        B = B.broadcast_to(A.shape)
        A_contig = A.contiguous()
        B = B.contiguous()
        n = A_contig.numel()
        mul_kernel_tt[(NUM_CTAS,)](A_contig, B, A_contig, n)
        if not A.is_contiguous():
            A.copy_(A_contig)
        return A
    else:
        A_contig = A.contiguous()
        n = A_contig.numel()
        mul_kernel_ts[(NUM_CTAS,)](A_contig, A_contig, float(B), n)
        if not A.is_contiguous():
            A.copy_(A_contig)
        return A
