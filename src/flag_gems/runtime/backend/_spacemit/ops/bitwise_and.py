import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bitwise_and_tensor_tensor"),
    key=["n_elements"],
)
@triton.jit
def bitwise_and_kernel_tt(
    A_ptr,
    B_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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
        out = a & b
        tl.store(out_blk, out, boundary_check=(0,))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bitwise_and_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def bitwise_and_kernel_ts(
    A_ptr,
    Out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
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
        out = a & scalar
        tl.store(out_blk, out.to(a.dtype), boundary_check=(0,))


def bitwise_and_tensor(A, B):
    logger.debug("GEMS_SPACEMIT BITWISE_AND")
    A, B = A.contiguous(), B.contiguous()
    out = torch.empty_like(A)
    n = A.numel()
    with torch_device_fn.device(A.device):
        bitwise_and_kernel_tt[(NUM_CTAS,)](A, B, out, n)
    return out


def bitwise_and_tensor_(A, B):
    logger.debug("GEMS_SPACEMIT BITWISE_AND_")
    A_c = A.contiguous()
    B = B.contiguous()
    n = A_c.numel()
    with torch_device_fn.device(A.device):
        bitwise_and_kernel_tt[(NUM_CTAS,)](A_c, B, A_c, n)
    if not A.is_contiguous():
        A.copy_(A_c)
    return A


def bitwise_and_scalar(A, B):
    logger.debug("GEMS_SPACEMIT BITWISE_AND_SCALAR")
    A = A.contiguous()
    out = torch.empty_like(A)
    n = A.numel()
    with torch_device_fn.device(A.device):
        bitwise_and_kernel_ts[(NUM_CTAS,)](A, out, int(B), n)
    return out


def bitwise_and_scalar_(A, B):
    logger.debug("GEMS_SPACEMIT BITWISE_AND__SCALAR")
    A_c = A.contiguous()
    n = A_c.numel()
    with torch_device_fn.device(A.device):
        bitwise_and_kernel_ts[(NUM_CTAS,)](A_c, A_c, int(B), n)
    if not A.is_contiguous():
        A.copy_(A_c)
    return A


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS_SPACEMIT BITWISE_AND_SCALAR_TENSOR")
    return bitwise_and_scalar(B, A)
