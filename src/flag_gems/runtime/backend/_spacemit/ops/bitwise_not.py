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
    configs=runtime.get_tuned_config("bitwise_not"),
    key=["n_elements"],
)
@triton.jit
def bitwise_not_kernel(
    X_ptr,
    Out_ptr,
    n_elements,
    IS_BOOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(total_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        x_blk = tl.make_block_ptr(
            base=X_ptr,
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

        x = tl.load(x_blk, boundary_check=(0,))
        if IS_BOOL:
            # make_block_ptr treats pointer_type<i1> as pointer_type<i8>, so the
            # loaded value stays i8 (0/1) instead of being cast back to i1. A raw
            # `~x` would xor the full byte (0xFF), leaving every element non-zero.
            # xor 1 gives the correct logical negation for bool tensors.
            out = x ^ 1
        else:
            out = ~x
        tl.store(out_blk, out, boundary_check=(0,))


def bitwise_not(A):
    logger.debug("GEMS_SPACEMIT BITWISE_NOT")
    A = A.contiguous()
    out = torch.empty_like(A)
    n = A.numel()
    is_bool = A.dtype == torch.bool
    with torch_device_fn.device(A.device):
        bitwise_not_kernel[(NUM_CTAS,)](A, out, n, is_bool)
    return out


def bitwise_not_(A):
    logger.debug("GEMS_SPACEMIT BITWISE_NOT_")
    A_c = A.contiguous()
    n = A_c.numel()
    is_bool = A_c.dtype == torch.bool
    with torch_device_fn.device(A.device):
        bitwise_not_kernel[(NUM_CTAS,)](A_c, A_c, n, is_bool)
    if not A.is_contiguous():
        A.copy_(A_c)
    return A
