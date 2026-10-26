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
    configs=runtime.get_tuned_config("rsqrt"),
    key=["n_elements"],
)
@triton.jit
def rsqrt_kernel(
    X_ptr,
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
        out = tl.rsqrt(x.to(tl.float32))
        tl.store(out_blk, out.to(x.dtype), boundary_check=(0,))


def rsqrt(A):
    logging.debug("GEMS_SPACEMIT RSQRT")
    A = A.contiguous()
    out = torch.empty_like(A)
    n_elements = A.numel()
    with torch_device_fn.device(A.device):
        rsqrt_kernel[(NUM_CTAS,)](A, out, n_elements)
    return out


def rsqrt_(A):
    logging.debug("GEMS_SPACEMIT RSQRT_")
    A_contig = A.contiguous()
    n_elements = A_contig.numel()
    with torch_device_fn.device(A.device):
        rsqrt_kernel[(NUM_CTAS,)](A_contig, A_contig, n_elements)
    if not A.is_contiguous():
        A.copy_(A_contig)
    return A
