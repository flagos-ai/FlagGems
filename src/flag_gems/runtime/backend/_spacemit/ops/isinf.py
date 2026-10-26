import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_isinf = tl_extra_shim.isinf

NUM_CTAS = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("eq_tensor_scalar"),
    key=["n_elements"],
)
@triton.jit
def isinf_kernel(
    A_ptr,
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
        out_blk = tl.make_block_ptr(
            base=Out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        a = tl.load(a_blk, boundary_check=(0,))
        out = tl.where(_isinf(a), 1, 0).to(Out_ptr.type.element_ty)
        tl.store(out_blk, out, boundary_check=(0,))


def isinf(A):
    logger.debug("GEMS_SPACEMIT ISINF")
    A = A.contiguous()
    out = torch.empty(A.shape, dtype=torch.uint8, device=A.device)
    n = A.numel()
    isinf_kernel[(NUM_CTAS,)](A, out, n)
    return out.view(torch.bool)