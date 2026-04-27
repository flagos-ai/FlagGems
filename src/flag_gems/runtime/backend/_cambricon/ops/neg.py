import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 65536}, num_stages=1, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 131072}, num_stages=1, num_warps=1),
    ],
    key=["n_elements"],
)
@triton.jit
def neg_func(
    X_ptr,
    OUT_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for off in range(block_start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(X_ptr + offsets, mask=mask)
        tl.store(OUT_ptr + offsets, -x, mask=mask)


def neg(A):
    logger.debug("GEMS_CAMBRICON NEG")
    A = A.contiguous()
    out = torch.empty_like(A)
    N = A.numel()
    if N == 0:
        return out
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(A.device):
        neg_func[grid_fn](A, out, N)
    return out


def neg_(A):
    logger.debug("GEMS_CAMBRICON NEG_")
    A_contig = A.contiguous()
    N = A_contig.numel()
    if N == 0:
        return A
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(A.device):
        neg_func[grid_fn](A_contig, A_contig, N)
    if not A.is_contiguous():
        A.copy_(A_contig)
    return A
