import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

NUM_VECTOR_CORES = 48


@libentry()
@triton.jit
def logical_and_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tl.program_id(0)
    for task_id in range(pid, num_tasks, NCORE):
        base_offset = task_id * BLOCK_SIZE
        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < N
            x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
            y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
            result = x.to(tl.int1) & y.to(tl.int1)
            tl.store(out_ptr + offsets, result, mask=mask)


def _compute_grid(N):
    BLOCK_SIZE = 8192
    BLOCK_SIZE_SUB = 1024
    num_tasks = triton.cdiv(N, BLOCK_SIZE)
    ncore = min(num_tasks, NUM_VECTOR_CORES)
    return ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB


# Fallback for broadcasting cases
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_and_func(x, y):
    return x.to(tl.int1) & y.to(tl.int1)


def logical_and(A, B):
    logger.debug("GEMS_ASCEND LOGICAL_AND")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and A.shape == B.shape:
        A = A.contiguous()
        B = B.contiguous()
        out = torch.empty_like(A, dtype=torch.bool)
        N = A.numel()
        ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB = _compute_grid(N)
        logical_and_kernel[(ncore,)](A, B, out, N, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB, ncore)
        return out
    return logical_and_func(A, B)


def logical_and_(A, B):
    logger.debug("GEMS_ASCEND LOGICAL_AND_")
    if isinstance(B, torch.Tensor) and A.shape == B.shape:
        A_contig = A.contiguous()
        B = B.contiguous()
        N = A_contig.numel()
        ncore, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB = _compute_grid(N)
        logical_and_kernel[(ncore,)](A_contig, B, A_contig, N, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB, ncore)
        if not A.is_contiguous():
            A.copy_(A_contig)
        return A
    logical_and_func(A, B, out0=A)
    return A
