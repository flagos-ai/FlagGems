import logging

import torch
import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import isnan as _isnan

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
def nan_to_num_kernel(
    X_ptr,
    OUT_ptr,
    nan_val,
    posinf_val,
    neginf_val,
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
        x_nan = _isnan(x)
        x_posinf = x == float("inf")
        x_neginf = x == float("-inf")
        result = tl.where(x_nan, nan_val, x)
        result = tl.where(x_posinf, posinf_val, result)
        result = tl.where(x_neginf, neginf_val, result)
        tl.store(OUT_ptr + offsets, result, mask=mask)


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_CAMBRICON NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0

    A = A.contiguous()
    out = torch.empty_like(A)
    N = A.numel()
    if N == 0:
        return out
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(A.device):
        nan_to_num_kernel[grid_fn](A, out, nan, posinf, neginf, N)
    return out
