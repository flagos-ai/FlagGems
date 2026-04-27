import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from ..utils import TOTAL_CORE_NUM
from ..utils.pointwise_dynamic import pointwise_dynamic

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
@triton.jit(do_not_specialize=["threshold_val", "value_val"])
def threshold_kernel(
    X_ptr,
    OUT_ptr,
    n_elements,
    threshold_val,
    value_val,
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
        result = tl.where(x > threshold_val, x, value_val)
        tl.store(OUT_ptr + offsets, result, mask=mask)


# keep backward using pointwise_dynamic
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def threshold_backward_kernel(grad_output, self, threshold):
    return tl.where(self > threshold, grad_output, 0)


def threshold(self, threshold_val, value_val):
    logger.debug("GEMS_CAMBRICON THRESHOLD FORWARD")
    A = self.contiguous()
    out = torch.empty_like(A)
    N = A.numel()
    if N == 0:
        return out
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(A.device):
        threshold_kernel[grid_fn](A, out, N, threshold_val, value_val)
    return out


def threshold_backward(grad_output, self, threshold_val):
    logger.debug("GEMS_CAMBRICON THRESHOLD BACKWARD")
    return threshold_backward_kernel(grad_output, self, threshold_val)
