import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# scatter_reduce: Triton kernel for scatter-reduce sum operation.
# Accumulates src values into output at scattered indices using atomic_add.
# Uses float32 accumulation for numerical stability.
@libentry()
@triton.jit
def scatter_reduce_sum_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    src_numel,
    inner_size,
    src_dim_size,
    out_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < src_numel

    inner_idx = offsets % inner_size
    tmp = offsets // inner_size
    outer_idx = tmp // src_dim_size

    src_val = tl.load(src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    idx = tl.load(index_ptr + offsets, mask=mask, other=0)

    out_offset = outer_idx * out_dim_size * inner_size + idx * inner_size + inner_idx
    tl.atomic_add(out_ptr + out_offset, src_val, mask=mask)


def scatter_reduce_two(self, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE")

    dim = dim % self.ndim
    orig_dtype = self.dtype

    if reduce == "sum":
        out = (
            self.to(torch.float32)
            if include_self
            else torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        )

        shape = src.shape
        inner_size = 1
        for d in range(dim + 1, len(shape)):
            inner_size *= shape[d]

        src_numel = src.numel()
        if src_numel > 0:
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(src_numel, BLOCK_SIZE),)
            with torch_device_fn.device(self.device):
                scatter_reduce_sum_kernel[grid](
                    src.contiguous(),
                    index.contiguous(),
                    out,
                    src_numel,
                    inner_size,
                    shape[dim],
                    self.shape[dim],
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        return out.to(orig_dtype)
    else:
        # For prod, mean, amax, amin: delegate to native implementation
        out = self.clone()
        out.scatter_reduce_(dim, index, src, reduce, include_self=include_self)
        return out
