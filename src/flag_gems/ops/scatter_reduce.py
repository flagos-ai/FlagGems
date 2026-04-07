import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# scatter_reduce: Reduces values from src into self at indices specified
# by index tensor. Uses Triton atomic operations for thread-safe
# accumulation. Accumulates in float32 for numerical stability.
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
    self_contig = self.contiguous()
    src_contig = src.contiguous()
    index_contig = index.contiguous()
    orig_dtype = self.dtype

    # Use float32 accumulation buffer
    if include_self:
        out = self_contig.to(torch.float32)
    else:
        out = torch.zeros(
            self_contig.shape, dtype=torch.float32, device=self_contig.device
        )

    shape = src_contig.shape
    inner_size = 1
    for d in range(dim + 1, len(shape)):
        inner_size *= shape[d]
    src_dim_size = shape[dim]
    out_dim_size = self_contig.shape[dim]

    src_numel = src_contig.numel()
    if src_numel == 0:
        return out.to(orig_dtype)

    if reduce == "sum" or reduce == "mean":
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(src_numel, BLOCK_SIZE),)

        with torch_device_fn.device(self.device):
            scatter_reduce_sum_kernel[grid](
                src_contig,
                index_contig,
                out,
                src_numel,
                inner_size,
                src_dim_size,
                out_dim_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        if reduce == "mean":
            # Count elements per output position
            count = torch.zeros_like(out)
            if include_self:
                count.fill_(1.0)
            ones = torch.ones(src_contig.shape, dtype=torch.float32, device=self.device)
            with torch_device_fn.device(self.device):
                scatter_reduce_sum_kernel[grid](
                    ones,
                    index_contig,
                    count,
                    src_numel,
                    inner_size,
                    src_dim_size,
                    out_dim_size,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            out = out / count.clamp(min=1)
    else:
        # For prod, amax, amin: use PyTorch's scatter_reduce_ on float32 buffer
        out_native = self_contig.clone()
        out_native.scatter_reduce_(
            dim, index_contig, src_contig, reduce, include_self=include_self
        )
        return out_native

    return out.to(orig_dtype)
