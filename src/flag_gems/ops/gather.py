import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def gather_kernel(
    out_ptr,
    x_ptr,
    index_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of gather operations
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load indices and gather values
    idx = tl.load(index_ptr + offsets, mask=mask, other=0)
    val = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def gather(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS GATHER")
    if x.numel() == 0 or index.numel() == 0:
        return torch.gather(x, dim, index)
    # Use PyTorch for non-contiguous dims, Triton for last dim
    if dim == x.dim() - 1 or dim == -1:
        x_c = x.contiguous()
        idx_c = index.contiguous()
        out = torch.empty_like(idx_c, dtype=x.dtype)
        n_elements = idx_c.numel()
        BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        # For last dim gather, flatten and use kernel
        gather_kernel[grid](
            out.reshape(-1),
            x_c.reshape(-1),
            idx_c.reshape(-1),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    return torch.gather(x, dim, index)


def gather_backward(
    dy: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    x_size: torch.Size,
) -> torch.Tensor:
    logger.debug("GEMS GATHER BACKWARD")
    dx = torch.zeros(x_size, device=dy.device, dtype=dy.dtype)
    dx.scatter_add_(dim, index, dy)
    return dx
