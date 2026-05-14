import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,
    weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one embedding vector lookup
    pid = tl.program_id(axis=0)
    idx = tl.load(indices_ptr + pid)

    # Vectorized load of embedding vector
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < embedding_dim

    # Load embedding weights for this index
    weight_offsets = idx * embedding_dim + offsets
    emb = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)

    # Store to output
    out_offsets = pid * embedding_dim + offsets
    tl.store(out_ptr + out_offsets, emb, mask=mask)


def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS EMBEDDING")
    if weight.numel() == 0 or indices.numel() == 0:
        return weight.new_empty((*indices.shape, weight.shape[-1]))
    orig_shape = indices.shape
    indices_flat = indices.reshape(-1)
    n = indices_flat.shape[0]
    embedding_dim = weight.shape[1]
    out = torch.empty(n, embedding_dim, device=weight.device, dtype=weight.dtype)
    BLOCK_SIZE = triton.next_power_of_2(min(embedding_dim, 4096))
    embedding_kernel[(n,)](
        out, weight, indices_flat, n, embedding_dim, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.reshape(*orig_shape, embedding_dim)


def embedding_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS EMBEDDING BACKWARD")
    # Use PyTorch's optimized scatter_add for backward
    indices_flat = indices.reshape(-1)
    grad_flat = grad.reshape(-1, grad.shape[-1])
    grad_weight = torch.zeros(
        num_weights, grad.shape[-1], device=grad.device, dtype=grad.dtype
    )
    grad_weight.scatter_add_(
        0, indices_flat.unsqueeze(1).expand_as(grad_flat), grad_flat
    )
    return grad_weight
