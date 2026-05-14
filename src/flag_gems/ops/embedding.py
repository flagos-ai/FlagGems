import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedding_kernel(out_ptr, weight_ptr, indices_ptr, n, emb_dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = tl.load(indices_ptr + pid)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < emb_dim
    emb = tl.load(weight_ptr + idx * emb_dim + cols, mask=mask, other=0.0)
    tl.store(out_ptr + pid * emb_dim + cols, emb, mask=mask)


def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS EMBEDDING")
    if weight.numel() == 0 or indices.numel() == 0:
        return weight.new_empty((*indices.shape, weight.shape[-1]))
    orig_shape = indices.shape
    indices_flat = indices.reshape(-1)
    n = indices_flat.shape[0]
    emb_dim = weight.shape[1]
    out = torch.empty(n, emb_dim, device=weight.device, dtype=weight.dtype)
    BS = triton.next_power_of_2(min(emb_dim, 4096))
    embedding_kernel[(n,)](out, weight, indices_flat, n, emb_dim, BLOCK_SIZE=BS)
    return out.reshape(*orig_shape, emb_dim)


def embedding_backward(grad, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    logger.debug("GEMS EMBEDDING BACKWARD")
    indices_flat = indices.reshape(-1)
    grad_flat = grad.reshape(-1, grad.shape[-1])
    grad_weight = torch.zeros(num_weights, grad.shape[-1], device=grad.device, dtype=grad.dtype)
    grad_weight.scatter_add_(0, indices_flat.unsqueeze(1).expand_as(grad_flat), grad_flat)
    return grad_weight
