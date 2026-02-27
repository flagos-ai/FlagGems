import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_backward_kernel(
    grad_inp,
    grad_out,
    index,
    M,  # outer_size
    N,  # src_dim_size (required by heuristics)
    inner_size,
    index_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward kernel for index_select.
    Scatters gradients from grad_out back to grad_inp at positions specified by index.

    Memory layout (for contiguous tensors):
    - grad_out has shape [outer, index_len, inner] when viewed as 3D
    - grad_inp has shape [outer, N, inner] when viewed as 3D

    Args:
        grad_inp: Output gradient tensor - target to scatter to
        grad_out: Input gradient tensor - source of gradients
        index: Indices tensor (1D)
        M: Product of dimensions before dim (outer_size)
        N: Original size at dim (src_dim_size)
        inner_size: Product of dimensions after dim
        index_len: Number of indices (size of grad_out at dim)
    """
    # Each program handles a block of (outer_idx, index_idx) pairs
    pid_outer = tle.program_id(axis=0)
    pid_idx = tle.program_id(axis=1)

    outer_offsets = pid_outer * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    idx_offsets = pid_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    outer_mask = outer_offsets < M
    idx_mask = idx_offsets < index_len
    combined_mask = outer_mask and idx_mask

    # Load indices
    indices = tl.load(index + idx_offsets, mask=idx_mask, other=0)
    valid_lower = indices >= 0
    valid_upper = indices < N
    index_valid_mask = valid_lower & valid_upper
    final_mask = combined_mask & index_valid_mask

    # For each inner position, scatter gradients
    for inner_idx in range(inner_size):
        # grad_out offset: outer * (index_len * inner_size) + idx * inner_size + inner
        grad_out_base = outer_offsets * (index_len * inner_size) + idx_offsets[None, :] * inner_size + inner_idx

        # grad_inp offset: outer * (N * inner_size) + indices * inner_size + inner
        grad_inp_base = outer_offsets * (N * inner_size) + indices[None, :] * inner_size + inner_idx

        grad_values = tl.load(grad_out + grad_out_base, mask=final_mask, other=0.0)
        tl.atomic_add(grad_inp + grad_inp_base, grad_values, mask=final_mask, sem="relaxed")


def index_select_backward(grad, self_sizes, dim, index):
    """
    Backward pass for index_select.

    Args:
        grad: Gradient tensor from the forward output
        self_sizes: List of sizes of the original input tensor
        dim: Dimension along which index_select was performed
        index: Indices used in the forward pass

    Returns:
        Gradient tensor with shape self_sizes
    """
    logger.debug("GEMS INDEX SELECT BACKWARD")

    ndim = len(self_sizes)
    assert dim >= -ndim and dim < ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % ndim
    index_len = index.numel()

    # Handle fp16/bf16 by computing in fp32
    compute_dtype = grad.dtype
    if grad.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    # Create output tensor initialized to zeros
    grad_inp = torch.zeros(
        self_sizes, dtype=compute_dtype, device=grad.device
    )

    # Make grad contiguous for the kernel
    grad_contig = grad.contiguous()

    # Compute M (outer_size) = product of dims before dim
    M = 1
    for i in range(dim):
        M *= self_sizes[i]

    # Compute inner_size = product of dims after dim
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= self_sizes[i]

    # N = original size at dim
    N = self_sizes[dim]

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )

    index_select_backward_kernel[grid](
        grad_inp, grad_contig, index,
        M, N, inner_size, index_len
    )

    # Convert back to original dtype if needed
    if grad.dtype in (torch.float16, torch.bfloat16):
        grad_inp = grad_inp.to(grad.dtype)

    return grad_inp
