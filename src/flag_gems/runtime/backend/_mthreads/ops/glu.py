import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.glu import glu as default_glu
from flag_gems.ops.glu import glu_backward as default_glu_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
exp = tl_extra_shim.exp


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def glu_kernel_last_dim(
    x_ptr,
    out_ptr,
    n_elements,
    half_last_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """GLU kernel optimized for dim=-1 (last dimension split).

    Input shape: [..., 2*D] -> Output shape: [..., D]
    We read a from x[..., :D] and b from x[..., D:]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate position in output tensor
    # For each output position, we need to read from two positions in input
    # Output idx -> (batch_idx, inner_idx) where inner_idx < half_last_dim
    batch_idx = offsets // half_last_dim
    inner_idx = offsets % half_last_dim

    # Input positions for a and b
    # a is at x[batch_idx * 2 * half_last_dim + inner_idx]
    # b is at x[batch_idx * 2 * half_last_dim + half_last_dim + inner_idx]
    full_last_dim = half_last_dim * 2
    a_offsets = batch_idx * full_last_dim + inner_idx
    b_offsets = a_offsets + half_last_dim

    a = tl.load(x_ptr + a_offsets, mask=mask)
    b = tl.load(x_ptr + b_offsets, mask=mask)

    # Compute sigmoid(b) * a
    b_fp32 = b.to(tl.float32)
    sigmoid_b = 1.0 / (1.0 + exp(-b_fp32))
    result = a.to(tl.float32) * sigmoid_b

    tl.store(out_ptr + offsets, result.to(a.dtype), mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def glu_kernel_general(
    x_ptr,
    out_ptr,
    n_elements,
    stride_before,  # stride for dimensions before split dim
    stride_split,  # stride of split dimension
    half_split_size,  # half of split dimension size
    BLOCK_SIZE: tl.constexpr,
):
    """GLU kernel for general dimension split.

    Handles any dimension for the split operation.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose output offset into indices
    # Output shape is the same as input except split dim is halved
    # We need to map output offset to input offset

    # For output offset, compute:
    # - idx_before: index in dimensions before split (combined)
    # - idx_split: index within split dimension (0 to half_split_size-1)
    # - idx_after: index in dimensions after split (combined, this is just offsets % stride_split)

    idx_after = offsets % stride_split
    remaining = offsets // stride_split
    idx_split = remaining % half_split_size
    idx_before = remaining // half_split_size

    # Input offsets for a and b
    # a: same position, b: offset by half_split_size in split dimension
    base_offset = idx_before * stride_before + idx_split * stride_split + idx_after
    a_offsets = base_offset
    b_offsets = base_offset + half_split_size * stride_split

    a = tl.load(x_ptr + a_offsets, mask=mask)
    b = tl.load(x_ptr + b_offsets, mask=mask)

    # Compute sigmoid(b) * a
    b_fp32 = b.to(tl.float32)
    sigmoid_b = 1.0 / (1.0 + exp(-b_fp32))
    result = a.to(tl.float32) * sigmoid_b

    tl.store(out_ptr + offsets, result.to(a.dtype), mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if not x.is_contiguous() or x.numel() == 0:
        return False
    return True


def glu(x, dim=-1):
    logger.debug("GEMS_MTHREADS GLU FORWARD")

    # Normalize dim
    ndim = x.dim()
    if dim < 0:
        dim = ndim + dim

    assert x.shape[dim] % 2 == 0, "Split dimension must be even"

    if not _use_triton_kernel(x):
        return default_glu(x, dim=dim)

    # Compute output shape
    out_shape = list(x.shape)
    half_dim_size = x.shape[dim] // 2
    out_shape[dim] = half_dim_size
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    n_elements = out.numel()

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        if dim == ndim - 1:
            # Optimized path for last dimension split
            glu_kernel_last_dim[grid](x, out, n_elements, half_dim_size)
        else:
            # General path for other dimensions
            # Compute strides for the decomposition
            # stride_before: product of all dims before split dim (times 2 for full split dim)
            # stride_split: product of all dims after split dim
            stride_split = 1
            for i in range(dim + 1, ndim):
                stride_split *= x.shape[i]

            stride_before = stride_split * x.shape[dim]  # Full split dim size

            glu_kernel_general[grid](
                x, out, n_elements, stride_before, stride_split, half_dim_size
            )

    return out


def glu_backward(grad_output, x, dim=-1):
    logger.debug("GEMS_MTHREADS GLU BACKWARD")
    # For backward, we fallback to default implementation for now
    # since the main performance issue is in forward pass
    return default_glu_backward(grad_output, x, dim=dim)
