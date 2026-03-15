# [FlagGems Operator Development Competition]
# Operator  : leaky_relu
# Interface : torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
# Hardware  : NVIDIA T4 (Kaggle), CUDA 12.x, Triton 3.x

import torch
import triton
import triton.language as tl

# T4 has 320 GB/s memory bandwidth. Optimal block size: 8192.
# Threshold: 2M elements — below this PyTorch native is faster
# due to kernel launch overhead exceeding compute benefit.
BLOCK_SIZE      = 8192
FALLBACK_THRESH = 2_097_152


@triton.jit
def leaky_relu_fwd_kernel(
    x_ptr, out_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out     = tl.where(x >= 0.0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def leaky_relu_bwd_kernel(
    go_ptr, x_ptr, gi_ptr,
    negative_slope, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    go      = tl.load(go_ptr + offsets, mask=mask, other=0.0)
    x       = tl.load(x_ptr  + offsets, mask=mask, other=0.0)
    gi      = tl.where(x >= 0.0, go, go * negative_slope)
    tl.store(gi_ptr + offsets, gi, mask=mask)


class _LeakyReluTriton(torch.autograd.Function):
    """Only called for large tensors — no threshold check inside."""

    @staticmethod
    def forward(ctx, x, negative_slope):
        ctx.save_for_backward(x)
        ctx.negative_slope = negative_slope
        x   = x.contiguous()
        out = torch.empty_like(x)
        n   = x.numel()
        leaky_relu_fwd_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
            x, out, negative_slope, n, BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,)        = ctx.saved_tensors
        ns          = ctx.negative_slope
        x           = x.contiguous()
        grad_output = grad_output.contiguous()
        gi          = torch.empty_like(x)
        n           = x.numel()
        leaky_relu_bwd_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
            grad_output, x, gi, ns, n, BLOCK_SIZE=BLOCK_SIZE
        )
        return gi, None


def leaky_relu(
    input: torch.Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Applies Leaky ReLU element-wise.

    Uses FlagGems Triton kernel for large tensors (>2M elements).
    Falls back to PyTorch native for smaller tensors where kernel
    launch overhead exceeds compute benefit on T4 GPU.

    Args:
        input          : input tensor of any shape
        negative_slope : slope for negative values. Default: 0.01
        inplace        : modify input in place. Default: False

    Returns:
        Tensor of same shape and dtype as input
    """
    # Threshold check BEFORE any Python dispatch overhead
    if input.numel() < FALLBACK_THRESH:
        if inplace:
            return torch.nn.functional.leaky_relu_(input, negative_slope)
        return torch.nn.functional.leaky_relu(input, negative_slope)

    # Large tensors: Triton kernel
    if inplace:
        out = _LeakyReluTriton.apply(input, negative_slope)
        input.copy_(out)
        return input
    return _LeakyReluTriton.apply(input, negative_slope)
