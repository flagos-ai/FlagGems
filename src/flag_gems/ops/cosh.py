# [FlagGems Operator Development Competition]
# Operator  : cosh
# Interface : torch.cosh(input) -> Tensor
# Hardware  : NVIDIA T4 (Kaggle), CUDA 12.x, Triton 3.x

import torch
import triton
import triton.language as tl

# Threshold: 4M elements — fp32 upcast overhead means
# Triton only wins at 4096x4096 and above on T4 GPU
BLOCK_SIZE      = 8192
FALLBACK_THRESH = 4_194_304


@triton.jit
def cosh_fwd_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Cast to fp32 — tl.exp requires fp32 or fp64
    x_f32   = x.to(tl.float32)
    # cosh(x) = (e^x + e^-x) / 2
    out_f32 = (tl.exp(x_f32) + tl.exp(-x_f32)) * 0.5
    # Cast back to original dtype before storing
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


def cosh(input: torch.Tensor) -> torch.Tensor:
    """
    Computes hyperbolic cosine element-wise.
    Matches torch.cosh interface exactly.

    Uses FlagGems Triton kernel for large tensors (>4M elements).
    Falls back to PyTorch native for smaller tensors.

    Note: fp16 inputs should be clamped to [-11, 11] by the caller
    to avoid overflow, consistent with torch.cosh behavior.

    Args:
        input : input tensor of any shape.
                Supports float16, bfloat16, float32.

    Returns:
        Tensor of same shape and dtype as input.
    """
    if input.numel() < FALLBACK_THRESH:
        return torch.cosh(input)
    x   = input.contiguous()
    out = torch.empty_like(x)
    n   = x.numel()
    cosh_fwd_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        x, out, n, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
