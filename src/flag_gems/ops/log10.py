# [FlagGems Operator Development Competition]
# Operator  : log10
# Interface : torch.log10(input) -> Tensor
# Hardware  : NVIDIA T4 (Kaggle), CUDA 12.x, Triton 3.x

import torch
import triton
import triton.language as tl

# Threshold: 4M elements — fp32 upcast overhead means
# Triton only wins at 4096x4096 and above on T4 GPU
BLOCK_SIZE      = 8192
FALLBACK_THRESH = 4_194_304


@triton.jit
def log10_fwd_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    # other=1.0 so masked elements compute log(1)=0, never log(0)=-inf
    x       = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    # Cast to fp32 — tl.log requires fp32 or fp64
    x_f32   = x.to(tl.float32)
    # log10(x) = ln(x) / ln(10) = ln(x) * (1/ln(10))
    # Constant inlined as literal — Triton JIT cannot read Python globals
    LOG10E  = 0.4342944819032518
    out_f32 = tl.log(x_f32) * LOG10E
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


def log10(input: torch.Tensor) -> torch.Tensor:
    """
    Computes base-10 logarithm element-wise.
    Matches torch.log10 interface exactly.

    Uses FlagGems Triton kernel for large tensors (>4M elements).
    Falls back to PyTorch native for smaller tensors.

    Args:
        input : input tensor of any shape. Must contain positive values.
                Supports float16, bfloat16, float32.

    Returns:
        Tensor of same shape and dtype as input.
        Returns -inf for zero inputs, nan for negative inputs —
        consistent with torch.log10 behavior.
    """
    if input.numel() < FALLBACK_THRESH:
        return torch.log10(input)
    x   = input.contiguous()
    out = torch.empty_like(x)
    n   = x.numel()
    log10_fwd_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        x, out, n, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
