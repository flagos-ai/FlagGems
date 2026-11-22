import torch
import triton
from .mm_conv_glu_fused_kernel import mm_conv_glu_fused_kernel

def mm_conv_glu_fused(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused MM + Broadcast Bias + GLU
    Equivalent:
        y = x @ w + bias
        out = y[..., :D//2] * torch.sigmoid(y[..., D//2:])
    Args:
        x: [M, K]
        w: [K, D], D must be even
        bias: [D]
    Returns:
        out: [M, D//2]
    """
    M, K = x.shape
    Kk, D = w.shape
    assert K == Kk, "K dim mismatch between x and w"
    assert D % 2 == 0, "GLU requires last dimension D even"
    assert bias.shape == (D,), "bias shape must be [D]"
    N = D // 2

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"])
    )
    mm_conv_glu_fused_kernel[grid](
        x, w, bias, out,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=64,
    )
    return out
