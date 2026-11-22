import torch
import triton
from .mm_conv_glu_fused_kernel import mm_conv_glu_fused_kernel

def mm_conv_glu_fused(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused MM + BiasAdd + GLU
    Equivalent:
    y = x @ w + bias
    out = y[..., :D//2] * sigmoid(y[..., D//2:])
    Args:
        x: [M, K]
        w: [K, N], N must be even
        bias: [N]
    Returns:
        out: [M, N//2]
    """
    M, K = x.shape
    Kk, N = w.shape
    assert K == Kk
    assert N % 2 == 0, "GLU requires last dim N even"
    assert bias.shape == (N,)

    out_shape = (M, N // 2)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

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
    )
    return out
