import torch
import triton
from .mm_conv_glu_fused_kernel import mm_conv_glu_fused_kernel

def mm_conv_glu_fused(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused MM + Broadcast Bias + GLU
    Equivalent:
        full = x @ w + bias
        out = full[..., :D//2] * torch.sigmoid(full[..., D//2:])
    Host pre-split weight/bias, avoid Iluvatar triton dynamic slice limitation
    Args:
        x: [M, K] privateuseone
        w: [K, D], D must be even
        bias: [D]
    Returns:
        out: [M, D//2]
    """
    assert x.device.type == "privateuseone"
    M, K = x.shape
    Kk, D = w.shape
    assert K == Kk, "K dim mismatch between x and w"
    assert D % 2 == 0, "GLU requires last dimension D even"
    assert bias.shape == (D,), "bias shape must be [D,]"
    N = D // 2

    wa = w[:, :N].contiguous()
    wg = w[:, N:].contiguous()
    ba = bias[:N].contiguous()
    bg = bias[N:].contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    mm_conv_glu_fused_kernel[grid](
        x, wa, wg, ba, bg, out,
        M, K, N,
        x.stride(0), x.stride(1),
        wa.stride(0), wa.stride(1),
        wg.stride(0), wg.stride(1),
        ba.stride(0),
        bg.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return out
