import torch
import triton
import triton.language as tl

@triton.jit
def glu_kernel(
    x_ptr, out_ptr,
    M, K,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    # x shape [M, 2K], stride_xm, stride_xk
    x_base = x_ptr + offs_m[:, None] * (2*K)
    a = tl.load(
        x_base + offs_k[None, :],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0
    )
    b = tl.load(
        x_base + K + offs_k[None, :],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0
    )
    out = a * tl.sigmoid(b)
    tl.store(
        out_ptr + offs_m[:, None] * K + offs_k[None, :],
        out,
        mask=mask_m[:, None] & mask_k[None, :]
    )

def fused_mm_glu(x: torch.Tensor) -> torch.Tensor:
    """
    GLU: a * sigmoid(b)
    x: [M, 2*K]
    return: [M, K]
    """
    assert x.dim() == 2, "only support 2D input"
    assert x.shape[-1] % 2 == 0, "last dim must be even"
    M = x.shape[0]
    K = x.shape[1] // 2
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(K, META["BLOCK_K"]),
    )
    glu_kernel[grid](
        x, out, M, K,
        BLOCK_M=64, BLOCK_K=64,
    )
    return out
