import torch
import triton
import triton.language as tl

@triton.jit
def glu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M * (N // 2)

    # GLU: split x into a, b along last dim; out = a * sigmoid(b)
    a_offs = offs
    b_offs = offs + M * (N // 2)

    a = tl.load(x_ptr + a_offs, mask=mask, other=0.0)
    b = tl.load(x_ptr + b_offs, mask=mask, other=0.0)
    out = a * tl.sigmoid(b)
    tl.store(out_ptr + offs, out, mask=mask)

def fused_mm_glu(x: torch.Tensor) -> torch.Tensor:
    """
    GLU: a * sigmoid(b)
    x shape: [M, 2*K]
    return shape: [M, K]
    """
    assert x.shape[-1] % 2 == 0, "last dim must be even for GLU"
    M = x.shape[0]
    N = x.shape[1]
    K = N // 2

    out = torch.empty((M, K), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M * K, meta["BLOCK_SIZE"]),)
    glu_kernel[grid](x, out, M, N, BLOCK_SIZE=64)
    return out
