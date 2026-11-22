import torch
import triton
import triton.language as tl

@triton.jit
def _mm_conv_glu_kernel(
    a_ptr, w_ptr, out_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        w_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        a_val = tl.load(a, mask=a_mask, other=0.0)
        w_val = tl.load(w, mask=w_mask, other=0.0)
        acc += tl.dot(a_val, w_val)
        a += BLOCK_K * stride_ak
        w += BLOCK_K * stride_wn

    out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out, acc, mask=out_mask)

def iluvatar_mm_conv_glu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Fused MM + GLU:
    y = mm(x, weight) -> split last dim into a, b; out = a * sigmoid(b)
    weight.shape = [K, 2*O]
    output shape = [M, O]
    example: x (2,8), weight (8,8) => mm output (2,8) -> split (2,4) & (2,4) -> out (2,4)
    """
    M, K = x.shape
    Kw, twoO = weight.shape
    assert K == Kw
    assert twoO % 2 == 0
    O = twoO // 2

    mm_out = torch.empty((M, twoO), device=x.device, dtype=x.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(twoO, BLOCK_N))
    _mm_conv_glu_kernel[grid](
        x, weight, mm_out,
        M, K, twoO,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        mm_out.stride(0), mm_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    # 正确GLU：对半切
    a, b = mm_out.split(O, dim=-1)
    return a * torch.sigmoid(b)
