import torch
import triton
import triton.language as tl

@triton.jit
def _mm_conv_glu_kernel(
    a_ptr, w_ptr, out_ptr,
    M, Cout, K,
    stride_am, stride_ak,
    stride_wk, stride_w2c,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N * 2), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        w = tl.load(
            w_ptr + (k + offs_k)[:, None] * stride_wk + (offs_n * 2)[None, :] * stride_w2c,
            mask=((k + offs_k)[:, None] < K) & ((offs_n * 2)[None, :] < Cout * 2),
            other=0.0
        )
        acc += tl.dot(a, w)

    x = acc[:, 0:BLOCK_N]
    gate = acc[:, BLOCK_N:BLOCK_N*2]
    glu_val = x * tl.sigmoid(gate)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < Cout)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        glu_val.to(tl.float16),
        mask=out_mask
    )

def iluvatar_mm_conv_glu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert w.shape[1] % 2 == 0
    M, K = x.shape
    K2, two_cout = w.shape
    assert K == K2
    Cout = two_cout // 2
    out = torch.empty((M, Cout), device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_N))

    _mm_conv_glu_kernel[grid](
        x, w, out,
        M, Cout, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out
