import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, out_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(pid_b < B) & (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + pid_b * stride_bb + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(pid_b < B) & ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    out_val = acc.to(tl.float16)
    tl.store(
        out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out_val,
        mask=(pid_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    )

def iluvatar_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _bmm_kernel[grid](
        a, b, out,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
