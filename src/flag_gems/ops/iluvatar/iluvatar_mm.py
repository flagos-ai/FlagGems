import torch
import triton
import triton.language as tl

@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_offs + offs_k)[None, :] * stride_ak,
            mask=mask_m[:, None] & ((k_offs + offs_k) < K)[None, :],
            other=0.0
        )
        b = tl.load(
            b_ptr + (k_offs + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k_offs + offs_k) < K)[:, None] & mask_n[None, :],
            other=0.0
        )
        acc += tl.dot(a, b)

    out = acc.to(tl.float16)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=out_mask)

def iluvatar_mm(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # host侧：python整数上取整，禁止tl.cdiv
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    _mm_kernel[grid](
        a, b, out,
        M, K, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out
