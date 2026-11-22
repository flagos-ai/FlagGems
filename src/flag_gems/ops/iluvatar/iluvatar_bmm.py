import torch
import triton
import triton.language as tl

@triton.jit
def _iluvatar_bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
):
    pid = tl.program_id(axis=0)
    tiles_mn = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    batch_id = pid // tiles_mn
    rem = pid % tiles_mn
    pid_m = rem // tl.cdiv(N, BLOCK_N)
    pid_n = rem % tl.cdiv(N, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_base = a_ptr + batch_id * stride_ab
    b_base = b_ptr + batch_id * stride_bb
    c_base = c_ptr + batch_id * stride_cb

    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K
        mask_k = k_offs + offs_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def iluvatar_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.device.type == "privateuseone"
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (B * triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
    _iluvatar_bmm_kernel[grid](
        a, b, out,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return out
