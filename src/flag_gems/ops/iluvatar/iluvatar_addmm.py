import torch
import triton
import triton.language as tl

@triton.jit
def _iluvatar_addmm_kernel(
    bias_ptr, mat1_ptr, mat2_ptr, out_ptr,
    M, N, K,
    beta, alpha,
    stride_bm, stride_bn,
    stride_m1m, stride_m1k,
    stride_m2k, stride_m2n,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    m1_ptrs = mat1_ptr + offs_m[:, None] * stride_m1m + offs_k[None, :] * stride_m1k
    m2_ptrs = mat2_ptr + offs_k[:, None] * stride_m2k + offs_n[None, :] * stride_m2n

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K
        mask_k = k_offs + offs_k < K
        m1 = tl.load(m1_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        m2 = tl.load(m2_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(m1, m2)
        m1_ptrs += BLOCK_SIZE_K * stride_m1k
        m2_ptrs += BLOCK_SIZE_K * stride_m2k

    bias_ptrs = bias_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    b = tl.load(bias_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    acc = beta * b + alpha * acc

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def iluvatar_addmm(bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """FlagGems standard: addmm(bias, mat1, mat2, beta, alpha)"""
    assert bias.device.type == "privateuseone"
    M, N = bias.shape
    M1, K = mat1.shape
    K2, N2 = mat2.shape
    assert M == M1 and N == N2 and K == K2
    out = torch.empty_like(bias)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    _iluvatar_addmm_kernel[grid](
        bias, mat1, mat2, out,
        M, N, K,
        beta, alpha,
        bias.stride(0), bias.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
    )
    return out
