"""
FP8 Matrix Multiplication — Triton Kernel (Block-wise Scaling)
Fixed config version for H20 deployment (no autotune warmup).

API:
    fp8_matmul(a, a_s, b, b_s) -> Tensor

    a:   (..., K)                    float8_e4m3fn, contiguous
    a_s: (..., K // group_size)      float32, per-token-group scale
    b:   (N, K)                      float8_e4m3fn, contiguous
    b_s: (N // group_size, K // group_size)  float32, per-block scale
    group_size = 128

    Returns: (..., N) bfloat16

Based on v45. Fixed config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=128,
GROUP_SIZE_M=4, num_stages=3, num_warps=4 (best for M>=128 on H20).
"""

import torch
import triton
import triton.language as tl

GROUP_SIZE = 128

# Fixed config — best for M>=128 on H20 (covers majority of production shapes)
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 128
GROUP_SIZE_M = 4
NUM_STAGES = 3
NUM_WARPS = 4

# Debug print helper (flush immediately for real-time visibility)
# def _p(msg):
#     rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
#     print(f"[fp8_matmul][rank{rank}] {msg}", flush=True)


@triton.jit
def _fp8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_as_k,
    stride_bs_n,
    stride_bs_k,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    as_ptrs = As + offs_m * stride_as_m
    offs_bs_n = offs_n // GROUP_K
    bs_scalar_n_idx = pid_n * BLOCK_N // GROUP_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_iters):
        k_idx = k * BLOCK_K // GROUP_K
        a_s = tl.load(as_ptrs + k_idx * stride_as_k)

        if BLOCK_N <= GROUP_K:
            b_s_val = tl.load(Bs + bs_scalar_n_idx * stride_bs_n + k_idx * stride_bs_k)
        else:
            b_s = tl.load(Bs + offs_bs_n * stride_bs_n + k_idx * stride_bs_k)

        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)

        dot = tl.dot(a, tl.trans(b))

        if BLOCK_N <= GROUP_K:
            acc += dot * (a_s[:, None] * b_s_val)
        else:
            acc += dot * (a_s[:, None] * b_s[None, :])

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def fp8_matmul(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Block-wise scaled FP8 matrix multiplication.

    Args:
        a:   (..., K)                        float8_e4m3fn, contiguous
        a_s: (..., K // 128)                 float32, per-token-group scale
        b:   (N, K)                          float8_e4m3fn, contiguous
        b_s: (N // 128, K // 128)            float32, per-block scale
    Returns:
        (..., N) bfloat16
    """
    assert b.ndim == 2
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    K = a.size(-1)
    M = a.numel() // K
    N, K2 = b.shape
    assert K == K2

    if scale_dtype == torch.float8_e8m0fnu:
        a_s = a_s.to(torch.float32)
        b_s = b_s.to(torch.float32)

    out_shape = (*a.size()[:-1], N)
    a_2d = a.view(M, K)
    a_s_2d = a_s.view(M, -1)

    C = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _fp8_matmul_kernel[grid](
        a_2d,
        b,
        C,
        a_s_2d,
        b_s,
        M,
        N,
        K,
        a_2d.stride(0),
        a_2d.stride(1),
        b.stride(0),
        b.stride(1),
        C.stride(0),
        C.stride(1),
        a_s_2d.stride(0),
        a_s_2d.stride(1),
        b_s.stride(0),
        b_s.stride(1),
        GROUP_K=GROUP_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )

    return C.view(out_shape)
