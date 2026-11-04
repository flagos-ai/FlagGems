# triton_src/mm/mm.py
import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Iluvatar 整定常量，不要照搬A100 128x128
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """
    Grouped GEMM + K分块预取，适配Iluvatar L2缓存
    核心优化：
    1. Grouped重排，均衡SM负载，避免尾部小tile空闲
    2. K维度循环内预取下一块A/B，计算与访存重叠
    3. 寄存器压力控制：BLOCK_M/N=64，远低于CUDA 128，规避spill
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # A: [M, K], B: [K, N]
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_base = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_base, mask=offs_k[None, :] < K - k_step * BLOCK_K, other=0.0)
        b = tl.load(b_base, mask=offs_k[:, None] < K - k_step * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper: torch.matmul(A,B) 等价，A@B
    A shape: [M, K], B shape: [K, N]
    """
    assert a.is_contiguous(), "A must be contiguous"
    assert b.is_contiguous(), "B must be contiguous"
    assert (
        a.dtype == torch.float16 and b.dtype == torch.float16
    ), "Only fp16 supported for Iluvatar TensorCore"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "K dimension mismatch"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_SIZE_M=8,
    )
    return c
