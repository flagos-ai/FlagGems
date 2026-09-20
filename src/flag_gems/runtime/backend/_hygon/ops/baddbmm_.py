import torch
import triton
import triton.language as tl


@triton.jit
def _baddbmm_lean(
    a,
    b,
    c,
    B,
    M,
    N,
    K,
    beta,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Contiguous specialization: a (B,M,K), b (B,K,N), c (B,M,N) row-major.
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a + pid_b * (M * K) + offs_m[:, None] * K + offs_k[None, :]
    b_base = b + pid_b * (K * N) + offs_k[:, None] * N + offs_n[None, :]
    c_base = c + pid_b * (M * N) + offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN:
        for k0 in range(0, K, BLOCK_K):
            a_tile = tl.load(a_base)
            b_tile = tl.load(b_base)
            acc = tl.dot(a_tile, b_tile, acc)
            a_base += BLOCK_K
            b_base += BLOCK_K * N
        c_val = tl.load(c_base)
        out = alpha.to(tl.float32) * acc + beta.to(tl.float32) * c_val
        tl.store(c_base, out.to(c_val.dtype))
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            a_tile = tl.load(a_base, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b_tile = tl.load(b_base, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc = tl.dot(a_tile, b_tile, acc)
            a_base += BLOCK_K
            b_base += BLOCK_K * N
        c_val = tl.load(c_base, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        out = alpha.to(tl.float32) * acc + beta.to(tl.float32) * c_val
        tl.store(c_base, out.to(c_val.dtype), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _baddbmm_general(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    beta,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a
        + pid_b * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b
        + pid_b * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    c_ptrs = (
        c
        + pid_b * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN:
        for k0 in range(0, K, BLOCK_K):
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            acc = tl.dot(a_tile, b_tile, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_val = tl.load(c_ptrs)
        out = alpha.to(tl.float32) * acc + beta.to(tl.float32) * c_val
        tl.store(c_ptrs, out.to(c_val.dtype))
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            a_tile = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc = tl.dot(a_tile, b_tile, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_val = tl.load(c_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        out = alpha.to(tl.float32) * acc + beta.to(tl.float32) * c_val
        tl.store(c_ptrs, out.to(c_val.dtype), mask=m_mask[:, None] & n_mask[None, :])


def _pick_config(dtype, M, N):
    small = (M <= 384) and (N <= 384)
    if dtype in (torch.float16, torch.bfloat16):
        if small:
            return (64, 64, 32, 4, 2)
        if (M <= 2048) and (N <= 2048):
            return (128, 128, 32, 2, 2)
        return (128, 128, 32, 4, 2)
    # float32 and friends
    if small:
        return (64, 64, 32, 8, 2)
    return (64, 64, 32, 8, 2)


def run(self, batch1, batch2, *, beta=1, alpha=1):
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    beta = float(beta)
    alpha = float(alpha)

    B, M, N = self.shape
    K = batch1.shape[2]
    BLOCK_M, BLOCK_N, BLOCK_K, nw, ns = _pick_config(self.dtype, M, N)
    even = (M % BLOCK_M == 0) and (N % BLOCK_N == 0) and (K % BLOCK_K == 0)
    kp = 2 if self.dtype == torch.float32 else 1
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), B)

    if batch1.is_contiguous() and batch2.is_contiguous() and self.is_contiguous():
        _baddbmm_lean[grid](
            batch1,
            batch2,
            self,
            B,
            M,
            N,
            K,
            beta,
            alpha,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=8,
            EVEN=even,
            num_warps=nw,
            num_stages=ns,
            kpack=kp,
        )
    else:
        _baddbmm_general[grid](
            batch1,
            batch2,
            self,
            M,
            N,
            K,
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            self.stride(0),
            self.stride(1),
            self.stride(2),
            beta,
            alpha,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=8,
            EVEN=even,
            num_warps=nw,
            num_stages=ns,
            kpack=kp,
        )
    return self


# Alias for FlagGems import convention
baddbmm_ = run
