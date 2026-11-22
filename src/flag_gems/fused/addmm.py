import torch
import triton
import triton.language as tl


@triton.jit
def addmm_kernel(
    bias_ptr,
    mat1_ptr,
    mat2_ptr,
    out_ptr,
    M,
    N,
    K,
    beta,
    alpha,
    stride_m1_m,
    stride_m1_k,
    stride_m2_k,
    stride_m2_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mat1_ptrs = mat1_ptr + offs_m[:, None] * stride_m1_m + offs_k[None, :] * stride_m1_k
    mat2_ptrs = mat2_ptr + offs_k[:, None] * stride_m2_k + offs_n[None, :] * stride_m2_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(mat1_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(mat2_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        mat1_ptrs += BLOCK_K * stride_m1_k
        mat2_ptrs += BLOCK_K * stride_m2_k

    acc = acc * alpha
    bias = tl.load(bias_ptr + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0)
    acc = acc + bias * beta

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
):
    """
    torch.addmm(input, mat1, mat2, *, beta=beta, alpha=alpha)
    out = beta * input + alpha * (mat1 @ mat2)
    Restriction: input bias support shape (N,)
    """
    assert mat1.dim() == 2 and mat2.dim() == 2, "mat1 and mat2 must be 2D"
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2, f"K mismatch: mat1 K={K}, mat2 K={K2}"
    assert input.shape == (N,), f"input bias must be shape ({N},), got {input.shape}"

    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    addmm_kernel[grid](
        input,
        mat1,
        mat2,
        out,
        M,
        N,
        K,
        beta,
        alpha,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out
