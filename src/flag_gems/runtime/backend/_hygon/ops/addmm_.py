import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fast path: shapes exactly divisible by the tile sizes (no masking anywhere).
# ---------------------------------------------------------------------------
@triton.jit
def _addmm_even(
    self_ptr,
    mat1_ptr,
    mat2_ptr,
    M,
    N,
    K,
    beta,
    alpha,
    stride_self_m,
    stride_self_n,
    stride_mat1_m,
    stride_mat1_k,
    stride_mat2_k,
    stride_mat2_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
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

    mat1_ptrs = (
        mat1_ptr + offs_m[:, None] * stride_mat1_m + offs_k[None, :] * stride_mat1_k
    )
    mat2_ptrs = (
        mat2_ptr + offs_k[:, None] * stride_mat2_k + offs_n[None, :] * stride_mat2_n
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(mat1_ptrs)
        b = tl.load(mat2_ptrs)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        mat1_ptrs += BLOCK_K * stride_mat1_k
        mat2_ptrs += BLOCK_K * stride_mat2_k

    self_ptrs = (
        self_ptr + offs_m[:, None] * stride_self_m + offs_n[None, :] * stride_self_n
    )
    s = tl.load(self_ptrs)
    acc = acc * alpha + s * beta
    tl.store(self_ptrs, acc)


# ---------------------------------------------------------------------------
# General path: masked loads/stores for arbitrary shapes.
# ---------------------------------------------------------------------------
@triton.jit
def _addmm_masked(
    self_ptr,
    mat1_ptr,
    mat2_ptr,
    M,
    N,
    K,
    beta,
    alpha,
    stride_self_m,
    stride_self_n,
    stride_mat1_m,
    stride_mat1_k,
    stride_mat2_k,
    stride_mat2_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
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

    mat1_ptrs = (
        mat1_ptr + offs_m[:, None] * stride_mat1_m + offs_k[None, :] * stride_mat1_k
    )
    mat2_ptrs = (
        mat2_ptr + offs_k[:, None] * stride_mat2_k + offs_n[None, :] * stride_mat2_n
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (k_off + offs_k[None, :] < K)
        b_mask = (k_off + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(mat1_ptrs, mask=a_mask, other=0.0)
        b = tl.load(mat2_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        mat1_ptrs += BLOCK_K * stride_mat1_k
        mat2_ptrs += BLOCK_K * stride_mat2_k

    self_ptrs = (
        self_ptr + offs_m[:, None] * stride_self_m + offs_n[None, :] * stride_self_n
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    s = tl.load(self_ptrs, mask=out_mask, other=0.0)
    acc = acc * alpha + s * beta
    tl.store(self_ptrs, acc, mask=out_mask)


def run(self, mat1, mat2, *, beta=1, alpha=1):
    """In-place addmm: self = beta * self + alpha * (mat1 @ mat2). Returns self."""
    assert mat1.dim() == 2 and mat2.dim() == 2
    M, K = mat1.shape
    K2, N = mat2.shape
    assert K == K2

    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    # self strides, broadcast-aware (PyTorch broadcasting semantics).
    if self.dim() == 2:
        stride_self_m, stride_self_n = self.stride()
    elif self.dim() == 1:
        stride_self_m, stride_self_n = 0, self.stride()[0]
    elif self.dim() == 0:
        stride_self_m, stride_self_n = 0, 0
    else:
        raise ValueError("unsupported self dim: %d" % self.dim())

    stride_mat1_m, stride_mat1_k = mat1.stride()
    stride_mat2_k, stride_mat2_n = mat2.stride()

    # Tuned per dtype family and shape (Hygon BW / gfx936, 64KB LDS, 64-thread warps).
    if self.dtype in (torch.float16, torch.bfloat16):
        if N >= 2048:
            # Fat N-tiles give ~179 TF fp16 / ~176 TF bf16 on large squares.
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS, STAGES = 128, 256, 32, 2, 4, 2
        elif M <= 512 and N <= 512:
            # Small shapes: many small CTAs shorten the latency-bound path.
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS, STAGES = 64, 64, 32, 8, 4, 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS, STAGES = 128, 128, 64, 8, 4, 2
    else:
        if M <= 512 and N <= 512:
            # Small fp32: 72 single-warp CTAs maximize latency hiding.
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS, STAGES = 32, 64, 32, 8, 1, 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, WARPS, STAGES = 128, 64, 32, 8, 4, 2

    even = (M % BLOCK_M == 0) and (N % BLOCK_N == 0) and (K % BLOCK_K == 0)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    # mmac5-ds6: HCU backend instruction-scheduling latency tuning (mmac=5,
    # ds-load-store=6); measured +1.3-1.7% on large fp16/bf16 tiles, neutral fp32.
    if even:
        _addmm_even[grid](
            self,
            mat1,
            mat2,
            M,
            N,
            K,
            beta,
            alpha,
            stride_self_m,
            stride_self_n,
            stride_mat1_m,
            stride_mat1_k,
            stride_mat2_k,
            stride_mat2_n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=WARPS,
            num_stages=STAGES,
            sched_latency="mmac5-ds6",
        )
    else:
        _addmm_masked[grid](
            self,
            mat1,
            mat2,
            M,
            N,
            K,
            beta,
            alpha,
            stride_self_m,
            stride_self_n,
            stride_mat1_m,
            stride_mat1_k,
            stride_mat2_k,
            stride_mat2_n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            num_warps=WARPS,
            num_stages=STAGES,
            sched_latency="mmac5-ds6",
        )
    return self


# Alias for FlagGems import convention
addmm_ = run
