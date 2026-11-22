# /root/FlagOS/FlagGems/src/flag_gems/ops/mm_glu_ops.py
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_mm_kernel(
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
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    b = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )
    acc = tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == b.shape[-2], "mm K dimension mismatch"
    M, K = a.shape[-2], a.shape[-1]
    _, N = b.shape[-2], b.shape[-1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _triton_mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(-2),
        a.stride(-1),
        b.stride(-2),
        b.stride(-1),
        c.stride(-2),
        c.stride(-1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
    )
    return c


@triton.jit
def _triton_addmm_kernel(
    bias_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_bias_m,
    stride_bias_n,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    beta: tl.constexpr,
    alpha: tl.constexpr,
    BIAS_IS_1D: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if BIAS_IS_1D:
        bias = tl.load(
            bias_ptr + offs_n[None, :] * stride_bias_n,
            mask=offs_n[None, :] < N,
            other=0.0,
        )
    else:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )

    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    b = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )
    acc = beta * bias + alpha * tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_addmm(
    bias: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    对齐PyTorch原生接口：addmm(bias, mat1, mat2, beta=1, alpha=1)
    支持 2D bias [M,N]、1D bias [N]广播
    """
    M, K = mat1.shape[-2], mat1.shape[-1]
    _, N = mat2.shape[-2], mat2.shape[-1]
    c = torch.empty((M, N), device=bias.device, dtype=bias.dtype)
    bias_is_1d = bias.dim() == 1
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _triton_addmm_kernel[grid](
        bias,
        mat1,
        mat2,
        c,
        M,
        N,
        K,
        bias.stride(-2) if not bias_is_1d else 0,
        bias.stride(-1),
        mat1.stride(-2),
        mat1.stride(-1),
        mat2.stride(-2),
        mat2.stride(-1),
        c.stride(-2),
        c.stride(-1),
        beta=beta,
        alpha=alpha,
        BIAS_IS_1D=bias_is_1d,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
    )
    return c


@triton.jit
def _triton_bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
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
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_b = pid // (num_pid_m * num_pid_n)
    rem = pid % (num_pid_m * num_pid_n)
    pid_m = rem // num_pid_n
    pid_n = rem % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a = tl.load(
        a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=(pid_b < B) & (offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    b = tl.load(
        b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(pid_b < B) & (offs_k[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )
    acc = tl.dot(a, b)
    tl.store(
        c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(pid_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 3 and b.dim() == 3, "bmm expects 3D tensor [B,M,K] @ [B,K,N]"
    B, M, K = a.shape
    _, _, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (B * triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _triton_bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
    )
    return c


@triton.jit
def _triton_glu_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    off_m = pid
    off_n = tl.arange(0, BLOCK_N)
    half_N = N // 2

    a = tl.load(x_ptr + off_m * stride_xm + off_n * stride_xn, mask=off_n < half_N, other=0.0)
    b = tl.load(
        x_ptr + off_m * stride_xm + (off_n + half_N) * stride_xn,
        mask=off_n < half_N,
        other=0.0,
    )
    y = a * tl.sigmoid(b)
    tl.store(out_ptr + off_m * stride_om + off_n * stride_on, y, mask=off_n < half_N)


def triton_glu_fused(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, "GLU last dimension must be even"
    out_shape = list(x.shape)
    out_shape[-1] = out_shape[-1] // 2
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    grid = (M,)
    _triton_glu_kernel[grid](
        x,
        out,
        M,
        N,
        x.stride(-2),
        x.stride(-1),
        out.stride(-2),
        out.stride(-1),
        BLOCK_N=64,
    )
    return out
