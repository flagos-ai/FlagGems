import torch
import triton
import triton.language as tl


@triton.jit
def _tril_kernel(in_ptr, out_ptr, M, N, B, diag,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)

    base = pid_b * M * N
    idxs = base + offs_m * N + offs_n

    x = tl.load(in_ptr + idxs, mask=mask, other=0)
    keep = offs_n <= (offs_m + diag)
    y = tl.where(keep, x, 0)
    tl.store(out_ptr + idxs, y, mask=mask)


def _launch_tril_kernel(input: torch.Tensor, out: torch.Tensor, diagonal: int):
    assert input.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert input.is_contiguous() and out.is_contiguous(), "Only contiguous tensors are supported"
    assert input.shape == out.shape, "Input and output must have the same shape"
    assert input.dtype == out.dtype, "Input and output must have the same dtype"

    if input.dim() < 2:
        out.copy_(input)
        return out

    M = input.size(-2)
    N = input.size(-1)
    B = input.numel() // (M * N)

    if M == 0 or N == 0 or B == 0:
        # Nothing to compute
        return out

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)

    _tril_kernel[grid](
        input, out,
        M, N, B, int(diagonal),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4
    )
    return out


def tril(input: torch.Tensor, diagonal: int = 0):
    out = torch.empty_like(input)
    return _launch_tril_kernel(input, out, diagonal)


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(input)
    _launch_tril_kernel(input, out, diagonal)
    return out