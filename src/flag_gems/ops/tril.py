import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _tril_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    diag,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Whole-block fast paths: a tile entirely above the diagonal is zero,
    # a tile entirely on/below the diagonal is a straight copy.
    block_min_n = pid_n * BLOCK_N
    block_max_n = block_min_n + BLOCK_N - 1
    block_min_m = pid_m * BLOCK_M
    block_max_m = block_min_m + BLOCK_M - 1

    base = pid_b * M * N
    idxs = base + offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    if block_min_n > block_max_m + diag:
        # Entire tile is strictly above the diagonal — write zeros.
        zero = tl.zeros([BLOCK_M, BLOCK_N], dtype=in_ptr.dtype.element_ty)
        tl.store(out_ptr + idxs, zero, mask=mask)
    elif block_max_n <= block_min_m + diag:
        # Entire tile is on/below the diagonal — straight copy.
        x = tl.load(in_ptr + idxs, mask=mask, other=0)
        tl.store(out_ptr + idxs, x, mask=mask)
    else:
        # Boundary tile — apply the per-element keep mask.
        x = tl.load(in_ptr + idxs, mask=mask, other=0)
        keep = offs_n <= (offs_m + diag)
        y = tl.where(keep, x, 0)
        tl.store(out_ptr + idxs, y, mask=mask)


def tril(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS TRIL")
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"

    input = input.contiguous()
    out = torch.empty_like(input)

    M = input.size(-2)
    N = input.size(-1)
    B = input.numel() // (M * N) if M * N > 0 else 0

    if M == 0 or N == 0 or B == 0:
        return out

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        B,
    )

    with torch_device_fn.device(input.device):
        _tril_kernel[grid](input, out, M, N, int(diagonal))
    return out


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    logger.debug("GEMS TRIL_OUT")
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert out is not None, "tril_out: out tensor is required"
    assert out.shape == input.shape, "Input and output must have the same shape"
    assert out.dtype == input.dtype, "Input and output must have the same dtype"
    assert out.device == input.device, "Input and output must be on the same device"

    input = input.contiguous()
    if not out.is_contiguous():
        # Run the kernel into a contiguous buffer, then copy.
        tmp = tril(input, diagonal)
        out.copy_(tmp)
        return out

    M = input.size(-2)
    N = input.size(-1)
    B = input.numel() // (M * N) if M * N > 0 else 0
    if M == 0 or N == 0 or B == 0:
        return out

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        B,
    )
    with torch_device_fn.device(input.device):
        _tril_kernel[grid](input, out, M, N, int(diagonal))
    return out
