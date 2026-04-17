import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


_TRIL_AUTOTUNE_CONFIGS = [
    triton.Config({"M_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
    triton.Config({"M_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"M_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
    triton.Config({"M_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 128}, num_warps=8, num_stages=2),
    triton.Config({"M_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 64}, num_warps=8, num_stages=2),
    triton.Config(
        {"M_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128}, num_warps=8, num_stages=2
    ),
]


@libentry()
@triton.autotune(configs=_TRIL_AUTOTUNE_CONFIGS, key=["batch", "M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_tile_kernel(
    X,
    Y,
    batch,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_m = tle.program_id(1)
    pid_n = tle.program_id(2)

    rows = pid_m * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)
    cols = pid_n * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    row_start = pid_m * M_BLOCK_SIZE
    row_end = row_start + M_BLOCK_SIZE - 1
    col_start = pid_n * N_BLOCK_SIZE
    col_end = col_start + N_BLOCK_SIZE - 1

    base = pid_b * M * N
    ptrs = X + base + rows[:, None] * N + cols[None, :]
    out_ptrs = Y + base + rows[:, None] * N + cols[None, :]

    # Entire tile lies in the strictly upper-triangular region.
    if col_start > row_end + diagonal:
        tl.store(out_ptrs, 0, mask=mask)
        return

    # Entire tile lies in the kept lower-triangular region.
    if col_end <= row_start + diagonal:
        values = tl.load(ptrs, mask=mask, other=0)
        tl.store(out_ptrs, values, mask=mask)
        return

    values = tl.load(ptrs, mask=mask, other=0)
    keep = cols[None, :] <= rows[:, None] + diagonal
    tl.store(out_ptrs, tl.where(keep, values, 0), mask=mask)


def _check_batch_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor

    if tensor.dim() < 2:
        return tensor.contiguous()

    n = tensor.size(-1)
    stride_row, stride_col = tensor.stride(-2), tensor.stride(-1)
    if not (stride_col == 1 and stride_row == n):
        return tensor.contiguous()

    expected_stride = tensor.size(-1) * tensor.size(-2)
    for i in range(tensor.dim() - 3, -1, -1):
        if tensor.stride(i) != expected_stride:
            return tensor.contiguous()
        expected_stride *= tensor.size(i)

    return tensor


def _launch_tril(input_to_use: torch.Tensor, out: torch.Tensor, diagonal: int):
    M, N = input_to_use.shape[-2:]
    if input_to_use.numel() == 0:
        return out

    batch = 1 if input_to_use.dim() == 2 else int(input_to_use.numel() / M / N)

    with torch_device_fn.device(input_to_use.device):
        grid = lambda meta: (
            batch,
            triton.cdiv(M, meta["M_BLOCK_SIZE"]),
            triton.cdiv(N, meta["N_BLOCK_SIZE"]),
        )
        tril_tile_kernel[grid](input_to_use, out, batch, M, N, diagonal)
    return out


def tril(input: torch.Tensor, diagonal: int = 0):
    logger.debug("GEMS TRIL")

    if input.dim() < 2:
        raise RuntimeError("tril: input tensor must have at least 2 dimensions")

    input_to_use = _check_batch_contiguous(input)
    out = torch.empty(
        input.shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    return _launch_tril(input_to_use, out, int(diagonal))


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    logger.debug("GEMS TRIL.OUT")

    if out is None:
        return tril(input, diagonal)
    if input.dim() < 2:
        raise RuntimeError("tril: input tensor must have at least 2 dimensions")
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {input.device}, but got {out.device} instead"
        )
    if out.shape != input.shape:
        out.resize_(input.shape)

    input_to_use = _check_batch_contiguous(input)
    if out.is_contiguous():
        return _launch_tril(input_to_use, out, int(diagonal))

    result = tril(input_to_use, diagonal)
    out.copy_(result)
    return out
