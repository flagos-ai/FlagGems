import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("triu"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * M_BLOCK_SIZE
    rows_in_block = tl.arange(0, M_BLOCK_SIZE) + row_start

    for n_offset in range(0, N, N_BLOCK_SIZE):
        x_block_ptr = tl.make_block_ptr(
            base=X,
            shape=(M, N),
            strides=(N, 1),
            offsets=(row_start, n_offset),
            block_shape=(M_BLOCK_SIZE, N_BLOCK_SIZE),
            order=(1, 0),
        )
        y_block_ptr = tl.make_block_ptr(
            base=Y,
            shape=(M, N),
            strides=(N, 1),
            offsets=(row_start, n_offset),
            block_shape=(M_BLOCK_SIZE, N_BLOCK_SIZE),
            order=(1, 0),
        )

        block_row_min = row_start
        block_row_max = row_start + M_BLOCK_SIZE - 1
        block_col_min = n_offset
        block_col_max = n_offset + N_BLOCK_SIZE - 1

        if block_row_max + diagonal <= block_col_min:
            x = tl.load(x_block_ptr, boundary_check=(0, 1))
            tl.store(y_block_ptr, x, boundary_check=(0, 1))
        elif block_row_min + diagonal > block_col_max:
            zeros = tl.full((M_BLOCK_SIZE, N_BLOCK_SIZE), 0, tl.load(x_block_ptr, boundary_check=(0, 1)).dtype)
            tl.store(y_block_ptr, zeros, boundary_check=(0, 1))
        else:
            x = tl.load(x_block_ptr, boundary_check=(0, 1))
            cols_in_block = tl.arange(0, N_BLOCK_SIZE) + n_offset
            row_idx = rows_in_block[:, None]
            col_idx = cols_in_block[None, :]
            triu_mask = row_idx + diagonal <= col_idx
            y = tl.where(triu_mask, x, 0.0)
            tl.store(y_block_ptr, y, boundary_check=(0, 1))


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("triu_batch"),
    key=["batch", "M", "N", "diagonal"],
)
@triton.jit(do_not_specialize=["diagonal"])
def triu_batch_kernel(
    X,
    Y,
    batch,
    M,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    row_block_id = tl.program_id(1)
    col_block_id = tl.program_id(2)

    batch_start = batch_id * BATCH_BLOCK_SIZE
    row_start = row_block_id * M_BLOCK_SIZE
    col_start = col_block_id * N_BLOCK_SIZE

    batch_offsets = tl.arange(0, BATCH_BLOCK_SIZE) + batch_start
    row_offsets = tl.arange(0, M_BLOCK_SIZE) + row_start
    col_offsets = tl.arange(0, N_BLOCK_SIZE) + col_start

    block_row_min = row_start
    block_row_max = row_start + M_BLOCK_SIZE - 1
    block_col_min = col_start
    block_col_max = col_start + N_BLOCK_SIZE - 1

    base_offsets = batch_offsets[:, None, None] * (M * N)
    row_offsets_3d = row_offsets[None, :, None] * N
    col_offsets_3d = col_offsets[None, None, :]
    offsets = base_offsets + row_offsets_3d + col_offsets_3d

    batch_mask = batch_offsets < batch
    row_mask = row_offsets < M
    col_mask = col_offsets < N
    valid_mask = batch_mask[:, None, None] & row_mask[None, :, None] & col_mask[None, None, :]

    if block_row_max + diagonal <= block_col_min:
        x = tl.load(X + offsets, mask=valid_mask, other=0.0)
        tl.store(Y + offsets, x, mask=valid_mask)
        return

    if block_row_min + diagonal > block_col_max:
        zeros = tl.zeros((BATCH_BLOCK_SIZE, M_BLOCK_SIZE, N_BLOCK_SIZE), dtype=tl.int1)
        tl.store(Y + offsets, zeros, mask=valid_mask)
        return

    x = tl.load(X + offsets, mask=valid_mask, other=0.0)
    triu_mask = row_offsets[None, :, None] + diagonal <= col_offsets[None, None, :]
    y = tl.where(triu_mask, x, 0.0)
    tl.store(Y + offsets, y, mask=valid_mask)


def triu(A, diagonal=0):
    logger.debug("GEMS_SPACEMIT TRIU")
    A = A.contiguous()
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]

    if diagonal >= N:
        return torch.zeros_like(A)
    if diagonal <= -M:
        return A.clone()

    out = torch.empty_like(A)
    with torch_device_fn.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A, out, M, N, diagonal)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(-1)
            out_view = out.view(-1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M, meta["M_BLOCK_SIZE"]),
                triton.cdiv(N, meta["N_BLOCK_SIZE"]),
            )
            triu_batch_kernel[grid](
                B,
                out_view,
                batch,
                M,
                N,
                diagonal,
            )
    return out
