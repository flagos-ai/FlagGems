import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from ..utils.shape_utils import can_use_int32_index


def cfggen():
    warps = [1, 2, 4, 8, 16, 32]
    configs = [
        triton.Config({"M_BLOCK_SIZE": 1, "N_BLOCK_SIZE": 2048}, num_warps=w)
        for w in warps
    ]
    return configs


def cfggen_batch():
    warps = [1, 2, 4, 8, 16, 32]
    configs = [
        triton.Config({"BATCH_BLOCK_SIZE": 1, "MN_BLOCK_SIZE": 512}, num_warps=w)
        for w in warps
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask

        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(row + diagonal <= cols, x, 0.0)
        tl.store(Y + cols, y, mask=mask)


@libentry()
@triton.autotune(configs=cfggen_batch(), key=["batch", "MN", "N", "diagonal"])
@triton.jit(do_not_specialize=["diagonal"])
def triu_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)
    if INT64_INDEX:
        batch_id = batch_id.to(tl.int64)
        mn_id = mn_id.to(tl.int64)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask and mn_mask
    x = tl.load(X + cols, mask, other=0.0)
    m = cols // N
    n = cols % N
    y = tl.where(m + diagonal <= n, x, 0.0)
    tl.store(Y + cols, y, mask=mask)


INT32_MAX = torch.iinfo(torch.int32).max


def triu(A, diagonal=0):
    logging.debug("GEMS TRIU")
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    use_int64_index = not can_use_int32_index(A)
    M, N = A.shape[-2:]
    with torch.cuda.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A, out, M, N, diagonal, INT64_INDEX=use_int64_index)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            triu_batch_kernel[grid](
                B, out, batch, M * N, N, diagonal, INT64_INDEX=use_int64_index
            )
            out = out.view(A.shape)
    return out
