import torch
import triton
import triton.language as tl
from .__libentry__ import libentry


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
@triton.jit
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
@triton.jit
def triu_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    # version 1
    # batch_id = tl.program_id(0)
    # row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    # batch_mask = row < batch
    # X += row * MN
    # Y += row * MN

    # for mn_offset in range(0, MN, MN_BLOCK_SIZE):
    #     cols = mn_offset + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    #     mn_mask = cols < MN
    #     mask = batch_mask and mn_mask
    #     x = tl.load(X + cols, mask, other=0.0)
    #     m = cols // N
    #     n = cols % N
    #     y = tl.where(m + diagonal <= n, x, 0.0)
    #     tl.store(Y + cols, y, mask=mask)

    # version 2
    batch_id = tl.program_id(0)
    mn_id = tl.program_id(1)
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


def triu(A, diagonal=0, *, out=None):
    if __debug__:
        print("FLAG TRIU")
    if out == None:
        O = torch.empty_like(A)
    else:
        O = out
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    if len(A.shape) == 2:
        grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
        triu_kernel[grid](A, O, M, N, diagonal)
    else:
        batch = int(torch.numel(A) / M / N)
        print(batch)
        A = A.contiguous()
        B = A.view(batch, -1)
        grid = lambda meta: (
            triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
            triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
        )
        triu_batch_kernel[grid](B, O, batch, M * N, N, diagonal)
        O = O.view(A.shape)
    return O
