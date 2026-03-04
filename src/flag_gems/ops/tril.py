import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def tril_kernel(
    X,
    Y,
    M,
    N,
    diagonal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offset = pid_batch * M * N + m_offset[:, None] * N + n_offset[None, :]
    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)

    x = tl.load(X + offset, mask=mask, other=0.0)

    keep_mask = m_offset[:, None] >= (n_offset[None, :] - diagonal)
    y = tl.where(keep_mask, x, 0.0)

    tl.store(Y + offset, y, mask=mask)


def tril(A, diagonal=0):
    logger.debug("GEMS TRIL")
    shape = A.shape
    if A.ndim < 2:
        raise ValueError("tril requires at least 2D tensor")

    M, N = shape[-2], shape[-1]
    batch_size = A.numel() // (M * N)

    out = torch.empty_like(A)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        batch_size,
    )

    tril_kernel[grid](
        A, out, M, N, diagonal, BLOCK_M=32, BLOCK_N=32
    )

    return out
