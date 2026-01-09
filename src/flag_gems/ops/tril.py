import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("tril"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask & n_mask

        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(cols <= row + diagonal, x, 0.0)
        tl.store(Y + cols, y, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("tril_batch"),
    key=["batch", "MN", "N", "diagonal"],
)
@triton.jit(do_not_specialize=["diagonal"])
def tril_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tle.program_id(0)
    mn_id = tle.program_id(1)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask & mn_mask
    x = tl.load(X + cols, mask, other=0.0)

    # Compute 2D indices from flattened position
    m = cols // N  # row index
    n = cols % N  # col index

    # Lower triangular condition
    y = tl.where(n <= m + diagonal, x, 0.0)
    tl.store(Y + cols, y, mask=mask)


def tril(A, diagonal=0):
    logger.debug("GEMS TRIL")
    A = A.contiguous()
    out = torch.empty_like(A)

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]

    with torch_device_fn.device(A.device):
        if len(A.shape) == 2:
            # Optimized path for 2D matrices
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            tril_kernel[grid](A, out, M, N, diagonal)
        else:
            # Batched path for higher-dimensional tensors
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            tril_batch_kernel[grid](
                B,
                out,
                batch,
                M * N,
                N,
                diagonal,
            )
            out = out.view(A.shape)

    return out


def tril_(A, diagonal=0):
    logger.debug("GEMS TRIL_")
    result = tril(A, diagonal)
    A.copy_(result)
    return A


def tril_out(A, diagonal=0, out=None):
    logger.debug("GEMS TRIL_OUT")
    if out is None:
        return tril(A, diagonal)

    result = tril(A, diagonal)
    out.copy_(result)
    return out
