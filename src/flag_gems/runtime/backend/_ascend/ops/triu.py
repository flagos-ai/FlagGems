import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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
    pid = tle.program_id(0)
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
@triton.autotune(
    configs=runtime.get_tuned_config("triu_batch"),
    key=["batch", "MN", "N", "diagonal"],
)
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
):
    batch_id = tle.program_id(0)
    mn_id = tle.program_id(1)
    batch_workers = tle.num_programs(0)

    total_batch_workloads = tl.cdiv(batch, BATCH_BLOCK_SIZE)
    batch_workloads = 1
    while batch_workloads < tl.cdiv(batch, total_batch_workloads):
        batch_workloads *= 2

    for w in range(batch_workloads):
        batch_work_id = batch_id + w * batch_workers
        row = batch_work_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
        batch_mask = row < batch
        NX = X + row * MN
        NY = Y + row * MN

        cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
        mn_mask = cols < MN
        mask = batch_mask and mn_mask
        x = tl.load(NX + cols, mask, other=0.0)
        m = cols // N
        n = cols % N
        y = tl.where(m + diagonal <= n, x, 0.0)
        tl.store(NY + cols, y, mask=mask)


INT32_MAX = torch.iinfo(torch.int32).max


def triu(A, diagonal=0):
    logger.debug("GEMS TRIU")
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    with torch_device_fn.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A, out, M, N, diagonal)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)

            def grid(meta):
                axis0 = triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"])
                axis1 = triton.cdiv(M * N, meta["MN_BLOCK_SIZE"])
                while axis0 * axis1 >= 65536:
                    axis0 = axis0 // 2
                return (
                    axis0,
                    axis1,
                )

            triu_batch_kernel[grid](
                B,
                out,
                batch,
                M * N,
                N,
                diagonal,
            )
            out = out.view(A.shape)
    return out
