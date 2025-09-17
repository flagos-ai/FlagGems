import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import can_use_int32_index

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


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
    NEED_LOOP: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    num_jobs = tl.num_programs(0)
    m_block_step = M_BLOCK_SIZE * num_jobs

    for m_offset in range(pid * M_BLOCK_SIZE, M, m_block_step):
        if NEED_LOOP:
            row = m_offset + tl.arange(0, M_BLOCK_SIZE)[:, None]
            m_mask = row < M
            PX = X + row * N
            PY = Y + row * N
            for n_offset in range(0, N, N_BLOCK_SIZE):
                cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
                n_mask = cols < N
                mask = m_mask and n_mask

                x = tl.load(PX + cols, mask, other=0.0)
                y = tl.where(row + diagonal <= cols, x, 0.0)
                tl.store(PY + cols, y, mask=mask)
        else:
            write = tl.empty([M_BLOCK_SIZE, N_BLOCK_SIZE], X.dtype.element_ty)
            cols = tl.arange(0, N_BLOCK_SIZE)
            repeat_num = tl.minimum(M_BLOCK_SIZE, M - m_offset)
            for i in tl.range(repeat_num, num_stages=0):
                cur_row = m_offset + i
                PX = X + cur_row * N
                rmask = cols >= cur_row + diagonal
                write[i, :] = tl.load(PX + cols, mask=rmask, other=0.0)

            row = m_offset + tl.arange(0, M_BLOCK_SIZE)
            offset = cols[None, :] + row[:, None] * N
            n_mask = row[:, None] < M
            tl.store(Y + offset, write, mask=n_mask)


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
    logger.debug("GEMS_CAMBRICON TRIU")
    A = A.contiguous()
    out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    use_int64_index = not can_use_int32_index(A)
    M, N = A.shape[-2:]
    with torch_device_fn.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (
                min(triton.cdiv(M, meta["M_BLOCK_SIZE"]), TOTAL_CORE_NUM),
            )
            # A large value for n_block_size can lead to insufficient MLU resources,
            # causing the compilation to fail. Therefore, a conservative upper limit of 8192
            # is currently set, but the actual maximum achievable value should be confirmed
            # based on real-world conditions.
            elements_bytes = torch.finfo(A.dtype).bits // 8
            n_block = min(256 * 1024 // elements_bytes, N)
            need_loop = n_block < N
            triu_kernel[grid](
                A,
                out,
                M,
                N,
                diagonal,
                N_BLOCK_SIZE=n_block,
                NEED_LOOP=need_loop,
                INT64_INDEX=use_int64_index,
            )
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
