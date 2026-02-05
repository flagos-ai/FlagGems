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


def _check_batch_contiguous(tensor, allow_zero_stride=True):
    if tensor.is_contiguous():
        return True, tensor

    dims = tensor.dim()

    if dims >= 2:
        n = tensor.size(-1)
        stride_row, stride_col = tensor.stride(-2), tensor.stride(-1)

        if not (stride_col == 1 and stride_row == n):
            return False, tensor.contiguous()

    if allow_zero_stride and dims <= 3:
        return True, tensor

    expected_stride = tensor.size(-1) * tensor.size(-2)
    for i in range(dims - 3, -1, -1):
        if (
            allow_zero_stride
            and i == 0
            and (tensor.stride(i) == 0 or tensor.size(i) == 1)
        ):
            continue

        if tensor.stride(i) != expected_stride:
            return False, tensor.contiguous()

        expected_stride *= tensor.size(i)

    return True, tensor


def triu(A, diagonal=0):
    logger.debug("GEMS TRIU")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    can_use_directly, A_input = _check_batch_contiguous(A, allow_zero_stride=False)

    out = torch.empty(
        A.shape, dtype=A.dtype, device=A.device, memory_format=torch.contiguous_format
    )

    M, N = A_input.shape[-2:]

    with torch_device_fn.device(A_input.device):
        if len(A_input.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            triu_kernel[grid](A_input, out, M, N, diagonal)
        else:
            batch = int(torch.numel(A_input) / M / N)
            B = A_input.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            triu_batch_kernel[grid](B, out, batch, M * N, N, diagonal)
            out = out.view(A.shape)

    return out


def triu_(A, diagonal=0):
    logger.debug("GEMS TRIU_ (inplace)")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)

    if not can_use_directly:
        logger.debug(
            "Input tensor does not satisfy contiguity requirements, "
            "using temporary tensor for computation"
        )

        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        with torch_device_fn.device(A.device):
            if len(A.shape) == 2:
                grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
                triu_kernel[grid](A_to_use, result_temp, M, N, diagonal)
            else:
                batch = int(torch.numel(A) / M / N)
                B = A_to_use.view(batch, -1)
                result_temp_flat = result_temp.view(batch, -1)
                grid = lambda meta: (
                    triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                    triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
                )
                triu_batch_kernel[grid](B, result_temp_flat, batch, M * N, N, diagonal)

        A.copy_(result_temp)
    else:
        with torch_device_fn.device(A.device):
            if len(A.shape) == 2:
                grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
                triu_kernel[grid](A, A, M, N, diagonal)
            else:
                batch = int(torch.numel(A) / M / N)
                B = A.view(batch, -1)
                grid = lambda meta: (
                    triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                    triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
                )
                triu_batch_kernel[grid](B, B, batch, M * N, N, diagonal)

    return A
