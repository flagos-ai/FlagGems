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
    pid_m = tle.program_id(0)
    pid_batch = tle.program_id(1)
    X += pid_batch * M * N
    Y += pid_batch * M * N

    row = pid_m * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
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
@triton.autotune(configs=runtime.get_tuned_config("tril"), key=["M", "N_EFF"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_copy_lower_kernel(
    X,
    Y,
    M,
    N,
    N_EFF,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_batch = tle.program_id(1)
    X += pid_batch * M * N
    Y += pid_batch * M * N

    row = pid_m * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N
    Y += row * N

    for n_offset in range(0, N_EFF, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        lower_mask = (cols <= row + diagonal) & m_mask & (cols < N_EFF)
        x = tl.load(X + cols, lower_mask, other=0.0)
        tl.store(Y + cols, x, mask=lower_mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("tril"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_zero_upper_kernel(
    X,
    M,
    N,
    diagonal,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_batch = tle.program_id(1)
    X += pid_batch * M * N

    row = pid_m * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    X += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        upper_mask = (cols > row + diagonal) & m_mask & n_mask
        tl.store(X + cols, 0.0, mask=upper_mask)


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


def tril(A, diagonal=0):
    logger.debug("GEMS TRIL")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    can_use_directly, A_input = _check_batch_contiguous(A, allow_zero_stride=False)

    M, N = A_input.shape[-2:]
    batch = A_input.numel() // (M * N) if A_input.dim() > 2 else 1

    if diagonal >= N - 1:
        return A_input.clone()

    N_eff = min(N, max(0, M + diagonal))

    if N_eff <= 0:
        return torch.zeros(A.shape, dtype=A.dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]), batch)

    total_bytes = A_input.numel() * A_input.element_size()
    if N_eff < N and total_bytes >= 1024 * 1024:
        out = torch.zeros(A.shape, dtype=A.dtype, device=A.device)
        with torch_device_fn.device(A_input.device):
            tril_copy_lower_kernel[grid](A_input, out, M, N, N_eff, diagonal)
    else:
        out = torch.empty(
            A.shape,
            dtype=A.dtype,
            device=A.device,
            memory_format=torch.contiguous_format,
        )
        with torch_device_fn.device(A_input.device):
            tril_kernel[grid](A_input, out, M, N, diagonal)

    return out


def tril_(A, diagonal=0):
    logger.debug("GEMS TRIL_ (inplace)")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)
    batch = A.numel() // (M * N) if A.dim() > 2 else 1

    rows_to_process = min(M, max(0, N - 1 - diagonal))

    if rows_to_process <= 0:
        return A

    if not can_use_directly:
        logger.debug(
            "Input tensor does not satisfy contiguity requirements, "
            "using temporary tensor for computation"
        )

        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]), batch)

        with torch_device_fn.device(A.device):
            tril_kernel[grid](A_to_use, result_temp, M, N, diagonal)

        A.copy_(result_temp)
    else:
        grid = lambda meta: (triton.cdiv(rows_to_process, meta["M_BLOCK_SIZE"]), batch)
        with torch_device_fn.device(A.device):
            if A.element_size() >= 4 or batch > 1:
                tril_zero_upper_kernel[grid](A, M, N, diagonal)
            else:
                tril_kernel[grid](A, A, M, N, diagonal)

    return A


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(input)
    assert out.shape == input.shape, "Input and output must have the same shape"
    assert out.dtype == input.dtype, "Input and output must have the same dtype"
    result = tril(input, diagonal)
    out.copy_(result)
    return out
