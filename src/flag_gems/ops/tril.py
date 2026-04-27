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
@triton.jit(do_not_specialize=["diag"])
def tril_kernel(
    X,
    Y,
    M,
    N,
    B,
    diag,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_b = tle.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < M) & (offs_n < N)

    keep = offs_n <= (offs_m + diag)
    load_mask = mask & keep

    base = pid_b * M * N
    idxs = base + offs_m * N + offs_n

    x = tl.load(X + idxs, mask=load_mask, other=0.0)
    tl.store(Y + idxs, x, mask=mask)


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

    out = torch.empty(
        A.shape, dtype=A.dtype, device=A.device, memory_format=torch.contiguous_format
    )

    M, N = A_input.shape[-2:]
    B = A_input.numel() // (M * N)

    with torch_device_fn.device(A_input.device):
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
            B,
        )
        tril_kernel[grid](A_input, out, M, N, B, diagonal)

    return out


def tril_(A, diagonal=0):
    logger.debug("GEMS TRIL_ (inplace)")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]
    B = A.numel() // (M * N)

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)

    if not can_use_directly:
        logger.debug(
            "Input tensor does not satisfy contiguity requirements, "
            "using temporary tensor for computation"
        )

        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        with torch_device_fn.device(A.device):
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
                B,
            )
            tril_kernel[grid](A_to_use, result_temp, M, N, B, diagonal)

        A.copy_(result_temp)
    else:
        with torch_device_fn.device(A.device):
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
                B,
            )
            tril_kernel[grid](A, A, M, N, B, diagonal)

    return A


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(input)
    assert out.shape == input.shape, "Input and output must have the same shape"
    assert out.dtype == input.dtype, "Input and output must have the same dtype"
    result = tril(input, diagonal)
    out.copy_(result)
    return out
