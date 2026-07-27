import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MAX_MATRIX_SIZE = 32


@libentry()
@triton.jit
def _slogdet_kernel(A_ptr, sign_ptr, logabsdet_ptr, n, MATRIX_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    row_indices = tl.arange(0, MATRIX_SIZE)
    col_indices = tl.arange(0, MATRIX_SIZE)
    rows = row_indices[:, None]
    cols = col_indices[None, :]
    valid = (rows < n) & (cols < n)
    matrix = tl.load(A_ptr + pid * n * n + rows * n + cols, mask=valid, other=0.0)

    # Keep the complete elimination state in SSA values. The XPU compiler does not
    # preserve dependencies between the scalar scattered loads/stores used by the
    # generic implementation, whereas this tile form lowers as dataflow operations.
    logabsdet = 0.0
    sign = 1.0
    singular = False
    for pivot_index in range(MATRIX_SIZE):
        if pivot_index < n:
            pivot_column = tl.sum(
                tl.where(cols == pivot_index, matrix, 0.0), axis=1
            ).to(tl.float32)
            pivot_column = tl.abs(pivot_column)
            candidate_rows = row_indices >= pivot_index
            pivot_abs = tl.max(tl.where(candidate_rows, pivot_column, -1.0), axis=0)
            pivot_row = tl.argmax(
                tl.where(candidate_rows, pivot_column, -1.0), axis=0
            )
            selected_row = tl.sum(
                tl.where(rows == pivot_row, matrix, 0.0), axis=0
            )
            current_row = tl.sum(
                tl.where(rows == pivot_index, matrix, 0.0), axis=0
            )
            matrix = tl.where(
                rows == pivot_index,
                selected_row[None, :],
                tl.where(rows == pivot_row, current_row[None, :], matrix),
            )
            current_row = selected_row

            pivot = tl.sum(
                tl.where(col_indices == pivot_index, current_row, 0.0), axis=0
            ).to(tl.float32)
            pivot_is_zero = pivot_abs < 1e-12
            singular |= pivot_is_zero
            pivot_safe = tl.where(pivot_is_zero, 1.0, pivot)
            sign *= tl.where(pivot_row != pivot_index, -1.0, 1.0)
            sign *= tl.where(pivot < 0.0, -1.0, 1.0)
            logabsdet += tl.log(tl.where(pivot_is_zero, 1.0, pivot_abs))

            factors = tl.sum(
                tl.where(cols == pivot_index, matrix, 0.0), axis=1
            ) / pivot_safe
            update_mask = (rows > pivot_index) & (cols > pivot_index)
            matrix = tl.where(
                update_mask,
                matrix - factors[:, None] * current_row[None, :],
                matrix,
            )

    tl.store(sign_ptr + pid, tl.where(singular, 0.0, sign))
    tl.store(logabsdet_ptr + pid, tl.where(singular, -float("inf"), logabsdet))


def linalg_slogdet(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_SLOGDET")
    if A.dtype != torch.float32:
        raise NotImplementedError(f"linalg_slogdet: unsupported dtype {A.dtype}")
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("linalg_slogdet: expected batches of square matrices")

    n = A.shape[-1]
    if n > _MAX_MATRIX_SIZE:
        raise NotImplementedError(
            f"linalg_slogdet: matrix size {n} exceeds {_MAX_MATRIX_SIZE}"
        )

    batch_shape = A.shape[:-2]
    batch_size = 1
    for dimension in batch_shape:
        batch_size *= dimension

    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    if batch_size == 0:
        return torch.zeros_like(sign), torch.full_like(logabsdet, float("-inf"))

    with torch_device_fn.device(A.device):
        _slogdet_kernel[(batch_size,)](
            A,
            sign.reshape(-1),
            logabsdet.reshape(-1),
            n,
            MATRIX_SIZE=_MAX_MATRIX_SIZE,
            num_warps=1,
        )
    return sign, logabsdet
