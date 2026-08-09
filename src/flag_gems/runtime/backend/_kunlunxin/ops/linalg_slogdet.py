import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .linalg_lu_factor import (
    _lu_scale_column_kernel,
    _lu_swap_rows_kernel,
    _lu_update_trailing_kernel,
)

logger = logging.getLogger(__name__)

_MAX_MATRIX_SIZE = 32
# Fixed elimination width (rows) for the pivoting chain.  The vendored
# pivot-search chain proved unreliable on XPU:
#  - non-power-of-two tail blocks and BLOCK_P == 1 read one lane past their
#    buffers and corrupt other batch elements;
#  - at batched grids the trailing-update kernel corrupts cells it never
#    addresses;
#  - tl.argmax returns the sentinel 0 on ~25% of the pivot entries (both for
#    real swaps and for no-ops), so swap parity cannot be recovered from it.
# This implementation therefore performs partial-pivot LU elimination of the
# real leading n x n block of a zero-padded (batch, 64, 64) buffer with its
# own pivot selection (a plain 1-D 64-lane tl.max followed by the minimum
# matching row via tl.min -- no tl.argmax, no masked reduce), reuses the
# swap/scale/update vector kernels (verified safe when launched with one
# batch element per call), and fuses sign/logabsdet with a fully unrolled
# scalar post kernel (no data-dependent addressing).  Padded lanes stay zero
# and never win the pivot search except on singular columns, whose result is
# overridden to (0, -inf) by the post kernel.
_PAD = 64


@triton.jit
def _pivot_col_kernel(LU, P, M, J: tl.constexpr):
    """Partial-pivot row for column J of a (1, 64, 64) padded matrix.

    best = max |LU[r][J]| over r in [J, M); the pivot row is the smallest r
    attaining best (LAPACK-style first strict maximum).  Pivot stored 1-based.
    An all-zero column (singular) yields sentinel row 64, recorded as J+1
    (no swap); the exact-zero diagonal is then flagged by the post kernel.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, 64)
    values = tl.load(LU + pid * M * M + rows * M + J)
    absv = tl.abs(values)
    best = tl.max(tl.where(rows >= J, absv, -1.0), axis=0)
    row = tl.min(tl.where((rows >= J) & (absv == best), rows, 64), axis=0)
    row = tl.where(row == 64, J, row)
    tl.store(P + pid * M + J, row + 1)


def _factor_single(A2d, n):
    """Partial-pivot LU of the real leading n x n block of a zero-padded
    (64, 64) buffer; single batch element per call (the verified-safe
    per-launch scope).  Returns (lu (64, 64), pivots (64,)); only the first
    n pivot entries are meaningful.

    The pivot buffer is placed 64 words behind the 64x64 LU region of one
    allocation: the trailing-update kernel's masked stores can land one word
    past its tile (observed zeroing the first pivot), so a 64-word guard gap
    between LU and the pivot buffer absorbs those stray writes.
    """
    m = _PAD
    storage = torch.zeros(1, m * m + 64 + m, device=A2d.device, dtype=torch.float32)
    lu = storage[:, : m * m].view(1, m, m)
    lu[0, :n, :n] = A2d
    pivots = storage[:, 4096 + 64 : 4096 + 64 + 64].view(torch.int32)
    pivots = pivots.view(1, m)

    with torch_device_fn.device(A2d.device):
        for j in range(n):
            _pivot_col_kernel[(1,)](lu, pivots, m, j, num_warps=4)
            _lu_swap_rows_kernel[(1,)](
                lu,
                pivots,
                m,
                m,
                m,
                j,
                BLOCKS=triton.cdiv(m, 64),
                BLOCK_N=64,
                num_warps=4,
            )
            if j + 1 < m:
                _lu_scale_column_kernel[(1,)](
                    lu,
                    m,
                    m,
                    j,
                    BLOCKS=triton.cdiv(m - j - 1, 64),
                    BLOCK_M=64,
                    num_warps=4,
                )
            if j + 1 < m:
                _lu_update_trailing_kernel[(1 * (m - j - 1) * triton.cdiv(m - j - 1, 128),)](
                    lu,
                    m,
                    m,
                    j,
                    ROWS=m - j - 1,
                    BLOCKS=triton.cdiv(m - j - 1, 128),
                    BLOCK_N=128,
                    num_warps=4,
                )
    return lu[0], pivots[0]


@libentry()
@triton.jit
def _slogdet_post_kernel(
    LU_ptr, pivots_ptr, sign_ptr, logabsdet_ptr, N: tl.constexpr, PAD: tl.constexpr
):
    """Fuse sign/logabsdet from the (padded) LU factorization.

    det(A) = (-1)^swaps * prod(diag(U)); pivots are 1-indexed LAPACK-style so
    a swap at step i is pivots[i] != i + 1. PAD is the padded row stride.
    All offsets are straight-line scalar loads (no data-dependent addressing)
    and the loop is a fully unrolled constant loop of length N, the pattern
    the XPU compiler lowers reliably (no 2-D tile reductions).
    """
    pid = tle.program_id(0).to(tl.int64)
    logabsdet = 0.0
    sign = 1.0
    zero_diag = 0
    base = LU_ptr + pid * PAD * PAD
    piv_base = pivots_ptr + pid * PAD
    for i in range(N):
        piv = tl.load(piv_base + i).to(tl.int32)
        sign = sign * tl.where(piv != i + 1, -1.0, 1.0)
        diag = tl.load(base + i * PAD + i).to(tl.float32)
        # A NaN diagonal can only arise from a zero pivot (0/0 scale on a
        # singular matrix); treat it like a zero pivot.
        is_zero = (diag == 0.0) | (diag != diag)
        zero_diag = zero_diag + is_zero.to(tl.int32)
        sign = sign * tl.where(diag < 0.0, -1.0, 1.0)
        logabsdet = logabsdet + tl.log(tl.abs(diag))

    singular = zero_diag > 0
    tl.store(sign_ptr + pid, tl.where(singular, 0.0, sign))
    tl.store(logabsdet_ptr + pid, tl.where(singular, float("-inf"), logabsdet))


def linalg_slogdet(A):
    logger.debug("GEMS_KUNLUNXIN LINALG_SLOGDET")
    if A.dtype != torch.float32:
        raise NotImplementedError(f"linalg_slogdet: unsupported dtype {A.dtype}")
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("linalg_slogdet: expected batches of square matrices")

    n = A.shape[-1]
    if n == 0 or n > _MAX_MATRIX_SIZE:
        raise NotImplementedError(
            f"linalg_slogdet: matrix size {n} out of supported range "
            f"(1..{_MAX_MATRIX_SIZE})"
        )

    batch_shape = A.shape[:-2]
    batch_size = 1
    for dimension in batch_shape:
        batch_size *= dimension

    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    if batch_size == 0:
        return torch.zeros_like(sign), torch.full_like(logabsdet, float("-inf"))

    if not A.is_contiguous():
        A = A.contiguous()
    A3 = A.reshape(batch_size, n, n)
    sign_flat = sign.reshape(-1)
    logabs_flat = logabsdet.reshape(-1)
    with torch_device_fn.device(A.device):
        for b in range(batch_size):
            lu, pivots = _factor_single(A3[b], n)
            _slogdet_post_kernel[(1,)](
                lu,
                pivots,
                sign_flat[b : b + 1],
                logabs_flat[b : b + 1],
                n,
                _PAD,
                num_warps=1,
            )
    return sign, logabsdet