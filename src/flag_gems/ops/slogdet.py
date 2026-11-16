# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Matrices up to this size are factored entirely in registers as a single
# (BLOCK, BLOCK) tile. Above it the tiled path walks the matrix in memory, so
# the block size no longer scales with the matrix size (a 1025x1025 input would
# otherwise round up to a 2048x2048 tile and exceed Triton's tensor limits).
_SLOGDET_BLOCK_MAX = 64
_SLOGDET_TILE = 32


@libentry()
@triton.jit
def _slogdet_register_kernel(
    A,
    sign_out,
    logabsdet_out,
    N,
    BLOCK_N: tl.constexpr,
):
    """LU with partial pivoting for one small matrix, held in registers."""
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)

    offsets = pid * N * N + rows[:, None] * N + cols[None, :]
    load_mask = (rows[:, None] < N) & (cols[None, :] < N)
    work = tl.load(A + offsets, mask=load_mask, other=0.0)

    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        col_k = tl.sum(tl.where(cols[None, :] == k, work, 0.0), axis=1)
        abs_col = tl.abs(col_k)
        abs_col = tl.where((rows < k) | (rows >= N), -1.0, abs_col)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)

        row_k = tl.sum(tl.where(rows[:, None] == k, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where(rows[:, None] == k, row_p[None, :], work)
        work = tl.where(rows[:, None] == pivot_row, row_k[None, :], work)
        swap_count = tl.where(pivot_row != k, swap_count + 1, swap_count)

        col_k = tl.sum(tl.where(cols[None, :] == k, work, 0.0), axis=1)
        pivot = tl.sum(tl.where(rows == k, col_k, 0.0), axis=0)

        # A zero pivot leaves the trailing submatrix untouched: the matrix is
        # exactly singular and the zero diagonal is detected after the loop.
        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)
        multipliers = tl.where(rows > k, col_k / safe_pivot, 0.0)
        u_row = row_p
        update_mask = (rows[:, None] > k) & (cols[None, :] > k)
        work = tl.where(update_mask, work - multipliers[:, None] * u_row[None, :], work)

    diag = tl.sum(tl.where(rows[:, None] == cols[None, :], work, 0.0), axis=0)
    diag = tl.where(cols < N, diag, 1.0)

    # Only an exactly zero diagonal entry means a singular matrix. Comparing
    # against a fixed epsilon would report diag([1e-11, 1]) as singular, while
    # ATen returns a finite logabsdet for it.
    any_zero = tl.max(tl.where(diag == 0.0, 1, 0), axis=0)
    safe_diag = tl.where(diag == 0.0, 1.0, diag)

    logabsdet = tl.sum(tl.log(tl.abs(safe_diag)), axis=0)
    neg_count = tl.sum(tl.where(safe_diag < 0.0, 1, 0), axis=0)
    nan_count = tl.sum(tl.where(safe_diag != safe_diag, 1, 0), axis=0)

    sign = tl.where((neg_count + swap_count) % 2 == 0, 1.0, -1.0)
    # NaN anywhere on the diagonal: ATen reports sign 0 and a NaN logabsdet.
    sign = tl.where(nan_count > 0, 0.0, sign)
    sign = tl.where(any_zero > 0, 0.0, sign)
    logabsdet = tl.where(any_zero > 0, -float("inf"), logabsdet)

    tl.store(sign_out + pid, sign.to(sign_out.dtype.element_ty))
    tl.store(logabsdet_out + pid, logabsdet.to(logabsdet_out.dtype.element_ty))


@libentry()
@triton.jit
def _slogdet_blocked_kernel(
    A,
    sign_out,
    logabsdet_out,
    N,
    BLOCK: tl.constexpr,
):
    """LU with partial pivoting for one large matrix, tiled through memory.

    ``A`` is a scratch copy and is overwritten with the factorization.
    """
    pid = tle.program_id(0)
    base = pid * N * N
    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        best_val = tl.full((), -1.0, dtype=A.dtype.element_ty)
        best_row = tl.full((), k, dtype=tl.int32)
        for i0 in range(k, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            col = tl.load(A + base + rows * N + k, mask=rows < N, other=0.0)
            abs_col = tl.where((rows >= k) & (rows < N), tl.abs(col), -1.0)
            tile_max = tl.max(abs_col, axis=0)
            tile_row = tl.min(tl.where(abs_col == tile_max, rows, N), axis=0)
            is_better = tile_max > best_val
            best_row = tl.where(is_better, tile_row, best_row)
            best_val = tl.where(is_better, tile_max, best_val)

        for j0 in range(0, N, BLOCK):
            cols = j0 + tl.arange(0, BLOCK)
            cmask = cols < N
            row_k = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
            row_p = tl.load(A + base + best_row * N + cols, mask=cmask, other=0.0)
            tl.store(A + base + k * N + cols, row_p, mask=cmask)
            tl.store(A + base + best_row * N + cols, row_k, mask=cmask)
        swap_count = tl.where(best_row != k, swap_count + 1, swap_count)

        tl.debug_barrier()

        pivot = tl.load(A + base + k * N + k)
        safe_pivot = tl.where(pivot == 0.0, 1.0, pivot)

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            tl.store(A + base + rows * N + k, col / safe_pivot, mask=rmask)

        tl.debug_barrier()

        for i0 in range(k + 1, N, BLOCK):
            rows = i0 + tl.arange(0, BLOCK)
            rmask = rows < N
            l_col = tl.load(A + base + rows * N + k, mask=rmask, other=0.0)
            for j0 in range(k + 1, N, BLOCK):
                cols = j0 + tl.arange(0, BLOCK)
                cmask = cols < N
                u_row = tl.load(A + base + k * N + cols, mask=cmask, other=0.0)
                tmask = rmask[:, None] & cmask[None, :]
                tile = tl.load(
                    A + base + rows[:, None] * N + cols[None, :], mask=tmask, other=0.0
                )
                tile = tile - l_col[:, None] * u_row[None, :]
                tl.store(A + base + rows[:, None] * N + cols[None, :], tile, mask=tmask)

        tl.debug_barrier()

    logabsdet = (
        tl.zeros((), dtype=tl.float64)
        if A.dtype.element_ty == tl.float64
        else tl.zeros((), dtype=tl.float32)
    )
    neg_count = tl.zeros((), dtype=tl.int32)
    nan_count = tl.zeros((), dtype=tl.int32)
    zero_count = tl.zeros((), dtype=tl.int32)

    for i0 in range(0, N, BLOCK):
        d = i0 + tl.arange(0, BLOCK)
        dmask = d < N
        diag = tl.load(A + base + d * N + d, mask=dmask, other=1.0)
        zero_count += tl.sum(tl.where(dmask & (diag == 0.0), 1, 0), axis=0)
        nan_count += tl.sum(tl.where(dmask & (diag != diag), 1, 0), axis=0)
        neg_count += tl.sum(tl.where(dmask & (diag < 0.0), 1, 0), axis=0)
        safe = tl.where(dmask & (diag != 0.0), diag, 1.0)
        logabsdet += tl.sum(tl.log(tl.abs(safe)), axis=0)

    sign = tl.where((neg_count + swap_count) % 2 == 0, 1.0, -1.0)
    sign = tl.where(nan_count > 0, 0.0, sign)
    sign = tl.where(zero_count > 0, 0.0, sign)
    logabsdet = tl.where(zero_count > 0, -float("inf"), logabsdet)

    tl.store(sign_out + pid, sign.to(sign_out.dtype.element_ty))
    tl.store(logabsdet_out + pid, logabsdet.to(logabsdet_out.dtype.element_ty))


@libentry()
@triton.jit
def _slogdet_complex_kernel(
    A,
    sign_re_out,
    sign_im_out,
    logabsdet_out,
    N,
    BLOCK_N: tl.constexpr,
):
    """LU with partial pivoting for a complex matrix held in registers.

    ``A`` points at the interleaved real view of the complex input, so the real
    part of element (i, j) lives at ``2 * (i * N + j)`` and the imaginary part
    one element later.
    """
    pid = tle.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)

    flat = pid * N * N + rows[:, None] * N + cols[None, :]
    load_mask = (rows[:, None] < N) & (cols[None, :] < N)
    re = tl.load(A + 2 * flat, mask=load_mask, other=0.0)
    im = tl.load(A + 2 * flat + 1, mask=load_mask, other=0.0)

    swap_count = tl.zeros((), dtype=tl.int32)

    for k in range(N):
        col_re = tl.sum(tl.where(cols[None, :] == k, re, 0.0), axis=1)
        col_im = tl.sum(tl.where(cols[None, :] == k, im, 0.0), axis=1)
        # Pivot on the modulus, which is what LAPACK's complex pivoting uses.
        abs_col = tl.sqrt(col_re * col_re + col_im * col_im)
        abs_col = tl.where((rows < k) | (rows >= N), -1.0, abs_col)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)

        row_k_re = tl.sum(tl.where(rows[:, None] == k, re, 0.0), axis=0)
        row_k_im = tl.sum(tl.where(rows[:, None] == k, im, 0.0), axis=0)
        row_p_re = tl.sum(tl.where(rows[:, None] == pivot_row, re, 0.0), axis=0)
        row_p_im = tl.sum(tl.where(rows[:, None] == pivot_row, im, 0.0), axis=0)
        re = tl.where(rows[:, None] == k, row_p_re[None, :], re)
        im = tl.where(rows[:, None] == k, row_p_im[None, :], im)
        re = tl.where(rows[:, None] == pivot_row, row_k_re[None, :], re)
        im = tl.where(rows[:, None] == pivot_row, row_k_im[None, :], im)
        swap_count = tl.where(pivot_row != k, swap_count + 1, swap_count)

        col_re = tl.sum(tl.where(cols[None, :] == k, re, 0.0), axis=1)
        col_im = tl.sum(tl.where(cols[None, :] == k, im, 0.0), axis=1)
        piv_re = tl.sum(tl.where(rows == k, col_re, 0.0), axis=0)
        piv_im = tl.sum(tl.where(rows == k, col_im, 0.0), axis=0)

        # Complex division by the pivot: z / w = z * conj(w) / |w|^2.
        denom = piv_re * piv_re + piv_im * piv_im
        safe_denom = tl.where(denom == 0.0, 1.0, denom)
        mult_re = tl.where(
            rows > k, (col_re * piv_re + col_im * piv_im) / safe_denom, 0.0
        )
        mult_im = tl.where(
            rows > k, (col_im * piv_re - col_re * piv_im) / safe_denom, 0.0
        )

        update_mask = (rows[:, None] > k) & (cols[None, :] > k)
        u_re = row_p_re
        u_im = row_p_im
        prod_re = mult_re[:, None] * u_re[None, :] - mult_im[:, None] * u_im[None, :]
        prod_im = mult_re[:, None] * u_im[None, :] + mult_im[:, None] * u_re[None, :]
        re = tl.where(update_mask, re - prod_re, re)
        im = tl.where(update_mask, im - prod_im, im)

    diag_re = tl.sum(tl.where(rows[:, None] == cols[None, :], re, 0.0), axis=0)
    diag_im = tl.sum(tl.where(rows[:, None] == cols[None, :], im, 0.0), axis=0)
    diag_re = tl.where(cols < N, diag_re, 1.0)
    diag_im = tl.where(cols < N, diag_im, 0.0)

    modulus = tl.sqrt(diag_re * diag_re + diag_im * diag_im)
    any_zero = tl.max(tl.where(modulus == 0.0, 1, 0), axis=0)
    nan_count = tl.sum(tl.where(modulus != modulus, 1, 0), axis=0)
    safe_mod = tl.where(modulus == 0.0, 1.0, modulus)

    logabsdet = tl.sum(tl.log(safe_mod), axis=0)

    # sign = prod(u_ii / |u_ii|), accumulated as a running complex product, then
    # flipped once per row swap.
    acc_re = tl.full((), 1.0, dtype=diag_re.dtype)
    acc_im = tl.full((), 0.0, dtype=diag_re.dtype)
    for i in range(N):
        z_re = tl.sum(tl.where(cols == i, diag_re / safe_mod, 0.0), axis=0)
        z_im = tl.sum(tl.where(cols == i, diag_im / safe_mod, 0.0), axis=0)
        new_re = acc_re * z_re - acc_im * z_im
        new_im = acc_re * z_im + acc_im * z_re
        acc_re = new_re
        acc_im = new_im

    parity = tl.where(swap_count % 2 == 0, 1.0, -1.0)
    acc_re = acc_re * parity
    acc_im = acc_im * parity

    is_bad = (any_zero > 0) | (nan_count > 0)
    acc_re = tl.where(is_bad, 0.0, acc_re)
    acc_im = tl.where(is_bad, 0.0, acc_im)
    logabsdet = tl.where(any_zero > 0, -float("inf"), logabsdet)

    tl.store(sign_re_out + pid, acc_re.to(sign_re_out.dtype.element_ty))
    tl.store(sign_im_out + pid, acc_im.to(sign_im_out.dtype.element_ty))
    tl.store(logabsdet_out + pid, logabsdet.to(logabsdet_out.dtype.element_ty))


def _real_dtype_of(dtype):
    """The dtype ATen uses for logabsdet: real for real input, the
    corresponding real type for complex input."""
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def _check_input(A):
    """Reject the same inputs ATen rejects, with the same messages."""
    if A.dim() < 2:
        raise RuntimeError(
            "linalg.slogdet: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "linalg.slogdet: A must be batches of square matrices, "
            f"but they are {A.shape[-2]} by {A.shape[-1]} matrices"
        )
    if A.dtype in (torch.float16, torch.bfloat16):
        name = "Half" if A.dtype == torch.float16 else "BFloat16"
        raise RuntimeError(
            f"linalg.slogdet: Low precision dtypes not supported. Got {name}"
        )
    if A.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise RuntimeError(
            "linalg.slogdet: Expected a floating point or complex tensor as input. "
            f"Got {A.dtype}"
        )


def _slogdet_forward(A):
    _check_input(A)

    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    real_dtype = _real_dtype_of(A.dtype)

    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # An empty batch has nothing to compute; a 0x0 matrix has an empty product,
    # so ATen returns sign 1 and logabsdet 0 for it.
    if batch_size == 0:
        return (
            torch.empty(batch_shape, dtype=A.dtype, device=A.device),
            torch.empty(batch_shape, dtype=real_dtype, device=A.device),
        )
    if n == 0:
        return (
            torch.ones(batch_shape, dtype=A.dtype, device=A.device),
            torch.zeros(batch_shape, dtype=real_dtype, device=A.device),
        )

    logabsdet = torch.empty(batch_shape, dtype=real_dtype, device=A.device)

    # The factorization runs in place, so work on a contiguous copy: a
    # non-contiguous input (a transpose or a strided view) would otherwise be
    # read with the wrong offsets and the caller's tensor would be mutated.
    A_work = A.contiguous().clone()

    if A.dtype in (torch.complex64, torch.complex128):
        sign_re = torch.empty(batch_shape, dtype=real_dtype, device=A.device)
        sign_im = torch.empty(batch_shape, dtype=real_dtype, device=A.device)
        if n > _SLOGDET_BLOCK_MAX:
            raise NotImplementedError(
                "FlagGems slogdet: complex matrices larger than "
                f"{_SLOGDET_BLOCK_MAX}x{_SLOGDET_BLOCK_MAX} are not supported yet, got {n}x{n}"
            )
        block_n = max(16, triton.next_power_of_2(n))
        with torch_device_fn.device(A.device):
            _slogdet_complex_kernel[(batch_size,)](
                torch.view_as_real(A_work),
                sign_re,
                sign_im,
                logabsdet,
                n,
                BLOCK_N=block_n,
                num_warps=4,
            )
        sign = torch.complex(sign_re, sign_im)
        return sign, logabsdet

    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    with torch_device_fn.device(A.device):
        if n <= _SLOGDET_BLOCK_MAX:
            block_n = max(16, triton.next_power_of_2(n))
            _slogdet_register_kernel[(batch_size,)](
                A_work, sign, logabsdet, n, BLOCK_N=block_n, num_warps=4
            )
        else:
            _slogdet_blocked_kernel[(batch_size,)](
                A_work, sign, logabsdet, n, BLOCK=_SLOGDET_TILE, num_warps=4
            )

    return sign, logabsdet


class _Slogdet(torch.autograd.Function):
    """Adds the ATen gradient so forward/backward parity can be checked.

    ``d logabsdet / dA = inv(A)^H``; ``sign`` carries no gradient.
    """

    @staticmethod
    def forward(ctx, A):
        sign, logabsdet = _slogdet_forward(A)
        ctx.save_for_backward(A)
        return sign, logabsdet

    @staticmethod
    def backward(ctx, grad_sign, grad_logabsdet):
        (A,) = ctx.saved_tensors
        if grad_logabsdet is None:
            return None
        inv_h = torch.linalg.inv(A).mH
        return grad_logabsdet[..., None, None] * inv_h


def slogdet(A):
    """
    Compute the sign and natural logarithm of the absolute value of the
    determinant of a square matrix (or a batch of square matrices).

    This is the FlagGems implementation of ``aten::slogdet``, the public alias
    for ``torch.linalg.slogdet``.

    Args:
        A: tensor of shape ``(*, n, n)`` where ``*`` is zero or more batch
            dimensions. float32, float64, complex64 and complex128 are accepted.

    Returns:
        A tuple ``(sign, logabsdet)``. ``sign`` has the dtype of ``A`` (a unit
        modulus complex number for complex input); ``logabsdet`` is real. For an
        exactly singular matrix, ``sign = 0`` and ``logabsdet = -inf``.
    """
    logger.debug("GEMS SLOGDET")
    if torch.is_grad_enabled() and A.requires_grad:
        return _Slogdet.apply(A)
    return _slogdet_forward(A)
