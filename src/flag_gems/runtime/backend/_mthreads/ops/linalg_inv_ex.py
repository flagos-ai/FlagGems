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

"""linalg_inv_ex: inverse of a square matrix (partial-pivot LU / Gauss-Jordan).

One Triton program owns one matrix. The program keeps the working matrix and the
inverse accumulator as two register-resident BLOCK x BLOCK tiles and performs
partially pivoted Gauss-Jordan elimination on the augmented system [A | I].
After N steps the left tile is the identity and the right tile is A^-1.

The pivot row is selected with LAPACK's rule (first maximum of |column|), and
``info`` follows the getrf convention (1-based index of the first exactly zero
pivot, 0 on success).
"""

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_inv_ex import linalg_inv_ex as default_linalg_inv_ex
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized.
_SUPPORTED_DTYPES = {torch.float32}

LinalgInvExResult = namedtuple("LinalgInvExResult", ["inverse", "info"])


@triton.jit
def _linalg_inv_ex_kernel(
    A,
    OUT,
    INFO,
    N,
    SMA,
    SROW,
    SCOL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    c = tl.arange(0, BLOCK)
    rows_ok = r < N
    mask = rows_ok[:, None] & (c[None, :] < N)

    a_off = pid * SMA + r[:, None] * SROW + c[None, :] * SCOL
    a = tl.load(A + a_off, mask=mask, other=0.0)

    inv = tl.where(r[:, None] == c[None, :], 1.0, 0.0).to(a.dtype)

    info_val = 0
    for k in range(0, N):
        # current pivot column (register tiles cannot be indexed dynamically)
        col = tl.sum(tl.where(c[None, :] == k, a, 0.0), axis=1)
        # pivot search restricted to rows k..N-1 (LAPACK: first maximum)
        abs_col = tl.where((r >= k) & rows_ok, tl.abs(col), -1.0)
        pv = tl.max(abs_col, axis=0)
        piv = tl.min(tl.where(abs_col == pv, r, BLOCK), axis=0)

        col_k = tl.sum(tl.where(r == k, col, 0.0), axis=0)
        # the signed pivot element (pv above is only the magnitude used to pick
        # the row); scaling by the magnitude would flip signs on negative pivots
        piv_val = tl.sum(tl.where(r == piv, col, 0.0), axis=0)
        row_k = tl.sum(tl.where(r[:, None] == k, a, 0.0), axis=0)
        row_p = tl.sum(tl.where(r[:, None] == piv, a, 0.0), axis=0)
        irow_k = tl.sum(tl.where(r[:, None] == k, inv, 0.0), axis=0)
        irow_p = tl.sum(tl.where(r[:, None] == piv, inv, 0.0), axis=0)

        # guard the division so a singular step stays finite; info records it
        safe = tl.where(piv_val == 0.0, 1.0, piv_val)
        # after swapping rows k and piv, column k holds col_p at row k and
        # col_k at row piv; every other row keeps its value
        f = tl.where((r != k) & rows_ok, tl.where(r == piv, col_k, col) / safe, 0.0)
        f_p = tl.where(piv == k, 0.0, col_k / safe)

        # row k becomes the (swapped-in) pivot row divided by the pivot, row piv
        # becomes the eliminated old row k, every other row takes the update
        a = tl.where(
            r[:, None] == k,
            (row_p / safe)[None, :],
            tl.where(
                (r[:, None] == piv) & (piv != k),
                (row_k - f_p * row_p)[None, :],
                a - f[:, None] * row_p[None, :],
            ),
        )
        inv = tl.where(
            r[:, None] == k,
            (irow_p / safe)[None, :],
            tl.where(
                (r[:, None] == piv) & (piv != k),
                (irow_k - f_p * irow_p)[None, :],
                inv - f[:, None] * irow_p[None, :],
            ),
        )
        info_val = tl.where((piv_val == 0.0) & (info_val == 0), k + 1, info_val)

    o_off = pid * (N * N) + r[:, None] * N + c[None, :]
    tl.store(OUT + o_off, inv, mask=mask)
    tl.store(INFO + pid, info_val)


_BLOCK = {i: max(4, 1 << (i - 1).bit_length()) for i in range(1, 257)}
_WARPS = {4: 1, 8: 1, 16: 1, 32: 4, 64: 4, 128: 8, 256: 8}


def _any_nonzero(value):
    """True when any info entry is non-zero; pure-Python walk of a nested list."""
    if isinstance(value, list):
        for item in value:
            if _any_nonzero(item):
                return True
        return False
    return value != 0


def _launch_params(n):
    block = _BLOCK.get(n)
    if block is None:
        block = max(4, triton.next_power_of_2(n))
    return block, _WARPS.get(block, 8)


def _use_triton_kernel(A):
    """Whether ``A`` can be handled by the MUSA kernel."""
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if A.is_complex() or A.dim() < 2:
        return False
    if A.shape[-1] != A.shape[-2]:
        return False
    return True


def linalg_inv_ex(A, *, check_errors=False):
    """Compute the inverse of a square matrix with partial-pivot Gauss-Jordan.

    Args:
        A: tensor of shape (*, n, n), float32 or float64.
        check_errors: raise torch.linalg.LinAlgError when the matrix is singular.

    Returns:
        LinalgInvExResult(inverse, info) with info an int32 tensor of shape A.shape[:-2].
    """
    logger.debug("GEMS_MTHREADS LINALG_INV_EX")

    if not _use_triton_kernel(A):
        return default_linalg_inv_ex(A, check_errors=check_errors)

    shape = A.shape
    n = shape[-1]
    batch = 1
    for s in shape[:-2]:
        batch *= s

    out = A.new_empty(shape)
    info = torch.empty(shape[:-2], dtype=torch.int32, device=A.device)

    if batch != 0 and n != 0:
        if A.is_contiguous():
            smat, srow, scol = n * n, n, 1
        else:
            nd = A.ndim
            srow = A.stride(nd - 2)
            scol = A.stride(nd - 1)
            if nd == 3:
                smat = A.stride(0)
            elif nd > 3:
                # no uniform matrix stride: materialize a dense copy
                A = A.contiguous()
                smat, srow, scol = n * n, n, 1
            else:
                smat = 0
        block, warps = _launch_params(n)
        with torch_device_fn.device(A.device):
            _linalg_inv_ex_kernel[(batch,)](
                A, out, info, n, smat, srow, scol, BLOCK=block, num_warps=warps
            )

    if check_errors and _any_nonzero(info.tolist()):
        raise torch.linalg.LinAlgError(
            "torch.linalg.inv_ex: The diagonal element of the LU decomposition is "
            "zero, the inversion could not be completed because the input matrix "
            "is singular."
        )

    return LinalgInvExResult(out, info)
