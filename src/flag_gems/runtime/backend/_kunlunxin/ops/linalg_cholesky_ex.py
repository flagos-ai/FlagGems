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

"""``aten::linalg_cholesky_ex`` for Kunlunxin XPU3.

The generic register-resident kernel (``flag_gems/ops/linalg_cholesky_ex.py``)
does not lower on this backend: ``tl.trans`` inside the tile and the
``[BLOCK, BLOCK]`` rank-1 update trip ``TritonXPULegalize``
(``'arith.addf' op requires the same type for all operands``, then
``out of resource: uni_sram``).  For n = 64 the static ``tl.static_range``
unroll additionally blows the compile time up past any usable budget.

This override reuses the row-serial structure that the backend already ships
for ``aten::cholesky`` (``_kunlunxin/ops/cholesky.py``): one program per matrix
and **one launch per row** ``i``, lanes spanning the columns.  A launch only
reads rows produced by *previous* launches, so every write -> read dependency
crosses a kernel boundary, which this backend orders (reading a row back
inside the same program is unordered and was measured to produce garbage).

Two differences from ``cholesky.py``:

* ``info`` is part of the contract, so each launch folds its pivot check into
  ``Info[b]`` (first non-positive pivot wins, 1-based) instead of raising.
* the input triangle to read follows the ``upper`` flag
  (``upper=False`` -> lower triangle, matching ATen), so a non-symmetric
  input is handled with ATen semantics.  The factor is always built as the
  upper factor ``U`` and the lower factor is returned as the ``mT`` view.

Order is capped at 64 by the generic implementation's contract (the test
suite asserts ``NotImplementedError`` above that); ``CBLK = next_pow2(n)`` is
at most 64, the widest 1-D tile this backend lowers for this loop shape.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_N = 64
# Widest 1-D tile the XPU3 legaliser lowers for this loop shape
# (see kernels/cholesky-kunlunxin.md in the KernelOpt wiki).
_ROW_CHUNK = 64


@triton.jit
def _lce_row_kernel(
    A_ptr,
    U_ptr,
    Info_ptr,
    N,
    batch_stride,
    row_stride,
    col_stride,
    i,
    LOWER: tl.constexpr,
    CBLK: tl.constexpr,
):
    """Upper-factor Cholesky of row ``i`` for every matrix; one launch per row.

    ``LOWER`` selects which triangle of ``A`` supplies row ``i`` of the
    Hermitian work matrix:

    * ``LOWER=False`` (``upper=True``): ``A[i, :]`` -- a contiguous row load;
    * ``LOWER=True`` (``upper=False``): ``A[:, i]`` -- a strided column load.

    ``Info_ptr[b]`` receives ``i + 1`` for the first row whose pivot
    ``A[i,i] - sum_k U[k,i]^2`` is not strictly positive (0 while the
    factorisation is still valid).  Only ordered comparisons are used (never a
    NaN ``x != x``, which this backend cannot select), and ``sqrt`` is guarded
    so a failed pivot cannot emit NaN/inf into the buffer.
    """
    b = tl.program_id(0)
    base = b * batch_stride
    lanes = tl.arange(0, CBLK)
    jmask = lanes < N
    dt = A_ptr.dtype.element_ty

    # Column i of the already-computed rows feeds both the pivot norm and the
    # rank-1 trailing update; fold them into one loop to halve the scalar loads.
    sq = tl.zeros((), dtype=dt)
    acc = tl.zeros((CBLK,), dtype=dt)
    for k in range(i):
        fk = tl.load(U_ptr + base + k * row_stride + i * col_stride)
        pk = tl.load(
            U_ptr + base + k * row_stride + lanes * col_stride,
            mask=jmask,
            other=0.0,
        )
        sq += fk * fk
        acc += fk * pk

    diag = tl.load(A_ptr + base + i * row_stride + i * col_stride)
    arg = diag - sq
    ok = arg > 0.0
    d = tl.sqrt(tl.where(ok, arg, 1.0))
    inv = 1.0 / d

    if LOWER:
        a_row = tl.load(
            A_ptr + base + lanes * row_stride + i * col_stride,
            mask=jmask,
            other=0.0,
        )
    else:
        a_row = tl.load(
            A_ptr + base + i * row_stride + lanes * col_stride,
            mask=jmask,
            other=0.0,
        )

    val = tl.where(lanes == i, d, tl.where(lanes > i, (a_row - acc) * inv, 0.0))
    tl.store(U_ptr + base + i * row_stride + lanes * col_stride, val, mask=jmask)

    prev = tl.load(Info_ptr + b)
    tl.store(
        Info_ptr + b,
        tl.where(ok, prev, tl.where(prev == 0, i + 1, prev)),
    )


def linalg_cholesky_ex(self, upper=False, check_errors=False):
    """Cholesky decomposition returning ``(factor, info)``.

    Mirrors ``aten::linalg_cholesky_ex``: ``upper`` picks the returned
    triangle *and* the input triangle that is read; ``info[b]`` is 0 on
    success or the 1-based order of the first non-positive pivot.  A non-zero
    ``info`` raises ``torch.linalg.LinAlgError`` only when ``check_errors`` is
    set.
    """
    logger.debug("GEMS_KUNLUNXIN LINALG_CHOLESKY_EX")
    a = self
    if a.dim() < 2:
        raise RuntimeError(
            "linalg.cholesky: The input tensor A must have at least 2 dimensions."
        )
    if a.shape[-1] != a.shape[-2]:
        raise RuntimeError(
            "linalg.cholesky: A must be batches of square matrices, but they are "
            "%d by %d matrices" % (a.shape[-2], a.shape[-1])
        )
    if a.is_complex():
        raise NotImplementedError(
            "linalg_cholesky_ex: complex Hermitian input is not supported"
        )

    n = int(a.shape[-1])
    batch_shape = a.shape[:-2]

    # A zero-sized matrix / batch is vacuously factorised; ATen hands back
    # fresh empty outputs and a zero info instead of launching anything.
    if n == 0 or a.numel() == 0:
        return torch.return_types.linalg_cholesky_ex(
            (
                torch.empty_like(a),
                torch.zeros(batch_shape, dtype=torch.int32, device=a.device),
            )
        )

    if n > _MAX_N:
        raise NotImplementedError(
            "linalg_cholesky_ex kernel supports N <= %d" % _MAX_N
        )

    num = int(a.numel()) // (n * n)
    a2 = a.reshape(num, n, n)
    if not a2.is_contiguous():
        # Data staging only (vendor copy): the row kernel indexes the matrices
        # with their real strides, so it needs a dense input.
        a2 = a2.contiguous()

    # The kernel writes every element of the requested (upper) triangle and
    # stores exact zeros on the other side, so no initialisation is needed.
    out = torch.empty((num, n, n), dtype=a2.dtype, device=a2.device)
    # Read-modify-write accumulator: must start at 0 for every matrix.
    info = torch.zeros(num, dtype=torch.int32, device=a2.device)

    cblk = triton.next_power_of_2(n)
    lower_input = not bool(upper)
    for i in range(n):
        _lce_row_kernel[(num,)](
            a2,
            out,
            info,
            n,
            a2.stride(0),
            a2.stride(1),
            a2.stride(2),
            i,
            LOWER=lower_input,
            CBLK=cblk,
            num_warps=1,
            num_stages=1,
        )

    if check_errors:
        # Tiny D2H read of the failure flags (no numeric work on the host).
        first = 0
        for flag in info.tolist():
            if flag != 0:
                first = int(flag)
                break
        if first != 0:
            raise torch.linalg.LinAlgError(
                "linalg.cholesky_ex: The factorization could not be completed "
                "because the input is not positive-definite (the leading "
                "minor of order %d is not positive-definite)." % first
            )

    if upper:
        factor = out.reshape(a.shape)
    else:
        # U^T is the lower factor; keep the transpose as a view like ATen does.
        factor = out.mT.reshape(a.shape)

    return torch.return_types.linalg_cholesky_ex((factor, info.reshape(batch_shape)))


__all__ = ["linalg_cholesky_ex"]
