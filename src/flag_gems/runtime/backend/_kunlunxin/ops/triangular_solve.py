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

"""XPU override for ``torch.triangular_solve``.

The generic implementation needs ``tl.gather`` to pull one component out of a
live register tile (``_panel_solve_kernel`` / ``_inverse_solve_kernel``), which
this Triton backend cannot lower (``failed to legalize operation 'tt.gather'``).
The solve is therefore re-expressed as a column-resident substitution, the same
structure as the XPU ``linalg_lu_solve`` override:

* one program per ``(batch, RHS column)``;
* the whole RHS column lives in a single ``BLOCK_N`` register vector ``w``;
* step ``i`` extracts its scalar component with a one-hot weight
  ``tl.sum(w * oh, axis=0)`` and applies the rank-1 update in place, so no
  gather is ever needed.

``transpose`` needs no special handling: the shared driver passes ``A.mT`` and
flips ``upper``, and every address here is computed from the matrix shape, so
``tl.trans`` is never used.

Two measured XPU constraints shape the kernel:

* ``tl.load`` masks derived from a scalar computed *inside* the kernel are
  silently dropped (the full column is read), so no load mask is used: rows
  past ``N`` are clamped to row ``N - 1`` and discarded by the store mask,
  which only compares against the ``N`` argument and does work.
* ignored triangle entries may hold ``inf``/``nan`` (the caller is allowed to
  leave them poisoned), and ``0 * inf`` would poison the accumulator. They are
  zeroed in the contiguous working copy of ``A`` with a select (``torch.where``)
  before the launch, so the kernel can read the whole column.

Only the dense solver is replaced: shape/dtype/out/autograd validation, the
broadcast rules, the legacy column-major result layout and both public entry
points come from ``flag_gems.ops.triangular_solve``.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.triangular_solve import _triangular_solve
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.ops.linalg_solve_triangular import (
    linalg_solve_triangular,
)

logger = logging.getLogger(__name__)

# A whole RHS column stays in registers only below this block width; larger
# widths fail to compile with "out of resource: uni_sram" (measured at 512).
_MAX_COLUMN_BLOCK = 128


@triton.jit
def _trsm_column_kernel(
    A,
    B,
    X,
    N,
    K,
    BLOCK_N: tl.constexpr,
    UPPER: tl.constexpr,
    UNIT: tl.constexpr,
):
    # A, B are contiguous (batch, ., .) buffers and X is the legacy
    # column-major (k, n) buffer seen as (n, k):
    #   A[b, r, c] -> b * N * N + r * N + c
    #   B[b, r, c] -> b * N * K + r * K + c
    #   X[b, r, c] -> b * N * K + r + c * N
    pid = tl.program_id(0)
    batch = pid // K
    col = pid % K

    r = tl.arange(0, BLOCK_N)
    # Clamped row index: avoids a load mask (unreliable here) while staying
    # inside the allocation. Lanes at or past N never reach the result.
    rc = tl.minimum(r, N - 1)
    in_range = r < N
    a_base = A + batch * N * N
    w = tl.load(B + batch * N * K + rc * K + col)

    # Forward (lower) or backward (upper) substitution.
    for t in range(N):
        if UPPER:
            i = N - 1 - t
        else:
            i = t
        # One-hot weight selecting row i; built arithmetically because this
        # backend miscompiles comparisons kept live across the serial loop.
        df = (r - i).to(tl.float32)
        oh = tl.maximum(1.0 - tl.abs(df), 0.0)
        w_i = tl.sum(w * oh, axis=0)
        if not UNIT:
            w_i = w_i / tl.load(a_base + i * N + i)
        a_col = tl.load(a_base + rc * N + i)
        w = w - a_col * w_i
        w = w + oh * (w_i - w)

    tl.store(X + batch * N * K + r + col * N, w, mask=in_range)


_MASKS = {}


def _triangle_mask(n, upper, strict, device):
    """Cached bool mask that keeps only the triangle the solve reads."""
    key = (n, upper, strict, str(device))
    mask = _MASKS.get(key)
    if mask is None:
        index = torch.arange(n, device=device)
        if upper:
            mask = (
                index[:, None] < index[None, :]
                if strict
                else index[:, None] <= index[None, :]
            )
        else:
            mask = (
                index[:, None] > index[None, :]
                if strict
                else index[:, None] >= index[None, :]
            )
        _MASKS[key] = mask
    return mask


def _solve_dense(A, B, *, upper, unitriangular, out=None):
    n, k = B.shape[-2:]
    batch = tuple(B.shape[:-2])
    batches = math.prod(batch)
    # Legacy BLAS layout: the (k, n) buffer is filled and handed back as its
    # transpose, which is exactly the layout the shared driver expects. When the
    # driver hands over an output with that exact layout, write into it in
    # place; otherwise allocate it.
    result = (
        out
        if out is not None
        else torch.empty((*batch, k, n), dtype=B.dtype, device=B.device).mT
    )
    if B.numel():
        block_n = max(triton.next_power_of_2(n), 16)
        if block_n <= _MAX_COLUMN_BLOCK:
            # A unit-triangular solve never reads the diagonal, so a poisoned
            # diagonal must be dropped as well; otherwise only the ignored
            # (opposite) triangle is dropped.
            mask = _triangle_mask(n, bool(upper), bool(unitriangular), B.device)
            A_work = torch.where(
                mask,
                A.reshape(batches, n, n),
                torch.zeros((), dtype=A.dtype, device=A.device),
            )
            B_work = B.reshape(batches, n, k).contiguous()
            with torch_device_fn.device(B.device):
                _trsm_column_kernel[(batches * k,)](
                    A_work,
                    B_work,
                    result,
                    n,
                    k,
                    BLOCK_N=block_n,
                    UPPER=bool(upper),
                    UNIT=bool(unitriangular),
                    num_warps=1,
                    num_stages=1,
                )
        else:
            # A whole column no longer fits in the register/SRAM budget
            # ("out of resource: uni_sram" for BLOCK_N >= 512), so use this
            # backend's blocked XPU trsm, which streams the solved rows.
            solved = linalg_solve_triangular(
                A,
                B,
                upper=bool(upper),
                left=True,
                unitriangular=bool(unitriangular),
            )
            result.copy_(solved)
    return result


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS TRIANGULAR_SOLVE")
    return _triangular_solve(
        B, A, upper, transpose, unitriangular, _solve_dense, solver_out=True
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B, A, upper, transpose, unitriangular, _solve_dense, X, M, solver_out=True
    )
