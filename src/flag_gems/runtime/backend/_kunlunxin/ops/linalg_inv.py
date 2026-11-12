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

"""Kunlunxin backend override for ``linalg_inv``.

The generic ``_inv_kernel`` keeps two full ``BLOCK_N``-by-``BLOCK_N`` register
tiles live across a data-dependent serial pivot loop.  The ``TritonXPULegalize``
MLIR pipeline crashes on exactly that combination on this backend (same class
of failure as ``linalg_eig``), so the generic entry point cannot compile here.

This override composes the two pieces that already compile on the backend:

* ``_linalg_lu_factor`` — the O(k) pivoted elimination already shipped as the
  ``linalg_lu_factor`` override (pure Triton, no fallback), returns the compact
  ``LU`` factors and 1-indexed pivots;
* ``_xpu_lu_solve_impl`` — the column-resident substitution kernel from the
  ``linalg_lu_solve`` override.

``A^-1`` is obtained by solving ``A X = I`` column by column, so no register
tile survives across the pivot loop.  Validation and empty handling mirror the
generic entry point so behaviour does not diverge.
"""

import logging

import torch

from flag_gems.runtime import torch_device_fn

from .linalg_lu_factor import _linalg_lu_factor
from .linalg_lu_solve import _xpu_lu_solve_impl

logger = logging.getLogger(__name__)


def linalg_inv(A):
    """Batched inverse via LU factorisation + column substitution."""
    logger.debug("GEMS_KUNLUNXIN LINALG_INV")

    if A.dim() < 2:
        raise RuntimeError(
            "linalg.inv: The input tensor A must have at least 2 dimensions."
        )
    if A.shape[-1] != A.shape[-2]:
        raise RuntimeError(
            "linalg.inv: A must be batches of square matrices, but they are "
            "%d by %d matrices" % (A.shape[-2], A.shape[-1])
        )
    if A.dtype in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "linalg.inv: Low precision dtypes not supported. Got %s"
            % ("Half" if A.dtype == torch.float16 else "BFloat16")
        )

    n = A.shape[-1]
    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)

    batch_shape = tuple(A.shape[:-2])
    batch = 1
    for s in batch_shape:
        batch *= int(s)
    if batch == 0 or n == 0 or A.numel() == 0:
        return out

    A_work = A.reshape(batch, n, n).contiguous()
    LU_work, piv_work = _linalg_lu_factor(A_work, True)

    # Solve A X = I; each RHS column of X is a column of A^-1.
    eye = torch.eye(n, dtype=A_work.dtype, device=A_work.device)
    rhs = eye.unsqueeze(0).expand(batch, n, n)
    with torch_device_fn.device(A.device):
        inv = _xpu_lu_solve_impl(LU_work, piv_work, rhs, batch, n, n)

    out.copy_(inv.reshape(A.shape))
    return out


__all__ = ["linalg_inv"]
