# Copyright 2026, The FlagOS Contributors.
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
"""Kunlunxin (XPU) linalg_solve_ex.

The generic single-program kernel fuses A->LU copy, partial-pivoted LU
factorisation, and the two triangular solves behind a nest of scalar
``while`` loops.  On this backend those scalar loops with data-dependent
``if`` pivots are miscompiled: even a *diagonal* matrix yields NaN/Inf and a
well-conditioned SPD system comes out with an O(1) residual.

This override composes the two ops that are already correct on this backend —
``linalg_lu_factor_ex`` (vendor LU with pivoting) followed by the generic
``linalg_lu_solve`` (forward/back substitution) — reproducing
``torch.linalg.solve_ex`` semantics (``result, info``) without touching the
generic implementation.
"""

import logging

import torch

from flag_gems.ops.linalg_lu_solve import linalg_lu_solve

from .linalg_lu_factor_ex import linalg_lu_factor_ex

logger = logging.getLogger(__name__)


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    """Solve the linear system AX = B, returning ``(result, info)``.

    Identical contract to the generic implementation; only the execution path
    differs (factor-then-substitute through the standalone LU ops).
    """
    logger.debug("GEMS LINALG_SOLVE_EX (kunlunxin)")

    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("A and B must be at least 2D")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix")

    if A.numel() == 0 or B.numel() == 0:
        batch_shape = A.shape[:-2]
        info = torch.zeros(batch_shape, dtype=torch.int32, device=A.device)
        return B.clone(), info

    LU, pivots, info = linalg_lu_factor_ex(A, pivot=True, check_errors=False)
    result = linalg_lu_solve(LU, pivots, B, left=left)

    if check_errors and torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            "linalg.solve_ex: The diagonal element of the LU decomposition is zero."
        )

    return result, info
