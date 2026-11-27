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

"""Kunlunxin backend for linalg_solve_ex (public profile).

The generic implementation fuses the LU factorization and the two triangular
substitutions into a single kernel; on this XPU the XTriton backend
mis-schedules that fusion once n is large enough (the LU step is wrong for
n >= 11 while the substitution-only kernels are exact, see
``_linalg_solve_ex``).

The public profile only returns (result, info), so it does not need the LU
factors at all: it solves the augmented ``[A | B]`` system with a
partial-pivoted Gauss-Jordan elimination (``_linalg_solve_ex_gj``), which needs
3 launches for batch <= 4 and one launch per column otherwise instead of the
per-step LU chain.  ``XA = B``, complex inputs and n outside the measured range
keep the LU-based ``_linalg_solve_ex`` implementation, which also serves the
private profile with its real LU/pivots outputs.
"""

import logging

from flag_gems.runtime.backend._kunlunxin.ops._linalg_solve_ex_gj import (
    _solve_ex_public,
)

logger = logging.getLogger(__name__)


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_EX")
    return _solve_ex_public(A, B, left=left, check_errors=check_errors)
