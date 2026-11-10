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
``_linalg_solve_ex``).  This public variant therefore delegates to the private
``_linalg_solve_ex`` (multi-kernel ``_linalg_lu_factor`` + substitution),
which returns (result, LU, pivots, info); only (result, info) is exposed.
"""

import logging

from flag_gems.runtime.backend._kunlunxin.ops._linalg_solve_ex import (
    _linalg_solve_ex as _linalg_solve_ex_impl,
)

logger = logging.getLogger(__name__)


def linalg_solve_ex(A, B, *, left=True, check_errors=False):
    logger.debug("GEMS_KUNLUNXIN LINALG_SOLVE_EX")
    result, _, _, info = _linalg_solve_ex_impl(
        A, B, left=left, check_errors=check_errors
    )
    return result, info
