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

from flag_gems.ops.triangular_solve import _triangular_solve

from .linalg_solve_triangular import linalg_solve_triangular

logger = logging.getLogger(__name__)


def triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    logger.debug("GEMS_METAX TRIANGULAR_SOLVE")
    return _triangular_solve(
        B, A, upper, transpose, unitriangular, linalg_solve_triangular
    )


def triangular_solve_out(
    B, A, upper=True, transpose=False, unitriangular=False, *, X, M
):
    logger.debug("GEMS_METAX TRIANGULAR_SOLVE_OUT")
    return _triangular_solve(
        B, A, upper, transpose, unitriangular, linalg_solve_triangular, X, M
    )
