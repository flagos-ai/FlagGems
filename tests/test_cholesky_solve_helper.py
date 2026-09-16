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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _factor(shape, dtype):
    matrix = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    identity = torch.eye(shape[-1], device=flag_gems.device, dtype=dtype)
    return torch.linalg.cholesky(matrix @ matrix.mT + 0.5 * identity)


# Covers unbatched, batched, and multiple-RHS cases. The raw aten helper does
# not broadcast a 2D factor across the rhs batch dim (that happens in the
# higher-level cholesky_solve), so factor and rhs share matching batch dims.
CHOLESKY_SOLVE_CASES = [
    ((5, 3), (5, 5)),
    ((2, 5, 3), (2, 5, 5)),
    ((3, 8, 4), (3, 8, 8)),
]


@pytest.mark.cholesky_solve_helper
@pytest.mark.parametrize("rhs_shape,factor_shape", CHOLESKY_SOLVE_CASES)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_solve_helper(rhs_shape, factor_shape, upper, dtype):
    factor = _factor(factor_shape, dtype)
    if upper:
        factor = factor.mT.contiguous()
    rhs = torch.randn(rhs_shape, device=flag_gems.device, dtype=dtype)
    ref_rhs = utils.to_reference(rhs)
    ref_factor = utils.to_reference(factor)
    expected = torch.ops.aten._cholesky_solve_helper(ref_rhs, ref_factor, upper)

    with flag_gems.use_gems():
        actual = torch.ops.aten._cholesky_solve_helper(rhs, factor, upper)

    utils.gems_assert_close(actual, expected, dtype, reduce_dim=rhs_shape[-2])
