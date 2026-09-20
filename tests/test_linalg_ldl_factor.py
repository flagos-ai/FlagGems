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


@pytest.mark.linalg_ldl_factor
@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (16, 16), (32, 32)])
# torch.linalg.ldl_factor on CUDA supports float32/float64 for this path.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("hermitian", [False, True])
def test_linalg_ldl_factor(shape, dtype, hermitian):
    # Create a symmetric positive definite matrix
    # A = A @ A.T + I ensures positive definiteness
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = A @ A.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    ref_A = utils.to_reference(A)

    ref_LD, ref_pivots = torch.linalg.ldl_factor(ref_A, hermitian=hermitian)
    with flag_gems.use_gems():
        res_LD, res_pivots = torch.linalg.ldl_factor(A, hermitian=hermitian)

    utils.gems_assert_close(res_LD, ref_LD, dtype)
    utils.gems_assert_equal(res_pivots, ref_pivots)
