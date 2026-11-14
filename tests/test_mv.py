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
from . import conftest as cfg

if cfg.QUICK_MODE:
    MN_SHAPES = [
        (1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    res_out = flag_gems.mv(matrix, vector)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)


@pytest.mark.mv
@pytest.mark.parametrize("M, N", [(4096, 64), (16384, 8)])
def test_mv_float64_precision(M, N):
    # float64 inputs must be accumulated in float64 (issue #6725).
    matrix = torch.randn((N, M), dtype=torch.float64, device=flag_gems.device)
    vector = torch.randn((M,), dtype=torch.float64, device=flag_gems.device)
    ref_out = torch.mv(matrix.cpu(), vector.cpu())

    res_out = flag_gems.mv(matrix, vector)

    assert res_out.dtype == torch.float64
    torch.testing.assert_close(res_out.cpu(), ref_out, rtol=1e-10, atol=1e-10)
