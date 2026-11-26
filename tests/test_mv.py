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
from .gemm_utils import accuracy_dtype_cases


@pytest.mark.mv
@pytest.mark.parametrize(
    "dtype, N, K, column_major", accuracy_dtype_cases("mv", cfg.QUICK_MODE)
)
def test_mv(N, K, dtype, column_major):
    matrix = (
        torch.randn((K, N), dtype=dtype, device=flag_gems.device).t()
        if column_major
        else torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    )
    vector = torch.randn((K,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    res_out = flag_gems.mv(matrix, vector)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mv
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="THead column-major MV")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows, k, padding", [(7, 33, 0), (7, 16387, 5), (37, 3073, 5), (1537, 257, 0)]
)
def test_thead_mv_column_major(rows, k, padding, dtype):
    storage = torch.randn((k, rows + padding), dtype=dtype, device=flag_gems.device)
    start = int(padding != 0)
    matrix = storage[:, start : start + rows].t()
    vector = torch.randn((k,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)
    expected = torch.mv(ref_matrix, ref_vector)

    with flag_gems.use_gems(include=["mv"]):
        actual = torch.mv(matrix, vector)

    utils.gems_assert_close(actual, expected, dtype, reduce_dim=k)
