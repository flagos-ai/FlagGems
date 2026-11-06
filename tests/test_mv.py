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
from benchmark.gemm_shapes import accuracy_cases, accuracy_shapes

from . import accuracy_utils as utils
from . import conftest as cfg

MN_SHAPES = accuracy_shapes("mv", cfg.QUICK_MODE)
FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES

# THead GEMM delivery coverage is BF16, including quick-mode runs.
if flag_gems.vendor_name == "thead":
    FLOAT_DTYPES = [torch.bfloat16]


@pytest.mark.mv
@pytest.mark.parametrize("N, K, column_major", accuracy_cases("mv", cfg.QUICK_MODE))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
