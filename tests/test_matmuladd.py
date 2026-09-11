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

# Shapes copied from the worktree shared BLAS tests for matmul-style coverage.
MNK_SHAPES = [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
]


def make_out(M, N, dtype, is_contiguous):
    if is_contiguous:
        return torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    return torch.empty((N, M), dtype=dtype, device=flag_gems.device).t()


@pytest.mark.matmuladd
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmuladd(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #3794: Skipping fp32 matmuladd test on TsingMicro because "
            "TX81 does not support fp32 dot."
        )

    # Test case 1: 1D bias
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    # Reference: matmul + bias
    ref_out = torch.matmul(ref_mat1, ref_mat2) + ref_bias
    with flag_gems.use_gems():
        res_out = flag_gems.matmuladd(mat1, mat2, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    # Test case 2: 2D bias (broadcasted)
    bias_2d = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias_2d = utils.to_reference(bias_2d, True)

    ref_out_2d = torch.matmul(ref_mat1, ref_mat2) + ref_bias_2d
    with flag_gems.use_gems():
        res_out_2d = flag_gems.matmuladd(mat1, mat2, bias_2d)

    utils.gems_assert_close(res_out_2d, ref_out_2d, dtype, reduce_dim=K)
