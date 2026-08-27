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
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    # Shape format: (M, N, K) for matmul: (M, K) x (K, N)
    # Includes small / edge cases, non-square matrices, and large square
    # matrices aligned with the BlasBenchmark shapes (384 / 1024 / 2048 / 4096)
    # so that accuracy is validated on the same sizes that are benchmarked.
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
        (384, 384, 384),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (128, 256, 512),
        (256, 1024, 4096),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.matmul_bias_activation
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_matmul_bias_activation(M, N, K, dtype):
    # Create input tensors
    input_tensor = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    # Reference: matmul + bias + relu
    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.relu(torch.mm(ref_input, ref_weight) + ref_bias)
    with flag_gems.use_gems():
        from flag_gems.fused.matmul_bias_activation import matmul_bias_activation

        res_out = matmul_bias_activation(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
