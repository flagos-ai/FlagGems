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
from .conftest import TO_CPU

# Shapes representative of common batch-norm workloads.
# The feature dimension is always shape[1].
SHAPES = [
    (16, 3, 32),
    (32, 32, 32),
    (8, 32, 224, 224),
    (32, 64, 56, 56),
    (64, 128, 28, 28),
    (16, 256, 14, 14),
]


@pytest.mark.batch_norm_gather_stats
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("num_segments", [2, 4])
def test_batch_norm_gather_stats(shape, dtype, num_segments):
    # batch_norm_gather_stats has no CPU backend in PyTorch, skip under --ref=cpu
    if TO_CPU:
        pytest.skip("batch_norm_gather_stats has no CPU backend; skip under --ref=cpu")

    C = shape[1]
    N = shape[0]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    count = N * spatial

    input_t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mean = torch.randn(num_segments, C, dtype=torch.float32, device=flag_gems.device)
    invstd = (
        torch.rand(num_segments, C, dtype=torch.float32, device=flag_gems.device) + 0.1
    )
    running_mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
    running_var = torch.rand(C, dtype=torch.float32, device=flag_gems.device) + 0.1

    ref_out = torch.batch_norm_gather_stats(
        input_t,
        mean,
        invstd,
        running_mean.clone(),
        running_var.clone(),
        0.1,
        1e-5,
        count,
    )

    with flag_gems.use_gems():
        res_out = torch.batch_norm_gather_stats(
            input_t,
            mean,
            invstd,
            running_mean.clone(),
            running_var.clone(),
            0.1,
            1e-5,
            count,
        )

    for ref_val, res_val in zip(ref_out, res_out):
        # Output is always float32 regardless of input dtype
        utils.gems_assert_close(res_val, ref_val, torch.float32)
