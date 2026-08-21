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


@pytest.mark.batch_norm_stats
@pytest.mark.skipif(
    TO_CPU, reason="torch.batch_norm_stats does not support CPU backend"
)
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_stats(shape, dtype):
    eps = 1e-5
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_mean, ref_invstd = torch.batch_norm_stats(ref_inp, eps)

    with flag_gems.use_gems():
        res_mean, res_invstd = torch.batch_norm_stats(inp, eps)

    # batch_norm_stats always returns float32 regardless of input dtype
    utils.gems_assert_close(res_mean, ref_mean, torch.float32)
    utils.gems_assert_close(res_invstd, ref_invstd, torch.float32)
