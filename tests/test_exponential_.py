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


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.exponential_()

    assert x.min() > 0


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_fast(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    with flag_gems.use_gems():
        x.exponential_()

    x_res = utils.to_reference(x)
    mean_res = torch.mean(x_res.to(torch.float32)).to(dtype)
    var_res = torch.var(x_res.to(torch.float32)).to(dtype)
    mean_ref = 1.0 / lambd
    var_ref = 1.0 / (lambd**2)

    assert torch.abs(mean_res - mean_ref) < mean_tol
    assert torch.abs(var_res - var_ref) < var_tol
