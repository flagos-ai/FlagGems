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

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES, to_reference


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson(shape, dtype):
    lam = 5.0
    inp = torch.full(size=shape, fill_value=lam, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out.to(torch.float32))
    var = torch.var(ref_out.to(torch.float32))

    assert torch.abs(mean - lam) < 0.3
    assert torch.abs(var - lam) < 0.5
    assert (res_out >= 0).all()


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson_varying_rates(shape, dtype):
    inp = torch.rand(size=shape, dtype=dtype, device=flag_gems.device) * 10 + 1

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    assert (res_out >= 0).all()
    assert torch.isfinite(res_out).all()


@pytest.mark.poisson
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson_large_lambda(shape, dtype):
    lam = 50.0
    inp = torch.full(size=shape, fill_value=lam, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out.to(torch.float32))
    var = torch.var(ref_out.to(torch.float32))

    assert torch.abs(mean - lam) < 1.0
    assert torch.abs(var - lam) < 5.0
    assert (res_out >= 0).all()


@pytest.mark.poisson
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_poisson_zero_rate(dtype):
    shape = (1000,)
    inp = torch.zeros(size=shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.poisson(inp)

    assert (res_out == 0).all()
