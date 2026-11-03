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


@pytest.mark.binomial
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binomial_metadata_and_range(shape, dtype):
    torch.manual_seed(0)
    count = torch.randint(1, 100, shape, device=flag_gems.device).to(dtype)
    prob = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    count_before = count.clone()
    prob_before = prob.clone()

    result = flag_gems.binomial(count, prob)

    assert result.shape == count.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0)
    assert torch.all(result <= count)
    assert torch.isfinite(result).all()
    utils.gems_assert_equal(count, count_before)
    utils.gems_assert_equal(prob, prob_before)


@pytest.mark.binomial
@pytest.mark.parametrize("n,p", [(10.0, 0.5), (20.0, 0.3), (100.0, 0.9)])
def test_binomial_distribution_moments(n, p):
    count = torch.full((10000,), n, device=flag_gems.device)
    prob = torch.full((10000,), p, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device).manual_seed(12345)

    result = flag_gems.binomial(count, prob, generator=generator).float()

    expected_mean = n * p
    expected_variance = n * p * (1 - p)
    assert abs(result.mean().item() - expected_mean) < max(0.3, 0.05 * expected_mean)
    assert abs(result.var().item() - expected_variance) < max(
        0.5, 0.15 * expected_variance + 0.5
    )


@pytest.mark.binomial
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binomial_deterministic_boundaries(dtype):
    count = torch.full((2000,), 8.0, dtype=dtype, device=flag_gems.device)
    zeros = torch.zeros_like(count)
    ones = torch.ones_like(count)

    result_zero_prob = flag_gems.binomial(count, zeros)
    result_one_prob = flag_gems.binomial(count, ones)
    result_zero_count = flag_gems.binomial(zeros, torch.full_like(count, 0.5))

    utils.gems_assert_equal(result_zero_prob, utils.to_reference(zeros, False))
    utils.gems_assert_equal(result_one_prob, utils.to_reference(count, False))
    utils.gems_assert_equal(result_zero_count, utils.to_reference(zeros, False))


@pytest.mark.binomial
def test_binomial_generator_contract():
    count = torch.full((2048,), 20.0, device=flag_gems.device)
    prob = torch.full((2048,), 0.4, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device)

    generator.manual_seed(2026)
    first = flag_gems.binomial(count, prob, generator=generator)
    second = flag_gems.binomial(count, prob, generator=generator)
    assert not torch.equal(first, second)

    generator.manual_seed(2026)
    replay = flag_gems.binomial(count, prob, generator=generator)
    utils.gems_assert_equal(first, replay)


@pytest.mark.binomial
@pytest.mark.binomial_out
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binomial_out_contract(shape, dtype):
    if flag_gems.vendor_name == "ascend" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "Ascend binomial.out falls back to CPU, "
            "which does not support Half/BFloat16"
        )
    count = torch.full(shape, 20.0, dtype=dtype, device=flag_gems.device)
    prob = torch.full(shape, 0.4, dtype=dtype, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    pointer = out.data_ptr()

    result = flag_gems.binomial_out(count, prob, out=out)

    assert result is out
    assert out.data_ptr() == pointer
    assert torch.all(out >= 0)
    assert torch.all(out <= count)
