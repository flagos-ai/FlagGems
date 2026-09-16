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
from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES, to_reference


@pytest.mark.binomial
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n, p", [(10, 0.5), (20, 0.3), (100, 0.9), (1, 0.5)])
def test_binomial(shape, dtype, n, p):
    count = torch.full(
        size=shape, fill_value=float(n), dtype=dtype, device=flag_gems.device
    )
    prob = torch.full(
        size=shape, fill_value=float(p), dtype=dtype, device=flag_gems.device
    )
    with flag_gems.use_gems():
        res_out = torch.binomial(count, prob)

    ref_count = to_reference(count)
    ref_prob = to_reference(prob)
    # CPU binomial doesn't support bfloat16, convert to float32 for reference
    if ref_count.dtype == torch.bfloat16:
        ref_count = ref_count.to(torch.float32)
        ref_prob = ref_prob.to(torch.float32)
    ref_out = torch.binomial(ref_count, ref_prob)
    mean = torch.mean(ref_out)
    var = torch.var(ref_out)

    expected_mean = n * p
    expected_var = n * p * (1 - p)

    assert torch.abs(mean - expected_mean) < max(0.3, 0.05 * expected_mean)
    assert torch.abs(var - expected_var) < max(0.5, 0.15 * expected_var + 0.5)
    assert (res_out >= 0).all()
    assert (res_out <= n).all()


@pytest.mark.binomial
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binomial_large_count(shape, dtype):
    # Large count triggers the normal-approximation branch of the kernel.
    n, p = 1000.0, 0.5
    count = torch.full(size=shape, fill_value=n, dtype=dtype, device=flag_gems.device)
    prob = torch.full(size=shape, fill_value=p, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.binomial(count, prob)

    ref_out = to_reference(res_out).to(torch.float32)
    mean = torch.mean(ref_out)

    assert torch.abs(mean - n * p) < 5.0
    assert (res_out >= 0).all()
    assert (res_out <= n).all()
    assert torch.isfinite(res_out).all()


@pytest.mark.binomial
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binomial_varying_prob(shape, dtype):
    count = torch.randint(1, 100, size=shape, device=flag_gems.device).to(dtype)
    prob = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.binomial(count, prob)

    assert (res_out >= 0).all()
    assert (res_out <= count).all()
    assert torch.isfinite(res_out).all()


@pytest.mark.binomial
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binomial_edge_probs(dtype):
    # 2000 elements: large enough to statistically confirm deterministic edge
    # cases (p==0/p==1) while keeping the test fast.
    shape = (2000,)
    count = torch.full(size=shape, fill_value=8.0, dtype=dtype, device=flag_gems.device)

    # p == 0 -> always 0, p == 1 -> always count
    prob0 = torch.zeros(size=shape, dtype=dtype, device=flag_gems.device)
    prob1 = torch.ones(size=shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        out0 = torch.binomial(count, prob0)
        out1 = torch.binomial(count, prob1)

    # p==0 is deterministically 0, p==1 is deterministically count (==8).
    utils.gems_assert_equal(out0, to_reference(torch.zeros_like(count)))
    utils.gems_assert_equal(out1, to_reference(count))


@pytest.mark.binomial
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binomial_zero_count(dtype):
    # 2000 elements: enough samples to confirm count==0 always yields 0.
    shape = (2000,)
    count = torch.zeros(size=shape, dtype=dtype, device=flag_gems.device)
    prob = torch.full(size=shape, fill_value=0.5, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.binomial(count, prob)

    # count==0 deterministically yields 0 regardless of prob.
    utils.gems_assert_equal(res_out, to_reference(torch.zeros_like(count)))


@pytest.mark.binomial_out
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binomial_out(shape, dtype):
    n, p = 20.0, 0.4
    count = torch.full(size=shape, fill_value=n, dtype=dtype, device=flag_gems.device)
    prob = torch.full(size=shape, fill_value=p, dtype=dtype, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.binomial.out(count, prob, out=out)

    assert res_out.data_ptr() == out.data_ptr()
    ref_out = to_reference(out).to(torch.float32)
    mean = torch.mean(ref_out)

    assert torch.abs(mean - n * p) < 0.5
    assert (out >= 0).all()
    assert (out <= n).all()
