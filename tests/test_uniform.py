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


@pytest.mark.uniform_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_uniform_(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    flag_gems.uniform_(x, -3, 3)

    # uniform samples from [from, to); a strict upper bound keeps an upper-edge
    # rounding bug from passing.
    assert (x < 3.0).all()
    assert (x >= -3.0).all()


@pytest.mark.uniform
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    x_copy = x.clone()
    out = flag_gems.uniform(x, -3, 3)

    # out-of-place uniform must not modify the input; that is the main semantic
    # difference from uniform_.
    assert torch.equal(x, x_copy)
    assert (out < 3.0).all()
    assert (out >= -3.0).all()


@pytest.mark.uniform
def test_uniform_rejects_inverted_interval():
    x = torch.randn(size=(8,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"\[from, to\)"):
        flag_gems.uniform(x, 3.0, -1.0)


@pytest.mark.uniform
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_uniform_upper_bound_zero(dtype):
    # `to == +0.0` is the case the one-ulp upper clamp used to get wrong: the
    # zero bit pattern is read as a small positive integer by the bitwise step,
    # so stepping "below" it wraps out of the number line onto an all-ones NaN.
    # `tl.minimum` then silently returns the sampled value, i.e. values at (or
    # above) the exclusive bound leak out. The interval below is chosen so the
    # clamp actually engages for low-precision dtypes (the scaled values round
    # up to the bound), which is what makes this a regression test rather than
    # a tautology.
    x = torch.empty(size=(2**20,), dtype=dtype, device=flag_gems.device)
    out = flag_gems.uniform(x, -1e-6, 0.0)

    assert not torch.isnan(out).any()
    assert (out < 0.0).all()
    assert (out >= -1e-6).all()
