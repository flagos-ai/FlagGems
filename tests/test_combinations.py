# Copyright 2026, The FlagOS Contributors.
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

# (n, r, with_replacement) triples. r == 2 and r == 3 take the closed-form
# unranking kernels; every other rank takes the generic binary-search one.
# The 300-element entries cross the 4 * 256-row tile boundary of the r == 2
# kernel and the 256-row block boundary of the r == 3 kernel, and the (3, 5)
# entry has r > n, so the output is empty.
COMBINATIONS_CASES = [
    (0, 2, False),
    (0, 2, True),
    (1, 2, False),
    (1, 2, True),
    (5, 0, False),
    (5, 0, True),
    (5, 1, False),
    (5, 1, True),
    (5, 2, False),
    (5, 2, True),
    (8, 3, False),
    (8, 3, True),
    (8, 4, False),
    (8, 4, True),
    (8, 5, True),
    (3, 5, False),
    (300, 2, False),
    (300, 2, True),
    (64, 3, False),
    (64, 3, True),
]


def _make_input(n, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, (n,), dtype=torch.uint8).to(
            dtype=torch.bool, device=flag_gems.device
        )
    if dtype in utils.FLOAT_DTYPES:
        return torch.randn((n,), dtype=dtype, device=flag_gems.device)
    return torch.randint(-1000, 1000, (n,), dtype=dtype, device=flag_gems.device)


@pytest.mark.combinations
@pytest.mark.parametrize("n,r,with_replacement", COMBINATIONS_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations(n, r, with_replacement, dtype):
    inp = _make_input(n, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("n,r,with_replacement", COMBINATIONS_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + utils.BOOL_TYPES)
def test_combinations_int_bool(n, r, with_replacement, dtype):
    inp = _make_input(n, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [2, 3, 4, 5])
def test_combinations_row_order(r):
    # Distinct values make the row order part of the contract: a combination
    # that is present but emitted at the wrong lexicographic position fails.
    inp = torch.arange(12, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, False)
    res_out = flag_gems.combinations(inp, r, False)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [2, 3, 5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations_non_contiguous(r, dtype):
    # The kernels index the input through its stride, so a step-2 view must be
    # read through the stride instead of linearly.
    base = torch.randn((64,), dtype=dtype, device=flag_gems.device)
    inp = base[::2]
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, False)
    res_out = flag_gems.combinations(inp, r, False)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations_nan_inf(r, dtype):
    values = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, False)
    res_out = flag_gems.combinations(inp, r, False)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.combinations
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations_default_arguments(dtype):
    # aten::combinations defaults to r=2 without replacement.
    inp = torch.randn((7,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp)
    res_out = flag_gems.combinations(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [-1, -3])
def test_combinations_rejects_negative_r(r):
    inp = torch.randn((5,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.combinations(inp, r)
    with pytest.raises(RuntimeError):
        torch.combinations(utils.to_reference(inp), r)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", [(), (3, 4), (2, 3, 4)])
def test_combinations_rejects_non_1d(shape):
    inp = torch.randn(shape, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.combinations(inp)
    with pytest.raises(RuntimeError):
        torch.combinations(utils.to_reference(inp))
