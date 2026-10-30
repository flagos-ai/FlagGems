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

# (n, r) pairs: n is the input vector length, r the combination width. The
# pairs are listed explicitly rather than crossed, because the row count grows
# as C(n, r) and a cross product would blow up (n=129, r=5 is 2.6e8 rows).
#
# Covered: r == 0 (empty 1-D result), n < r (empty 2-D result), n == r (single
# row), r == 2 and r == 3 (the two closed-form unranking kernels, including
# sizes past one program block of BLOCK=256 x 4 rows per lane), r == 4 (the
# closed-form r == 4 kernel, including sizes past one program block of
# BLOCK=256 x 4 rows per lane), and r >= 5 (the generic binary-search kernel).
COMBINATIONS_CASES = [
    (0, 0),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
    (3, 2),
    (5, 2),
    (64, 2),
    (129, 2),
    (3, 3),
    (5, 3),
    (16, 3),
    (64, 3),
    (4, 4),
    (16, 4),
    (129, 4),
    (130, 4),
    (6, 5),
    (12, 5),
]


@pytest.mark.combinations
@pytest.mark.parametrize("n, r", COMBINATIONS_CASES)
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations(n, r, with_replacement, dtype):
    inp = torch.randn((n,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, with_replacement=with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement=with_replacement)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("n, r", COMBINATIONS_CASES)
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + utils.BOOL_TYPES)
def test_combinations_int_bool(n, r, with_replacement, dtype):
    inp = torch.randint(0, 2, (n,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, with_replacement=with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement=with_replacement)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("with_replacement", [False, True])
def test_combinations_row_order(r, with_replacement):
    # Rows are emitted in lexicographic index order, so the exact row sequence
    # is part of the contract, not just the set of combinations. Distinct
    # values make a wrong order visible instead of cancelling out.
    inp = torch.arange(6, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r, with_replacement=with_replacement)
    res_out = flag_gems.combinations(inp, r, with_replacement=with_replacement)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations_non_contiguous(r, dtype):
    # Stride-2 slice of a larger buffer: the gather has to honour the input
    # stride instead of reading the underlying storage linearly.
    base = torch.randn((24,), dtype=dtype, device=flag_gems.device)
    inp = base[::2]
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r)
    res_out = flag_gems.combinations(inp, r)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_combinations_nan_inf(r, dtype):
    inp = torch.randn((8,), dtype=dtype, device=flag_gems.device)
    inp[0] = float("nan")
    inp[1] = float("inf")
    inp[2] = float("-inf")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp, r)
    res_out = flag_gems.combinations(inp, r)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.combinations
def test_combinations_default_r():
    # aten::combinations defaults to r=2, so the no-r call form must work.
    inp = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.combinations(ref_inp)
    res_out = flag_gems.combinations(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.combinations
def test_combinations_rejects_negative_r():
    inp = torch.randn((4,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.combinations(inp, -1)


@pytest.mark.combinations
@pytest.mark.parametrize("shape", [(), (3, 4)])
def test_combinations_rejects_non_1d_input(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.combinations(inp, 2)
