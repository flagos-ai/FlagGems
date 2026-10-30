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

MNK_SHAPES = [
    (1, 1, 1),
    (3, 4, 5),
    (8, 16, 32),
    (2, 64, 128),
    (32, 64, 128),
    (5, 17, 3),
]
HIGHER_RANK_SHAPES = [
    (3, 4, 2, 6),
    (2, 8, 8, 8),
]


def _assert_inputs_unchanged(actual, expected):
    assert actual.device == expected.device
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    utils.gems_assert_equal(actual, utils.to_reference(expected, False))


@pytest.mark.compute_linear_combination
@pytest.mark.underscore_compute_linear_combination
@pytest.mark.parametrize("m,n,k", MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination(m, n, k, dtype):
    torch.manual_seed(0)
    inp = torch.randn((n, k), dtype=dtype, device=flag_gems.device)
    coefficients = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    saved_inp = inp.clone()
    saved_coefficients = coefficients.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)

    reference = torch._compute_linear_combination(ref_inp, ref_coefficients)
    result = flag_gems._compute_linear_combination(inp, coefficients)

    assert result.shape == (m, k)
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, reduce_dim=n)
    _assert_inputs_unchanged(inp, saved_inp)
    _assert_inputs_unchanged(coefficients, saved_coefficients)


@pytest.mark.compute_linear_combination
@pytest.mark.underscore_compute_linear_combination
@pytest.mark.parametrize("m,n,a,b", HIGHER_RANK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_higher_rank(m, n, a, b, dtype):
    torch.manual_seed(1)
    inp = torch.randn((n, a, b), dtype=dtype, device=flag_gems.device)
    coefficients = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)

    reference = torch._compute_linear_combination(ref_inp, ref_coefficients)
    result = flag_gems._compute_linear_combination(inp, coefficients)

    assert result.shape == (m, a, b)
    utils.gems_assert_close(result, reference, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination
@pytest.mark.underscore_compute_linear_combination
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_noncontiguous(dtype):
    torch.manual_seed(2)
    m, n, a, b = 5, 7, 3, 4
    inp = torch.randn((n, b, a), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    coefficients = torch.randn((n, m), dtype=dtype, device=flag_gems.device).t()
    assert not inp.is_contiguous()
    assert not coefficients.is_contiguous()
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)

    reference = torch._compute_linear_combination(ref_inp, ref_coefficients)
    result = flag_gems._compute_linear_combination(inp, coefficients)

    utils.gems_assert_close(result, reference, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination
@pytest.mark.underscore_compute_linear_combination
@pytest.mark.parametrize("m,n,k", [(2, 8, 64), (4, 16, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_zeros(m, n, k, dtype):
    inp = torch.zeros((n, k), dtype=dtype, device=flag_gems.device)
    coefficients = torch.zeros((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)

    reference = torch._compute_linear_combination(ref_inp, ref_coefficients)
    result = flag_gems._compute_linear_combination(inp, coefficients)

    utils.gems_assert_close(result, reference, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination
@pytest.mark.underscore_compute_linear_combination
@pytest.mark.parametrize(
    "input_shape,coefficient_shape",
    [
        ((0, 4), (3, 0)),
        ((4, 5), (3, 2)),
        ((4, 5), (3, 4, 1)),
    ],
)
def test_compute_linear_combination_invalid_shapes(input_shape, coefficient_shape):
    torch.manual_seed(0)
    inp = torch.randn(input_shape, dtype=torch.float32, device=flag_gems.device)
    coefficients = torch.randn(
        coefficient_shape, dtype=torch.float32, device=flag_gems.device
    )

    with pytest.raises((AssertionError, RuntimeError)):
        flag_gems._compute_linear_combination(inp, coefficients)
