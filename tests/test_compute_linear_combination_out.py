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


def _reference_out(inp, coefficients, out):
    return torch.ops.aten._compute_linear_combination.out(
        inp, coefficients, out=out
    )


@pytest.mark.compute_linear_combination_out
@pytest.mark.underscore_compute_linear_combination_out
@pytest.mark.parametrize("m,n,k", MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_out(m, n, k, dtype):
    torch.manual_seed(0)
    inp = torch.randn((n, k), dtype=dtype, device=flag_gems.device)
    coefficients = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    saved_inp = inp.clone()
    saved_coefficients = coefficients.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)
    ref_out = torch.zeros((m, k), dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.zeros((m, k), dtype=dtype, device=flag_gems.device)
    out_pointer = out.data_ptr()

    # ATen accumulates into out, so both calls start from the same zero buffer.
    reference = _reference_out(ref_inp, ref_coefficients, ref_out)
    result = flag_gems._compute_linear_combination_out(inp, coefficients, out=out)

    assert reference is ref_out
    assert result is out
    assert out.data_ptr() == out_pointer
    assert out.shape == (m, k)
    assert out.dtype == dtype
    utils.gems_assert_close(out, reference, dtype, reduce_dim=n)
    assert inp.device == saved_inp.device
    assert coefficients.device == saved_coefficients.device
    utils.gems_assert_equal(inp, utils.to_reference(saved_inp))
    utils.gems_assert_equal(coefficients, utils.to_reference(saved_coefficients))


@pytest.mark.compute_linear_combination_out
@pytest.mark.underscore_compute_linear_combination_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_out_resize(dtype):
    torch.manual_seed(1)
    m, n, k = 3, 4, 5
    inp = torch.randn((n, k), dtype=dtype, device=flag_gems.device)
    coefficients = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)
    ref_out = torch.empty((0,), dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.empty((0,), dtype=dtype, device=flag_gems.device)

    try:
        reference = _reference_out(ref_inp, ref_coefficients, ref_out)
    except RuntimeError as error:
        if "doesn't match the broadcast shape" not in str(error):
            raise
        pytest.skip("Native linear-combination out cannot resize an empty output")
    result = flag_gems._compute_linear_combination_out(inp, coefficients, out=out)

    assert reference is ref_out
    assert result is out
    assert out.shape == (m, k)
    assert out.dtype == dtype
    utils.gems_assert_close(out, reference, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination_out
@pytest.mark.underscore_compute_linear_combination_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_compute_linear_combination_out_noncontiguous(dtype):
    torch.manual_seed(2)
    m, n, k = 5, 7, 9
    inp = torch.randn((n, k), dtype=dtype, device=flag_gems.device)
    coefficients = torch.randn((m, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_coefficients = utils.to_reference(coefficients, True)
    ref_out = torch.zeros((k, m), dtype=ref_inp.dtype, device=ref_inp.device).t()
    out = torch.zeros((k, m), dtype=dtype, device=flag_gems.device).t()
    out_pointer = out.data_ptr()
    out_stride = out.stride()
    assert not out.is_contiguous()

    # ATen accumulates into out, so both calls start from the same zero buffer.
    reference = _reference_out(ref_inp, ref_coefficients, ref_out)
    result = flag_gems._compute_linear_combination_out(inp, coefficients, out=out)

    assert reference is ref_out
    assert result is out
    assert out.data_ptr() == out_pointer
    assert out.stride() == out_stride
    utils.gems_assert_close(out, reference, dtype, reduce_dim=n)


@pytest.mark.compute_linear_combination_out
@pytest.mark.underscore_compute_linear_combination_out
def test_compute_linear_combination_out_rejects_dtype_mismatch():
    torch.manual_seed(0)
    inp = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    coefficients = torch.randn((3, 4), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((3, 5), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises((RuntimeError, AssertionError)):
        flag_gems._compute_linear_combination_out(
            inp, coefficients, out=out
        )
