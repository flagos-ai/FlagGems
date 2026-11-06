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
from . import conftest as cfg

if cfg.QUICK_MODE:
    # (input_shape, dim, index) triples.
    SELECT_COPY_CASES = [
        ((4, 6), 1, 2),
        ((4, 6), 0, 1),
        ((2, 3, 4), 1, 2),
    ]
else:
    # (input_shape, dim, index) triples. Contiguous inputs across every dim
    # position, negative dim/index normalization and empty tensors are covered.
    SELECT_COPY_CASES = [
        ((4, 6), 0, 0),
        ((4, 6), 0, 3),
        ((4, 6), 0, -1),
        ((4, 6), 1, 2),
        ((4, 6), 1, 5),
        ((4, 6), 1, -1),
        ((4, 6), -2, 1),
        ((4, 6), -1, 0),
        ((12,), 0, 5),
        ((12,), 0, -1),
        ((2, 3, 4), 0, 1),
        ((2, 3, 4), 1, 2),
        ((2, 3, 4), 2, 3),
        ((2, 3, 4), -1, -2),
        ((2, 4, 5, 7), 1, 2),
        ((2, 4, 5, 7), 2, 4),
        ((5,), 0, 4),
        ((1, 1), 1, 0),
        ((0, 4), 1, 1),
    ]


@pytest.mark.select_copy
@pytest.mark.parametrize("input_shape, dim, index", SELECT_COPY_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_select_copy(input_shape, dim, index, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.select_copy(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)

    assert list(res_out.shape) == list(ref_out.shape)
    assert res_out.is_contiguous()
    assert not torch._C._is_alias_of(res_out, inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim, index", [(0, 1), (0, -1), (1, 3), (1, 0)])
def test_accuracy_select_copy_noncontiguous_input(dtype, dim, index):
    base = torch.randn(6, 4, dtype=dtype, device=flag_gems.device)
    inp = base.t()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.select_copy(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim, index", [(0, 2), (1, 1), (2, -1)])
def test_accuracy_select_copy_strided_input(dtype, dim, index):
    # A stride-2 view: the strided gather path must honor the real strides.
    base = torch.randn(4, 8, 6, dtype=dtype, device=flag_gems.device)
    inp = base[:, ::2, :]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.select_copy(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim, index", [(0, 1), (1, 2)])
def test_accuracy_select_copy_broadcast_input(dtype, dim, index):
    # An expanded (zero-stride) input; the copy must materialize the values.
    base = torch.randn(1, 6, dtype=dtype, device=flag_gems.device)
    inp = base.expand(4, 6)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.select_copy(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim, index", [(0, 1), (1, 2)])
def test_accuracy_select_copy_storage_offset(dtype, dim, index):
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[4:12].reshape(2, 4)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.select_copy(ref_inp, dim, index)
    res_out = flag_gems.select_copy(inp, dim, index)
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.select_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_select_copy_independent_storage(dtype):
    inp = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.select_copy(inp, 1, 2)
    ref = utils.to_reference(res_out.clone())

    inp.add_(1.0)
    # The copy must not change when the source is mutated in place.
    utils.gems_assert_equal(res_out, ref)


@pytest.mark.select_copy
def test_accuracy_select_copy_error_paths():
    inp = torch.randn(4, 3, device=flag_gems.device)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.select_copy(inp, 1, 3)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.select_copy(inp, 1, 3)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.select_copy(inp, 1, -4)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.select_copy(inp, 1, -4)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.select_copy(torch.tensor(1.0, device=flag_gems.device), 0, 0)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.select_copy(torch.tensor(1.0, device=flag_gems.device), 0, 0)
