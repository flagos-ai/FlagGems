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
    RESHAPE_COPY_CASES = [
        ((4, 6), [4, 6]),
        ((4, 6), [24]),
        ((4, 6), [-1, 6]),
    ]
else:
    # (input_shape, size) pairs. Contiguous reshapes, dimension merges/splits,
    # added dimensions, -1 inference, high-rank and empty tensors are covered.
    RESHAPE_COPY_CASES = [
        ((4, 6), [4, 6]),
        ((4, 6), [24]),
        ((12,), [3, 4]),
        ((2, 3, 4), [6, 4]),
        ((2, 3, 4), [24]),
        ((4, 4), [2, 2, 4]),
        ((4, 6), [4, 6, 1]),
        ((4, 6, 1), [4, 6]),
        ((2, 3, 4, 5), [6, 4, 5]),
        ((2, 3, 4, 5), [120]),
        ((2, 3, 4, 5, 6), [6, 4, 5, 6]),
        ((4, 6), [-1, 6]),
        ((2, 3, 4), [2, -1]),
        ((24,), [-1]),
        ((0, 4), [0, 4]),
    ]


@pytest.mark.reshape_copy
@pytest.mark.parametrize("input_shape, size", RESHAPE_COPY_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy(input_shape, size, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, size)
    res_out = flag_gems._reshape_copy(inp, size)

    assert list(res_out.shape) == list(ref_out.shape)
    assert res_out.is_contiguous()
    assert not torch._C._is_alias_of(res_out, inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy_noncontiguous_input(dtype):
    base = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    inp = base.t()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, [24])
    res_out = flag_gems._reshape_copy(inp, [24])

    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy_broadcast_input(dtype):
    # An expanded (zero-stride) input cannot be reshaped as a view; the copy must
    # still gather the logical elements in row-major order.
    base = torch.randn(1, 6, dtype=dtype, device=flag_gems.device)
    inp = base.expand(4, 6)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, [4, 6])
    res_out = flag_gems._reshape_copy(inp, [4, 6])

    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy_storage_offset(dtype):
    base = torch.randn(6, 4, dtype=dtype, device=flag_gems.device)
    inp = base[2:]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, [16])
    res_out = flag_gems._reshape_copy(inp, [16])

    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy_independent_storage(dtype):
    inp = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems._reshape_copy(inp, [4, 6])
    ref = utils.to_reference(res_out.clone())

    inp.add_(1.0)
    # The copy must not change when the source is mutated in place.
    utils.gems_assert_equal(res_out, ref)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy_zero_dim(dtype):
    inp = torch.tensor(3.0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._reshape_copy(ref_inp, [1])
    res_out = flag_gems._reshape_copy(inp, [1])

    assert list(res_out.shape) == [1]
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.reshape_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_reshape_copy_size_mismatch(dtype):
    # As in PyTorch, a size whose product differs from numel is rejected.
    inp = torch.randn(4, 6, dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._reshape_copy(inp, [3, 6])
    with pytest.raises(RuntimeError):
        flag_gems._reshape_copy(inp, [3, 6])

    with pytest.raises(RuntimeError):
        flag_gems._reshape_copy(inp, [-1, -1])
