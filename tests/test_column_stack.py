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


_SHAPE_CASES = [
    pytest.param([(8,), (8,)], id="1d-pair"),
    pytest.param([(1024,), (1024,), (1024,)], id="1d-three"),
    pytest.param([(16, 256), (16, 128)], id="2d-pair"),
    pytest.param([(64, 32), (64, 16), (64, 8)], id="2d-three"),
    pytest.param([(5,), (5, 2), (5, 2)], id="mixed-1d-2d"),
    pytest.param(
        [(20, 320, 15), (20, 160, 15), (20, 80, 15)],
        id="3d-three",
    ),
    pytest.param([(), ()], id="scalar-pair"),
    pytest.param([(0,), (0,)], id="zero-size"),
]
_INVALID_SHAPES = [
    pytest.param([(16, 256), (8, 128)], id="mismatched-rows"),
    pytest.param([(5,), (6,)], id="mismatched-1d"),
    pytest.param([(5,), (5, 1, 1)], id="mismatched-rank"),
]


def _make_tensor(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(0, 0x7FFF, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )


def _reference_inputs(inputs, dtype):
    return [
        utils.to_reference(tensor, True)
        if dtype in utils.FLOAT_DTYPES
        else utils.to_reference(tensor)
        for tensor in inputs
    ]


def _assert_result(result, reference, dtype):
    assert result.shape == reference.shape
    assert result.dtype == dtype
    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(result, reference, dtype)
    else:
        utils.gems_assert_equal(result, reference)


@pytest.mark.column_stack
@pytest.mark.parametrize("shapes", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_column_stack(shapes, dtype):
    torch.manual_seed(0)
    inputs = [_make_tensor(shape, dtype) for shape in shapes]
    reference = torch.column_stack(_reference_inputs(inputs, dtype))

    result = flag_gems.column_stack(inputs)

    _assert_result(result, reference, dtype)


@pytest.mark.column_stack
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_column_stack_noncontiguous(dtype):
    torch.manual_seed(0)
    inputs = [
        torch.randn((4, 6), dtype=dtype, device=flag_gems.device).T,
        torch.randn((2, 6), dtype=dtype, device=flag_gems.device).T,
    ]
    assert all(not tensor.is_contiguous() for tensor in inputs)
    reference = torch.column_stack(_reference_inputs(inputs, dtype))

    result = flag_gems.column_stack(inputs)

    _assert_result(result, reference, dtype)


@pytest.mark.column_stack
def test_column_stack_mixed_dtype_and_tuple_input():
    torch.manual_seed(0)
    inputs = (
        torch.randn((5,), dtype=torch.float32, device=flag_gems.device),
        torch.randint(0, 17, (5,), dtype=torch.int32, device="cpu").to(
            flag_gems.device
        ),
    )
    reference_inputs = tuple(utils.to_reference(tensor) for tensor in inputs)
    reference = torch.column_stack(reference_inputs)

    result = flag_gems.column_stack(inputs)

    assert result.dtype == reference.dtype
    utils.gems_assert_equal(result, reference)


@pytest.mark.column_stack
@pytest.mark.parametrize("shapes", _INVALID_SHAPES)
def test_column_stack_rejects_incompatible_shapes(shapes):
    torch.manual_seed(0)
    inputs = [_make_tensor(shape, torch.float32) for shape in shapes]

    with pytest.raises(RuntimeError):
        flag_gems.column_stack(inputs)


@pytest.mark.column_stack
def test_column_stack_rejects_empty_tensor_list():
    with pytest.raises(RuntimeError):
        flag_gems.column_stack([])
