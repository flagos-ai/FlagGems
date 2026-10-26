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


def _assert_out_result(result, out, original_pointer, reference, dtype):
    assert result is out
    assert result.data_ptr() == original_pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(result, reference, dtype)
    else:
        utils.gems_assert_equal(result, reference)


@pytest.mark.column_stack_out
@pytest.mark.parametrize("shapes", _SHAPE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_column_stack_out(shapes, dtype):
    torch.manual_seed(0)
    inputs = [_make_tensor(shape, dtype) for shape in shapes]
    reference = torch.column_stack(_reference_inputs(inputs, dtype))
    out = torch.empty(reference.shape, dtype=dtype, device=flag_gems.device)
    original_pointer = out.data_ptr()

    result = flag_gems.column_stack_out(inputs, out=out)

    _assert_out_result(result, out, original_pointer, reference, dtype)


@pytest.mark.column_stack_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_column_stack_out_resizes_empty_buffer(dtype):
    torch.manual_seed(0)
    inputs = [_make_tensor((5,), dtype), _make_tensor((5, 2), dtype)]
    reference = torch.column_stack(_reference_inputs(inputs, dtype))
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)

    result = flag_gems.column_stack_out(inputs, out=out)

    assert result is out
    assert result.shape == reference.shape
    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(result, reference, dtype)
    else:
        utils.gems_assert_equal(result, reference)


@pytest.mark.column_stack_out
def test_column_stack_out_accepts_safe_dtype_cast():
    torch.manual_seed(0)
    inputs = [
        _make_tensor((5,), torch.float16),
        _make_tensor((5,), torch.float16),
    ]
    reference_inputs = [utils.to_reference(tensor) for tensor in inputs]
    reference_out = torch.empty(
        (5, 2), dtype=torch.float32, device=reference_inputs[0].device
    )
    reference_result = torch.column_stack(reference_inputs, out=reference_out)
    assert reference_result is reference_out

    out = torch.empty((5, 2), dtype=torch.float32, device=flag_gems.device)
    original_pointer = out.data_ptr()
    result = flag_gems.column_stack_out(inputs, out=out)

    assert result is out
    assert result.data_ptr() == original_pointer
    assert result.device == inputs[0].device
    assert result.dtype == torch.float32
    utils.gems_assert_equal(result, reference_out)


@pytest.mark.column_stack_out
def test_column_stack_out_rejects_unsafe_dtype_cast():
    if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU:
        pytest.skip("Kunlunxin native column_stack accepts unsafe out dtype casts")
    torch.manual_seed(0)
    inputs = [
        _make_tensor((5,), torch.float32),
        _make_tensor((5,), torch.float32),
    ]
    reference_inputs = [utils.to_reference(tensor) for tensor in inputs]
    reference_out = torch.empty(
        (5, 2), dtype=torch.int32, device=reference_inputs[0].device
    )
    with pytest.raises(Exception) as reference_error:
        torch.column_stack(reference_inputs, out=reference_out)

    out = torch.empty((5, 2), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(type(reference_error.value)):
        flag_gems.column_stack_out(inputs, out=out)
