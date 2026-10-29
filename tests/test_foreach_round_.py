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

_ATEN_OP = torch.ops.aten._foreach_round_


def _gems_op(*args, **kwargs):
    return flag_gems.foreach_round_(*args, **kwargs)


_FAMILY = "round"
_INTEGER_SUPPORTED = True
_COMPLEX_SUPPORTED = False
_SAME_DTYPE_INTEGER = True


def _sample(shape, dtype, *, noncontiguous=False):
    torch.manual_seed(0)
    source_shape = (
        tuple(reversed(shape)) if noncontiguous and len(shape) == 2 else shape
    )
    if dtype.is_complex:
        real = torch.rand(source_shape, dtype=torch.float32, device=flag_gems.device)
        imag = torch.rand(source_shape, dtype=torch.float32, device=flag_gems.device)
        value = torch.complex(real, imag).to(dtype)
    elif dtype.is_floating_point:
        value = torch.rand(source_shape, dtype=dtype, device=flag_gems.device)
    else:
        value = torch.randint(1, 5, source_shape, dtype=dtype, device=flag_gems.device)
    if dtype.is_floating_point:
        value.mul_(0.8).add_(0.1)
    if noncontiguous and len(shape) == 2:
        value = value.transpose(0, 1)
    return value


def _reference(values):
    references = []
    for value in values:
        if value.dtype.is_floating_point or value.dtype.is_complex:
            references.append(utils.to_reference(value.clone(), True))
        else:
            references.append(utils.to_reference(value.clone()))
    return references


def _output_dtype(dtype):
    if _FAMILY == "abs" and dtype == torch.complex64:
        return torch.float32
    if _FAMILY == "abs" and dtype == torch.complex128:
        return torch.float64
    return dtype


def _assert_values(actual, expected, expected_dtype):
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == expected_dtype
        if result.dtype.is_floating_point or result.dtype.is_complex:
            utils.gems_assert_close(result, reference, result.dtype)
        else:
            utils.gems_assert_equal(result, reference)


@pytest.mark.inplace
@pytest.mark.foreach_round_
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=["contiguous", "noncontiguous"]
)
def test_foreach_round__accuracy(dtype, noncontiguous):
    values = [
        _sample((4, 7), dtype, noncontiguous=noncontiguous),
        _sample((3, 5), dtype, noncontiguous=noncontiguous),
    ]
    reference_values = _reference(values)
    pointers = [value.untyped_storage().data_ptr() for value in values]
    strides = [value.stride() for value in values]

    expected_result = _ATEN_OP(reference_values)
    result = _gems_op(values)

    assert expected_result is None
    assert result is None
    _assert_values(values, reference_values, dtype)
    assert [value.untyped_storage().data_ptr() for value in values] == pointers
    assert [value.stride() for value in values] == strides


@pytest.mark.inplace
@pytest.mark.foreach_round_
def test_foreach_round__writes_through_noncontiguous_view():
    storage = _sample((8, 16), torch.float32)
    reference_storage = utils.to_reference(storage.clone(), True)
    view = storage[:, ::2]
    reference_view = reference_storage[:, ::2]
    pointer = storage.untyped_storage().data_ptr()

    assert _ATEN_OP([reference_view]) is None
    result = _gems_op([view])
    assert result is None

    utils.gems_assert_close(storage, reference_storage, storage.dtype)
    assert storage.untyped_storage().data_ptr() == pointer


@pytest.mark.foreach_round_
def test_foreach_round__integer_contract():
    if flag_gems.vendor_name in ("kunlunxin", "mthreads") and not utils.TO_CPU:
        pytest.skip(
            "backend native foreach does not support the integer reference contract"
        )
    values = [_sample((11,), torch.int32)]
    reference_values = _reference(values)
    if not _INTEGER_SUPPORTED or not _SAME_DTYPE_INTEGER:
        with pytest.raises(RuntimeError):
            _ATEN_OP(reference_values)
        with pytest.raises(RuntimeError):
            _gems_op(values)
        return

    assert _ATEN_OP(reference_values) is None
    result = _gems_op(values)
    assert result is None
    _assert_values(values, reference_values, torch.int32)


@pytest.mark.foreach_round_
def test_foreach_round__rejects_empty_list():
    with pytest.raises(RuntimeError):
        _gems_op([])
