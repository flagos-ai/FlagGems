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

_ATEN_OP = torch.ops.aten._foreach_round


def _gems_op(*args, **kwargs):
    return flag_gems.foreach_round(*args, **kwargs)


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


@pytest.mark.foreach_round
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=["contiguous", "noncontiguous"]
)
def test_foreach_round_accuracy(dtype, noncontiguous):
    values = [
        _sample((4, 7), dtype, noncontiguous=noncontiguous),
        _sample((3, 5), dtype, noncontiguous=noncontiguous),
    ]
    reference_values = _reference(values)

    expected = _ATEN_OP(reference_values)
    result = _gems_op(values)

    _assert_values(result, expected, _output_dtype(dtype))
    for output, source in zip(result, values):
        assert (
            output.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )


@pytest.mark.foreach_round
def test_foreach_round_mixed_shapes_and_empty_tensor():
    values = [
        _sample((17,), torch.float32),
        _sample((2, 3, 4), torch.float32),
        torch.empty(0, dtype=torch.float32, device=flag_gems.device),
    ]
    expected = _ATEN_OP(_reference(values))
    result = _gems_op(values)
    _assert_values(result, expected, torch.float32)


@pytest.mark.foreach_round
def test_foreach_round_integer_promotion():
    if flag_gems.vendor_name in ("kunlunxin", "mthreads") and not utils.TO_CPU:
        pytest.skip(
            "backend native foreach does not support the integer reference contract"
        )
    values = [_sample((11,), torch.int32)]
    reference_values = _reference(values)
    if not _INTEGER_SUPPORTED:
        with pytest.raises(RuntimeError):
            _ATEN_OP(reference_values)
        with pytest.raises(RuntimeError):
            _gems_op(values)
        return
    expected = _ATEN_OP(reference_values)
    result = _gems_op(values)
    _assert_values(result, expected, expected[0].dtype)


@pytest.mark.foreach_round
def test_foreach_round_complex_values():
    if flag_gems.vendor_name in ("kunlunxin", "enflame") and not utils.TO_CPU:
        pytest.skip(
            "backend native foreach does not support the complex reference contract"
        )
    values = [_sample((9,), torch.complex64)]
    reference_values = _reference(values)
    if not _COMPLEX_SUPPORTED:
        with pytest.raises(RuntimeError):
            _ATEN_OP(reference_values)
        with pytest.raises(RuntimeError):
            _gems_op(values)
        return
    expected = _ATEN_OP(reference_values)
    result = _gems_op(values)
    _assert_values(result, expected, _output_dtype(torch.complex64))


@pytest.mark.foreach_round
def test_foreach_round_rejects_empty_list():
    with pytest.raises(RuntimeError):
        _gems_op([])
