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

_ATEN_OP = torch.ops.aten._foreach_pow.ScalarList


def _candidate(*args):
    return flag_gems.foreach_pow_scalar_list(*args)


def _make_inputs(dtype):
    torch.manual_seed(0)
    shapes = [(16, 8), (7,), (2, 3, 4)]
    make = lambda: [
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
        for shape in shapes
    ]
    self_tensors = make()
    exponent = [0.5] * len(self_tensors)
    return self_tensors, exponent


@pytest.mark.foreach_pow_scalar_list
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_foreach_pow_scalar_list(dtype):
    self_tensors, exponent = _make_inputs(dtype)
    reference_self = [utils.to_reference(t.clone(), True) for t in self_tensors]
    reference_exponent = (
        [utils.to_reference(t, True) for t in exponent]
        if exponent
        and isinstance(exponent, list)
        and isinstance(exponent[0], torch.Tensor)
        else exponent
    )
    pointers = [t.data_ptr() for t in self_tensors]
    expected = torch.ops.aten._foreach_pow.ScalarList(
        reference_self, reference_exponent
    )
    result = _candidate(self_tensors, exponent)

    assert len(result) == len(expected)
    for actual, wanted, source in zip(result, expected, self_tensors):
        assert actual.shape == wanted.shape
        assert actual.dtype == source.dtype
        assert actual.device == source.device
        assert actual.data_ptr() != source.data_ptr()
        utils.gems_assert_close(actual, wanted, source.dtype)


@pytest.mark.foreach_pow_scalar_list
def test_foreach_pow_scalar_list_empty_list():
    with pytest.raises(RuntimeError):
        _ATEN_OP([], [])
    with pytest.raises(RuntimeError):
        _candidate([], [])


@pytest.mark.foreach_pow_scalar_list
def test_foreach_pow_scalar_list_bool_base():
    if flag_gems.vendor_name == "mthreads" and not utils.TO_CPU:
        pytest.skip("MUSA native pow does not support boolean-base ScalarList inputs")
    if flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU:
        pytest.skip(
            "Kunlunxin native pow does not support boolean-base ScalarList inputs"
        )
    tensors = [
        torch.tensor([True, True, False], dtype=torch.bool, device=flag_gems.device),
        torch.tensor([False, True], dtype=torch.bool, device=flag_gems.device),
    ]
    exponents = [0.5, 2]
    reference = [utils.to_reference(t) for t in tensors]

    expected = torch.ops.aten._foreach_pow.ScalarList(reference, exponents)
    result = _candidate(tensors, exponents)

    assert len(result) == len(expected)
    for actual, wanted in zip(result, expected):
        assert actual.dtype == wanted.dtype
        utils.gems_assert_equal(actual, wanted)
