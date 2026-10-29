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

from contextlib import contextmanager

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_EQUAL_SHAPES = [*utils.POINTWISE_SHAPES, (257,), (2, 0, 3)]
_BROADCAST_CASES = [
    pytest.param((7, 13), (1, 13), id="leading-dimension"),
    pytest.param((2, 3, 5), (1, 3, 1), id="multiple-dimensions"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]


def _is_suiyuan_double(inp):
    return flag_gems.vendor_name == "enflame" and inp.dtype == torch.float64


def _suiyuan_double_cpu(inp):
    # GCU typed Double copies can silently narrow. Raw storage preserves all bits.
    storage = inp.untyped_storage().cpu()
    result = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        storage, inp.storage_offset(), tuple(inp.shape), inp.stride()
    )
    if inp.is_neg():
        result = result.neg()
    return result.clone()


def _reference_input(inp, *, clone=False):
    if _is_suiyuan_double(inp) and utils.TO_CPU:
        return _suiyuan_double_cpu(inp)
    return utils.to_reference(inp.clone() if clone else inp, True)


def _readonly_snapshot(inp):
    # Readonly means unchanged physical bytes, including NaN payloads and signed zero.
    storage = inp.untyped_storage().cpu()
    return (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .clone()
    )


def _assert_readonly_value(inp, snapshot):
    assert torch.equal(_readonly_snapshot(inp), snapshot)


def _comparison_result(result):
    # Call only after target-device/dtype/shape and return-identity assertions.
    return (
        _suiyuan_double_cpu(result)
        if utils.TO_CPU and _is_suiyuan_double(result)
        else result
    )


def _input_metadata(inp):
    storage = inp.untyped_storage()
    return (
        inp.device,
        inp.dtype,
        inp.shape,
        inp.stride(),
        inp.storage_offset(),
        storage.data_ptr(),
        storage._cdata,
        storage.nbytes(),
        inp.requires_grad,
        inp.is_conj(),
        inp.is_neg(),
    )


@contextmanager
def _check_inputs(inp, other, *, self_readonly=False):
    original_metadata = (_input_metadata(inp), _input_metadata(other))
    other_before = _readonly_snapshot(other)
    inp_before = _readonly_snapshot(inp) if self_readonly else None
    try:
        yield
    finally:
        assert _input_metadata(inp) == original_metadata[0]
        assert _input_metadata(other) == original_metadata[1]
        _assert_readonly_value(other, other_before)
        if self_readonly:
            _assert_readonly_value(inp, inp_before)


def _make_positive(shape, dtype):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        if not utils.TO_CPU and all(size != 0 for size in shape):
            pytest.skip(
                "Enflame native igamma_ does not compute nonempty Double values"
            )
        # Preserve the caller's seed and the independent sequential input draws.
        value = torch.empty(shape, dtype=dtype, device="cpu").uniform_(1.0, 3.0)
        return (
            value.reshape(-1)
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(dtype)
            .reshape(shape)
        )
    return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 3.0)


def _assert_inplace_result(result, mutated_input, original_pointer, reference, dtype):
    assert result is mutated_input
    assert result.device == mutated_input.device
    assert result.data_ptr() == original_pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_close(
        _comparison_result(result), reference, dtype, equal_nan=True, atol=1e-3
    )


@pytest.mark.inplace
@pytest.mark.igamma_
@pytest.mark.parametrize("shape", _EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_equal_shape_(shape, dtype):
    torch.manual_seed(0)
    inp = _make_positive(shape, dtype)
    other = _make_positive(shape, dtype)
    original_pointer = inp.data_ptr()

    ref_inp = _reference_input(inp, clone=True)
    ref_other = _reference_input(other)
    ref_out = ref_inp.igamma_(ref_other)

    with _check_inputs(inp, other):
        result = flag_gems.igamma_(inp, other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.igamma_
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_broadcast_(self_shape, other_shape, dtype):
    torch.manual_seed(0)
    inp = _make_positive(self_shape, dtype)
    other = _make_positive(other_shape, dtype)
    original_pointer = inp.data_ptr()
    ref_inp = _reference_input(inp, clone=True)
    ref_other = _reference_input(other)
    ref_out = ref_inp.igamma_(ref_other)

    with _check_inputs(inp, other):
        result = flag_gems.igamma_(inp, other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.igamma_
def test_igamma_noncontiguous_():
    torch.manual_seed(0)
    inp = _make_positive((4, 7, 9), torch.float32).transpose(0, 2)
    other = _make_positive(inp.shape, torch.float32)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    assert not inp.is_contiguous()

    ref_inp = _reference_input(inp, clone=True)
    ref_other = _reference_input(other)
    ref_out = ref_inp.igamma_(ref_other)

    with _check_inputs(inp, other):
        result = flag_gems.igamma_(inp, other)

    assert result.stride() == original_stride
    _assert_inplace_result(result, inp, original_pointer, ref_out, torch.float32)


@pytest.mark.inplace
@pytest.mark.igamma_
def test_igamma_boundary_values_():
    inp = torch.tensor(
        [0.5, 1.0, 2.0, 4.0], dtype=torch.float32, device=flag_gems.device
    )
    other = torch.tensor(
        [0.0, float("inf"), 1.0, 8.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    original_pointer = inp.data_ptr()
    ref_inp = _reference_input(inp, clone=True)
    ref_other = _reference_input(other)
    ref_out = ref_inp.igamma_(ref_other)

    with _check_inputs(inp, other):
        result = flag_gems.igamma_(inp, other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, torch.float32)


@pytest.mark.inplace
@pytest.mark.igamma_
def test_igamma_rejects_nonbroadcastable_shape_():
    inp = torch.ones((2, 3), device=flag_gems.device)
    other = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with _check_inputs(inp, other, self_readonly=True):
            flag_gems.igamma_(inp, other)
