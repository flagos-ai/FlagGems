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

import importlib
import math
import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

COMPLEX_TO_REAL = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}
COMPLEX_DTYPES = [
    torch.complex32,
    torch.complex64,
    pytest.param(
        torch.complex128,
        marks=pytest.mark.skipif(
            not utils.fp64_is_supported,
            reason="complex128 requires device fp64 support",
        ),
    ),
]
NON_COMPLEX_DTYPES = (
    [torch.bool, torch.uint8, torch.int8]
    + utils.ALL_INT_DTYPES
    + utils.ALL_FLOAT_DTYPES
)
FP8_DTYPES = [
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
    if hasattr(torch, name)
]


def _make_complex_base(shape, dtype, device, requires_grad=False):
    real_dtype = COMPLEX_TO_REAL[dtype]
    real_storage = torch.arange(
        math.prod(shape) * 2,
        dtype=real_dtype,
        device=device,
    ).reshape((*shape, 2))
    return torch.view_as_complex(real_storage).detach().requires_grad_(requires_grad)


def _assert_real_view(result, input, expected):
    utils.gems_assert_equal(result, expected)
    assert result.dtype == COMPLEX_TO_REAL[input.dtype]
    assert result.shape == input.shape
    assert result.stride() == tuple(stride * 2 for stride in input.stride())
    assert result.storage_offset() == input.storage_offset() * 2
    assert result.data_ptr() == input.data_ptr()
    assert result.untyped_storage().data_ptr() == input.untyped_storage().data_ptr()
    assert torch._C._is_alias_of(result, input)
    assert result._is_view()
    assert not result.is_conj()


@pytest.mark.real
@pytest.mark.parametrize("is_conj", [False, True])
def test_real_uses_flag_gems_view_path(monkeypatch, is_conj):
    real_module = importlib.import_module("flag_gems.ops.real")
    view_as_real = real_module._VIEW_AS_REAL
    seen_sources = []

    def track_view_as_real(source):
        seen_sources.append(source)
        return view_as_real(source)

    monkeypatch.setattr(real_module, "_VIEW_AS_REAL", track_view_as_real)

    base = _make_complex_base(
        (3, 5), torch.complex64, flag_gems.device, requires_grad=True
    )
    input = base.conj() if is_conj else base
    expected = torch.real(utils.to_reference(input))

    with flag_gems.use_gems(include=["real"]):
        assert flag_gems.all_registered_ops() == ["real"]
        assert flag_gems.all_registered_keys() == ["real"]
        result = torch.real(input)

    assert len(seen_sources) == 1
    assert torch._C._is_alias_of(seen_sources[0], input)
    assert not seen_sources[0].is_conj()
    _assert_real_view(result, input, expected)


@pytest.mark.real
@pytest.mark.parametrize("shape", [(), (0,), (3, 5), (2, 3, 4)])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_real_complex_view_semantics(shape, dtype):
    input = _make_complex_base(shape, dtype, flag_gems.device)
    ref_input = utils.to_reference(input)
    expected = torch.real(ref_input)

    with flag_gems.use_gems():
        result = torch.real(input)

    _assert_real_view(result, input, expected)


@pytest.mark.real
@pytest.mark.parametrize("is_conj", [False, True])
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_real_noncontiguous_complex_view(dtype, is_conj):
    base = _make_complex_base((9, 11), dtype, flag_gems.device)
    ref_base = utils.to_reference(base)
    input = base[1:9:2, 2:11:3]
    ref_input = ref_base[1:9:2, 2:11:3]
    if is_conj:
        input = input.conj()
        ref_input = ref_input.conj()

    expected = torch.real(ref_input)
    with flag_gems.use_gems():
        result = torch.real(input)

    _assert_real_view(result, input, expected)
    assert input.is_conj() == is_conj


@pytest.mark.real
@pytest.mark.parametrize("dtype", NON_COMPLEX_DTYPES)
def test_real_non_complex_returns_input(dtype):
    base = torch.empty((7, 9), dtype=dtype, device=flag_gems.device)
    input = base[1:7:2, 2:9:3]

    with flag_gems.use_gems():
        result = torch.real(input)

    assert result is input
    assert result.data_ptr() == input.data_ptr()
    assert result.untyped_storage().data_ptr() == input.untyped_storage().data_ptr()
    assert result.shape == input.shape
    assert result.stride() == input.stride()
    assert result.storage_offset() == input.storage_offset()


@pytest.mark.real
@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_real_float8_returns_input(dtype):
    try:
        base = torch.empty((7, 9), dtype=dtype, device=flag_gems.device)
    except (RuntimeError, TypeError) as error:
        pytest.skip(f"device cannot allocate {dtype}: {error}")
    input = base[1:7:2, 2:9:3]

    with flag_gems.use_gems():
        result = torch.real(input)

    assert result is input
    assert result.data_ptr() == input.data_ptr()
    assert result.untyped_storage().data_ptr() == input.untyped_storage().data_ptr()
    assert result.stride() == input.stride()
    assert result.storage_offset() == input.storage_offset()


@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend" and not utils.TO_CPU,
    reason="Ascend native torch.isclose does not support complex64; run with --ref cpu.",
)
@pytest.mark.real
@pytest.mark.parametrize("is_conj", [False, True])
def test_real_autograd_view_relation(is_conj):
    base = _make_complex_base(
        (7, 9), torch.complex64, flag_gems.device, requires_grad=True
    )
    ref_base = utils.to_reference(base.detach().clone()).requires_grad_()
    input = base[1:7:2, 1:9:2]
    ref_input = ref_base[1:7:2, 1:9:2]
    if is_conj:
        input = input.conj()
        ref_input = ref_input.conj()

    expected = torch.real(ref_input)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with flag_gems.use_gems():
            result = torch.real(input)
        result_grad = torch.arange(
            result.numel(), dtype=result.dtype, device=result.device
        ).reshape(result.shape)
        (input_grad,) = torch.autograd.grad(result, base, result_grad)

    ref_result_grad = utils.to_reference(result_grad)
    (expected_grad,) = torch.autograd.grad(expected, ref_base, ref_result_grad)

    _assert_real_view(result, input, expected)
    utils.gems_assert_equal(input_grad, expected_grad)
    assert result.grad_fn is not None
    assert result._base is not None
    assert not any(
        "aten::real: an autograd kernel was not registered" in str(item.message)
        for item in caught
    )


@pytest.mark.real
def test_real_conjugate_view_version_tracking():
    leaf = _make_complex_base(
        (4,), torch.complex64, flag_gems.device, requires_grad=True
    )
    input = leaf.conj()

    with flag_gems.use_gems():
        result = torch.real(input)

    loss = (result * result).sum()

    with torch.no_grad():
        torch.view_as_real(leaf).select(-1, 0).add_(1)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend" and not utils.TO_CPU,
    reason="Ascend native torch.isclose does not support complex64; run with --ref cpu.",
)
@pytest.mark.real
def test_real_conjugate_nonleaf_inplace_view_replay():
    leaf = _make_complex_base(
        (4,), torch.complex64, flag_gems.device, requires_grad=True
    )
    base = leaf * 2
    input = base.conj()

    with flag_gems.use_gems():
        result = torch.real(input)

    result.add_(3)
    result.sum().backward()

    expected_grad = utils.to_reference(torch.full_like(leaf, 2))
    utils.gems_assert_equal(leaf.grad, expected_grad)
