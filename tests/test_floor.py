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

import inspect

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.floor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.floor(ref_inp)
    res_out = flag_gems.floor(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.floor_()
    res_out = flag_gems.floor_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.floor(ref_inp, out=ref_out)
    flag_gems.floor_out(inp, out=out)

    utils.gems_assert_equal(out, ref_out)


_HYGON_FLOOR_SHAPES = [(), (1,), (1023,), (1024,), (1025,), (32, 32), (1048577,)]


def _hygon_floor_input(shape):
    # Include values whose floor changes if subnormals are flushed to zero.
    values = torch.tensor(
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            0.0,
            -0.0,
            2.0**-149,
            -(2.0**-149),
            -0.25,
            0.25,
            -1.75,
            1.75,
            2.0**24,
            -(2.0**24),
        ],
        dtype=torch.float32,
    )
    size = 1
    for extent in shape:
        size *= extent
    return values.repeat((size + values.numel() - 1) // values.numel())[:size].reshape(
        shape
    )


def _assert_floor_special_bits(result, reference):
    result = result.cpu()
    zeros = reference == 0
    utils.gems_assert_equal(
        torch.signbit(result[zeros]), torch.signbit(reference[zeros])
    )
    # NaN payload/sign propagation is outside floor's value contract. Every
    # non-NaN output, including infinities and signed zeros, must match bitwise.
    not_nan = ~torch.isnan(reference)
    utils.gems_assert_equal(
        result.view(torch.int32)[not_nan], reference.view(torch.int32)[not_nan]
    )


@pytest.mark.floor
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon launch coverage")
@pytest.mark.parametrize("shape", _HYGON_FLOOR_SHAPES)
def test_floor_hygon_contiguous_special_values(shape):
    cpu_inp = _hygon_floor_input(shape)
    inp = cpu_inp.to(device=flag_gems.device)
    provider = inspect.getmodule(flag_gems.floor)
    assert provider._can_use_contiguous_path(inp)

    result = flag_gems.floor(inp)
    reference = torch.floor(cpu_inp)
    utils.gems_assert_equal(result.cpu(), reference, equal_nan=True)
    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    assert result.device == inp.device
    assert result.data_ptr() != inp.data_ptr()
    _assert_floor_special_bits(result, reference)
    utils.gems_assert_equal(inp.cpu(), cpu_inp, equal_nan=True)
    utils.gems_assert_equal(inp.cpu().view(torch.int32), cpu_inp.view(torch.int32))


@pytest.mark.floor
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon offset coverage")
def test_floor_hygon_contiguous_storage_offset():
    cpu_base = _hygon_floor_input((1026,))
    base = cpu_base.to(device=flag_gems.device)
    inp = base[1:1025]
    assert inp.storage_offset() == 1
    assert inspect.getmodule(flag_gems.floor)._can_use_contiguous_path(inp)

    result = flag_gems.floor(inp)
    reference = torch.floor(cpu_base[1:1025])
    utils.gems_assert_equal(result.cpu(), reference, equal_nan=True)
    utils.gems_assert_equal(base.cpu(), cpu_base, equal_nan=True)
    utils.gems_assert_equal(base.cpu().view(torch.int32), cpu_base.view(torch.int32))
    _assert_floor_special_bits(result, reference)


@pytest.mark.floor
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon fallback coverage")
@pytest.mark.parametrize(
    "kind",
    ["transpose", "slice", "fp16", "bf16", "empty", "requires_grad"],
)
def test_floor_hygon_fallback(kind, monkeypatch):
    cpu_inp = torch.linspace(-3.75, 3.75, 48, dtype=torch.float32).reshape(6, 8)
    inp = cpu_inp.to(device=flag_gems.device)
    if kind == "transpose":
        inp = inp.T
        cpu_inp = cpu_inp.T
    elif kind == "slice":
        inp = inp[:, ::2]
        cpu_inp = cpu_inp[:, ::2]
    elif kind in ("fp16", "bf16"):
        dtype = torch.float16 if kind == "fp16" else torch.bfloat16
        inp = inp.to(dtype=dtype)
        cpu_inp = cpu_inp.to(dtype=dtype)
    elif kind == "empty":
        inp = inp[:0]
        cpu_inp = cpu_inp[:0]
    elif kind == "requires_grad":
        inp.requires_grad_(True)

    provider = inspect.getmodule(flag_gems.floor)
    assert not provider._can_use_contiguous_path(inp)
    generic = provider._generic_floor
    calls = []

    def delegate(A):
        calls.append(A)
        return generic(A)

    monkeypatch.setattr(provider, "_generic_floor", delegate)
    result = flag_gems.floor(inp)
    assert len(calls) == 1 and calls[0] is inp
    utils.gems_assert_equal(result.cpu(), torch.floor(cpu_inp))
    utils.gems_assert_equal(inp.cpu(), cpu_inp)


@pytest.mark.floor
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon forward AD guard")
def test_floor_hygon_forward_ad_delegates(monkeypatch):
    provider = inspect.getmodule(flag_gems.floor)
    inp = torch.ones((1024,), dtype=torch.float32, device=flag_gems.device)
    calls = []

    def delegate(A):
        calls.append(A)
        return torch.empty_like(A)

    monkeypatch.setattr(provider, "_generic_floor", delegate)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(inp, torch.ones_like(inp))
        assert not provider._can_use_contiguous_path(dual)
        flag_gems.floor(dual)
        assert len(calls) == 1 and calls[0] is dual


@pytest.mark.floor
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon view guard")
def test_floor_hygon_negative_view_delegates(monkeypatch):
    provider = inspect.getmodule(flag_gems.floor)
    inp = torch._neg_view(
        torch.ones((1024,), dtype=torch.float32, device=flag_gems.device)
    )
    calls = []

    def delegate(A):
        calls.append(A)
        return torch.empty_like(A)

    monkeypatch.setattr(provider, "_generic_floor", delegate)
    assert not provider._can_use_contiguous_path(inp)
    flag_gems.floor(inp)
    assert len(calls) == 1 and calls[0] is inp
