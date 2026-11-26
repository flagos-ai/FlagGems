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
from decimal import Decimal

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.nan_to_num
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("nan", [None, 0.0, 2.3])
@pytest.mark.parametrize("posinf", [None, 999.0])
@pytest.mark.parametrize("neginf", [None, -999.0])
def test_nan_to_num(shape, dtype, nan, posinf, neginf):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base.view(-1)[0] = float("nan")
    if base.numel() > 1:
        base.view(-1)[1] = float("inf")
    if base.numel() > 2:
        base.view(-1)[2] = float("-inf")

    ref_input = utils.to_reference(base)
    ref_out = torch.nan_to_num(ref_input, nan=nan, posinf=posinf, neginf=neginf)

    res_out = flag_gems.nan_to_num(base, nan=nan, posinf=posinf, neginf=neginf)

    utils.gems_assert_equal(res_out, ref_out)


_HYGON_SHAPES = [(), (1,), (511,), (512,), (513,), (65536,), (65537,), (1048577,)]
_HYGON_REPLACEMENTS = [
    (None, None, None),
    (2.3, 999.0, -999.0),
    (-0.0, -0.0, 0.0),
    (float("inf"), -float("inf"), float("nan")),
    (torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max, 1),
    (2.0**-149, -(2.0**-149), -0.0),
]


def _hygon_nan_to_num_input(shape):
    # Initialize every bit, including NaN payloads, signed zeros, subnormals,
    # normal boundaries and maximum finite FP32 values.
    bits = [
        0x7FC00000,
        0xFFC00001,
        0x7F800000,
        0xFF800000,
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x3F000000,
        0xBF000000,
    ]
    values = torch.tensor(
        [b if b < 2**31 else b - 2**32 for b in bits], dtype=torch.int32
    ).view(torch.float32)
    size = 1
    for extent in shape:
        size *= extent
    return (
        values.repeat((size + len(bits) - 1) // len(bits))[:size].clone().reshape(shape)
    )


def _assert_hygon_nan_to_num_bits(result, reference):
    result = result.cpu()
    utils.gems_assert_equal(result, reference, equal_nan=True)
    not_nan = ~torch.isnan(reference)
    utils.gems_assert_equal(
        result.view(torch.int32)[not_nan], reference.view(torch.int32)[not_nan]
    )
    zeros = reference == 0
    utils.gems_assert_equal(
        torch.signbit(result[zeros]), torch.signbit(reference[zeros])
    )


@pytest.mark.nan_to_num
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon launch and scalar coverage"
)
@pytest.mark.parametrize("shape", _HYGON_SHAPES)
@pytest.mark.parametrize("replacements", _HYGON_REPLACEMENTS)
def test_nan_to_num_hygon_special_values(shape, replacements):
    cpu_inp = _hygon_nan_to_num_input(shape)
    inp = cpu_inp.to(device=flag_gems.device)
    provider = inspect.getmodule(flag_gems.nan_to_num)
    assert provider._can_use_contiguous_path(inp)
    assert provider._normalize_replacements(*replacements) is not None
    result = flag_gems.nan_to_num(inp, *replacements)
    reference = torch.nan_to_num(cpu_inp, *replacements)
    _assert_hygon_nan_to_num_bits(result, reference)
    assert result.shape == inp.shape and result.dtype == inp.dtype
    assert result.device == inp.device and result.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(inp.cpu().view(torch.int32), cpu_inp.view(torch.int32))


@pytest.mark.nan_to_num
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon offset coverage")
def test_nan_to_num_hygon_storage_offset():
    cpu_base = _hygon_nan_to_num_input((514,))
    base = cpu_base.to(device=flag_gems.device)
    inp = base[1:513]
    provider = inspect.getmodule(flag_gems.nan_to_num)
    assert inp.storage_offset() == 1 and provider._can_use_contiguous_path(inp)
    result = flag_gems.nan_to_num(
        inp, nan=-0.0, posinf=-float("inf"), neginf=float("nan")
    )
    reference = torch.nan_to_num(
        cpu_base[1:513], nan=-0.0, posinf=-float("inf"), neginf=float("nan")
    )
    _assert_hygon_nan_to_num_bits(result, reference)
    utils.gems_assert_equal(base.cpu().view(torch.int32), cpu_base.view(torch.int32))


@pytest.mark.nan_to_num
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon tensor fallback coverage"
)
@pytest.mark.parametrize(
    "kind", ["transpose", "slice", "fp16", "bf16", "empty", "requires_grad"]
)
def test_nan_to_num_hygon_fallback(kind, monkeypatch):
    cpu_inp = _hygon_nan_to_num_input((6, 8))
    inp = cpu_inp.to(device=flag_gems.device)
    if kind == "transpose":
        inp, cpu_inp = inp.T, cpu_inp.T
    elif kind == "slice":
        inp, cpu_inp = inp[:, ::2], cpu_inp[:, ::2]
    elif kind in ("fp16", "bf16"):
        dtype = torch.float16 if kind == "fp16" else torch.bfloat16
        inp, cpu_inp = inp.to(dtype=dtype), cpu_inp.to(dtype=dtype)
    elif kind == "empty":
        inp, cpu_inp = inp[:0], cpu_inp[:0]
    else:
        inp.requires_grad_(True)
    provider = inspect.getmodule(flag_gems.nan_to_num)
    assert not provider._can_use_contiguous_path(inp)
    generic = provider._generic_nan_to_num
    calls = []

    def delegate(A, **kwargs):
        calls.append((A, kwargs))
        return generic(A, **kwargs)

    monkeypatch.setattr(provider, "_generic_nan_to_num", delegate)
    result = flag_gems.nan_to_num(inp, nan=-0.0, posinf=123.0, neginf=-321.0)
    assert len(calls) == 1 and calls[0][0] is inp
    assert calls[0][1] == {"nan": -0.0, "posinf": 123.0, "neginf": -321.0}
    reference = torch.nan_to_num(cpu_inp, nan=-0.0, posinf=123.0, neginf=-321.0)
    utils.gems_assert_equal(result.cpu(), reference, equal_nan=True)
    utils.gems_assert_equal(inp.cpu(), cpu_inp, equal_nan=True)


@pytest.mark.nan_to_num
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon scalar guard coverage"
)
@pytest.mark.parametrize("name", ["nan", "posinf", "neginf"])
@pytest.mark.parametrize("value", [True, Decimal("1.25"), 1.0e39, -1.0e39, 10**1000])
def test_nan_to_num_hygon_scalar_delegates(name, value, monkeypatch):
    provider = inspect.getmodule(flag_gems.nan_to_num)
    inp = torch.ones((512,), dtype=torch.float32, device=flag_gems.device)
    kwargs = {"nan": None, "posinf": None, "neginf": None}
    kwargs[name] = value
    calls = []

    def delegate(A, **received):
        calls.append((A, received))
        return torch.empty_like(A)

    monkeypatch.setattr(provider, "_generic_nan_to_num", delegate)
    assert provider._normalize_replacements(**kwargs) is None
    flag_gems.nan_to_num(inp, **kwargs)
    assert len(calls) == 1 and calls[0][0] is inp
    assert calls[0][1][name] is value


@pytest.mark.nan_to_num
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon AD and view guards")
@pytest.mark.parametrize("kind", ["negative_view", "forward_ad"])
def test_nan_to_num_hygon_metadata_delegates(kind, monkeypatch):
    provider = inspect.getmodule(flag_gems.nan_to_num)
    primal = torch.ones((512,), dtype=torch.float32, device=flag_gems.device)
    calls = []

    def delegate(A, **kwargs):
        calls.append(A)
        return torch.empty_like(A)

    monkeypatch.setattr(provider, "_generic_nan_to_num", delegate)
    with torch.autograd.forward_ad.dual_level():
        inp = (
            torch._neg_view(primal)
            if kind == "negative_view"
            else torch.autograd.forward_ad.make_dual(primal, torch.ones_like(primal))
        )
        assert not provider._can_use_contiguous_path(inp)
        flag_gems.nan_to_num(inp)
        assert len(calls) == 1 and calls[0] is inp
