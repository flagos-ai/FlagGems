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

import pytest
import torch

import flag_gems

hygon_op = None
if flag_gems.vendor_name == "hygon":
    hygon_op = importlib.import_module(flag_gems.masked_fill.__module__)

pytestmark = [
    pytest.mark.masked_fill,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon fast path"),
]


def _inputs(shape, device, dtype=torch.float32):
    n = 1
    for extent in shape:
        n *= extent
    x = (torch.arange(n, dtype=torch.float32, device=device) % 31 - 15).reshape(shape)
    mask = (torch.arange(n, device=device) % 3 == 0).reshape(shape)
    return x.to(dtype), mask


def _assert_result(inp, mask, value):
    # The reference is always CPU ATen, outside use_gems.
    ref_value = value.cpu() if isinstance(value, torch.Tensor) else value
    expected = torch.masked_fill(inp.cpu(), mask.cpu(), ref_value)
    before = inp.clone()
    actual = flag_gems.masked_fill(inp, mask, value)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(inp, before, rtol=0, atol=0, equal_nan=True)
    assert actual.shape == expected.shape
    assert actual.dtype == inp.dtype
    assert actual.device == inp.device
    if inp.numel():
        assert actual.data_ptr() != inp.data_ptr()
    # Equality alone does not distinguish -0 from +0.
    if actual.is_floating_point():
        zeros = expected == 0
        assert torch.equal(
            torch.signbit(actual.cpu())[zeros], torch.signbit(expected)[zeros]
        )
    return actual


@pytest.mark.parametrize(
    "shape", [(1024,), (32, 64), (1, 2, 1, 2, 1, 2, 1, 128), (1 << 20,)]
)
@pytest.mark.parametrize(
    "value", [0.5, -0.0, float("inf"), -float("inf"), float("nan")]
)
def test_masked_fill_hygon_contiguous(shape, value, monkeypatch):
    inp, mask = _inputs(shape, flag_gems.device)
    assert flag_gems.masked_fill is hygon_op.masked_fill
    assert hygon_op._can_use_contiguous(inp, mask, value)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Eligible inputs unexpectedly used the generic fallback")

    monkeypatch.setattr(hygon_op, "_masked_fill", unexpected_fallback)
    _assert_result(inp, mask, value)


@pytest.mark.parametrize("mask_kind", ["all_false", "all_true", "mixed"])
def test_masked_fill_hygon_special_input(mask_kind):
    pattern = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1.0, -1.0, 2.0]
    )
    inp = pattern.repeat(128).to(flag_gems.device)
    if mask_kind == "mixed":
        mask = torch.arange(1024, device=flag_gems.device) % 2 == 0
    else:
        mask = torch.full((1024,), mask_kind == "all_true", device=flag_gems.device)
    assert hygon_op._can_use_contiguous(inp, mask, -0.0)
    _assert_result(inp, mask, -0.0)


@pytest.mark.parametrize(
    "kind",
    [
        "row_broadcast",
        "column_broadcast",
        "scalar_mask",
        "expanded_mask",
        "transpose_input",
        "transpose_mask",
        "slice_input",
        "slice_mask",
        "offset_input",
        "offset_mask",
        "channels_last",
        "empty",
        "scalar_input",
        "short",
        "tail",
        "over_max",
        "rank9",
        "requires_grad",
        "int_value",
        "bool_value",
        "tensor_value",
        "cpu_tensor_value",
    ],
)
def test_masked_fill_hygon_fallback(kind, monkeypatch):
    inp, mask = _inputs((32, 64), flag_gems.device)
    value = 0.5
    if kind == "row_broadcast":
        mask = mask[:1, :].clone()
    elif kind == "column_broadcast":
        mask = mask[:, :1].clone()
    elif kind == "scalar_mask":
        mask = torch.tensor(True, device=flag_gems.device)
    elif kind == "expanded_mask":
        mask = mask[:1, :].expand_as(inp)
    elif kind == "transpose_input":
        inp = inp.t()
        mask = mask.t().contiguous()
    elif kind == "transpose_mask":
        mask = mask.t()
        inp = inp.t().contiguous()
    elif kind == "slice_input":
        inp = _inputs((32, 128), flag_gems.device)[0][:, ::2]
    elif kind == "slice_mask":
        mask = _inputs((32, 128), flag_gems.device)[1][:, ::2]
    elif kind == "offset_input":
        inp = _inputs((2049,), flag_gems.device)[0][1:].reshape(32, 64)
    elif kind == "offset_mask":
        mask = _inputs((2049,), flag_gems.device)[1][1:].reshape(32, 64)
    elif kind == "channels_last":
        inp, mask = _inputs((2, 4, 16, 16), flag_gems.device)
        inp = inp.contiguous(memory_format=torch.channels_last)
    elif kind == "empty":
        inp, mask = _inputs((0, 64), flag_gems.device)
    elif kind == "scalar_input":
        inp, mask = _inputs((), flag_gems.device)
    elif kind in ("short", "tail", "over_max", "rank9"):
        shape = {
            "short": (17,),
            "tail": (1025,),
            "over_max": ((1 << 20) + 1024,),
            "rank9": (1,) * 8 + (1024,),
        }[kind]
        inp, mask = _inputs(shape, flag_gems.device)
    elif kind == "requires_grad":
        inp.requires_grad_()
    elif kind == "int_value":
        value = 7
    elif kind == "bool_value":
        value = True
    elif kind == "tensor_value":
        value = torch.tensor(0.5, device=flag_gems.device)
    elif kind == "cpu_tensor_value":
        value = torch.tensor(0.5)
    assert not hygon_op._can_use_contiguous(inp, mask, value)
    original = hygon_op._masked_fill
    calls = []

    def fallback(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(hygon_op, "_masked_fill", fallback)
    _assert_result(inp, mask, value)
    assert len(calls) == 1
    assert calls[0][0] is inp and calls[0][1] is mask and calls[0][2] is value


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
@pytest.mark.parametrize("tensor_value", [False, True])
def test_masked_fill_hygon_fallback_dtype(dtype, tensor_value):
    inp, mask = _inputs((1024,), flag_gems.device, dtype)
    value = (
        torch.tensor(1, dtype=dtype, device=flag_gems.device) if tensor_value else 1.0
    )
    assert not hygon_op._can_use_contiguous(inp, mask, value)
    _assert_result(inp, mask, value)


@pytest.mark.parametrize(
    "value", [3.4028234663852886e38, -3.4028234663852886e38, 2.0**-149, -(2.0**-149)]
)
def test_masked_fill_hygon_fp32_limits(value):
    inp, mask = _inputs((1024,), flag_gems.device)
    assert hygon_op._can_use_contiguous(inp, mask, value)
    _assert_result(inp, mask, value)


@pytest.mark.parametrize("kind", ["overflow", "bad_value_rank", "bad_mask_shape"])
def test_masked_fill_hygon_invalid_fallback(kind, monkeypatch):
    inp, mask = _inputs((1024,), flag_gems.device)
    value = 1.0
    if kind == "overflow":
        value = 1e40
    elif kind == "bad_value_rank":
        value = torch.tensor([1.0], device=flag_gems.device)
    else:
        mask = torch.ones((3,), dtype=torch.bool, device=flag_gems.device)
    assert not hygon_op._can_use_contiguous(inp, mask, value)
    # Preserve the generic implementation's own validation and exceptions;
    # this optimization does not change the generic API's error policy.
    original = hygon_op._masked_fill
    try:
        expected = original(inp, mask, value)
    except Exception as error:
        with pytest.raises(type(error)):
            flag_gems.masked_fill(inp, mask, value)
    else:
        actual = flag_gems.masked_fill(inp, mask, value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("tensor_value", [False, True])
def test_masked_fill_hygon_dispatch_and_inplace(tensor_value):
    generic = importlib.import_module("flag_gems.ops.masked_fill")
    assert flag_gems.masked_fill_ is generic.masked_fill_
    inp, mask = _inputs((1024,), flag_gems.device)
    value = torch.tensor(0.5, device=flag_gems.device) if tensor_value else 0.5
    expected = torch.masked_fill(
        inp.cpu(), mask.cpu(), value.cpu() if tensor_value else value
    )
    flag_gems.enable()
    actual = torch.masked_fill(inp, mask, value)
    target = inp.clone()
    returned = target.masked_fill_(mask, value)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
    assert returned is target
