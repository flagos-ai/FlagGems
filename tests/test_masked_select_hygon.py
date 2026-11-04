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

from . import accuracy_utils as utils

hygon_masked_select = None
if flag_gems.vendor_name == "hygon":
    # The backend imports _hygon.ops via its own sys.path. Use the actual
    # exported function's module so monkeypatches reach its globals rather
    # than creating a second module under the fully qualified package name.
    hygon_masked_select = importlib.import_module(flag_gems.masked_select.__module__)

pytestmark = [
    pytest.mark.masked_select,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon empty fast path"
    ),
]

EMPTY_SHAPES = [
    ((0,), (0,)),
    ((0, 3), (1, 3)),
    ((1, 3), (0, 3)),
    ((), (0,)),
    ((0, 3), ()),
    ((2, 0, 3), (1, 1, 3)),
    ((1, 0, 1), (2, 1, 3)),
]


def _fail_generic(*args, **kwargs):
    raise AssertionError("metadata-known empty result reached generic kernels")


@pytest.mark.parametrize("input_shape,mask_shape", EMPTY_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.bool]
)
def test_masked_select_hygon_empty(input_shape, mask_shape, dtype, monkeypatch):
    inp = torch.zeros(input_shape, dtype=dtype)
    mask = torch.ones(mask_shape, dtype=torch.bool)
    expected = torch.masked_select(inp, mask)
    inp, mask = inp.to(flag_gems.device), mask.to(flag_gems.device)
    monkeypatch.setattr(hygon_masked_select, "_generic_masked_select", _fail_generic)
    out = flag_gems.masked_select(inp, mask)
    assert out.shape == (0,)
    assert out.dtype == dtype and out.device == inp.device
    assert out is not inp and out is not mask
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_masked_select_hygon_empty_strides_offsets_and_nan(monkeypatch):
    backing = torch.full((4, 6), float("nan"), device=flag_gems.device)
    inp = backing[1:3, 1:2].expand(2, 3)
    mask_backing = torch.ones((4, 6), dtype=torch.bool, device=flag_gems.device)
    mask = mask_backing[:0, 1:4]
    expected = torch.masked_select(inp[:1].cpu(), mask.cpu())
    assert expected.numel() == 0
    monkeypatch.setattr(hygon_masked_select, "_generic_masked_select", _fail_generic)
    out = flag_gems.masked_select(inp[:1], mask)
    assert out.shape == (0,)
    assert torch.isnan(backing).all().item()
    assert mask_backing.all().item()


@pytest.mark.parametrize(
    "input_shape,mask_shape",
    [
        ((0, 3), (2, 3)),
        ((0, 3), (1, 4)),
        ((2, 0, 3), (2, 5, 1)),
        ((0, 3, 2), (1, 4, 2)),
    ],
)
def test_masked_select_hygon_invalid_empty_broadcast(input_shape, mask_shape):
    inp = torch.zeros(input_shape)
    mask = torch.ones(mask_shape, dtype=torch.bool)
    with pytest.raises(RuntimeError):
        torch.masked_select(inp, mask)
    inp, mask = inp.to(flag_gems.device), mask.to(flag_gems.device)
    with pytest.raises(RuntimeError, match="non-singleton dimension"):
        flag_gems.masked_select(inp, mask)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float32])
def test_masked_select_hygon_invalid_empty_mask_dtype(dtype):
    inp = torch.zeros((0, 3))
    mask = torch.ones((1, 3), dtype=dtype)
    with pytest.raises(RuntimeError, match="BoolTensor"):
        torch.masked_select(inp, mask)
    inp, mask = inp.to(flag_gems.device), mask.to(flag_gems.device)
    with pytest.raises(RuntimeError, match="BoolTensor"):
        flag_gems.masked_select(inp, mask)


def test_masked_select_hygon_empty_device_mismatch():
    inp = torch.zeros((0, 3), device=flag_gems.device)
    mask = torch.ones((1, 3), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.masked_select(inp, mask)


@pytest.mark.parametrize("n", [1, 17, 4096, 4097, 8193])
@pytest.mark.parametrize("selection", ["none", "all", "mixed"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_masked_select_hygon_nonempty_cardinality(n, selection, dtype, monkeypatch):
    inp = (torch.arange(n, dtype=torch.float32) / 8).to(dtype)
    mask = torch.arange(n) % 3 == 0
    if selection == "none":
        mask.zero_()
    elif selection == "all":
        mask.fill_(True)
    expected = torch.masked_select(inp, mask)
    inp, mask = inp.to(flag_gems.device), mask.to(flag_gems.device)
    generic = hygon_masked_select._generic_masked_select
    calls = []

    def record_generic(x, m):
        calls.append((x, m))
        return generic(x, m)

    monkeypatch.setattr(hygon_masked_select, "_generic_masked_select", record_generic)
    out = flag_gems.masked_select(inp, mask)
    assert len(calls) == 1
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "variant", ["strided", "transposed", "broadcast", "scalar_true", "scalar_false"]
)
def test_masked_select_hygon_nonempty_logical_order(variant):
    backing = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    backing[1, 1] = float("nan")
    backing[2, 3] = float("inf")
    mask_backing = torch.arange(48).reshape(6, 8) % 3 != 0
    mask_backing[1, 1] = True
    if variant == "strided":
        inp, mask = backing[1:5, 1:7:2], mask_backing[1:5, 1:7:2]
    elif variant == "transposed":
        inp, mask = backing.t(), mask_backing.t()
    elif variant == "broadcast":
        inp, mask = backing[:1, :], mask_backing[:, :1]
    else:
        inp, mask = torch.tensor(float("nan")), torch.tensor(variant == "scalar_true")
    expected = torch.masked_select(inp, mask)
    # Rebuild views on device, rather than making the CPU view contiguous in to().
    if variant in ("strided", "transposed", "broadcast"):
        db, mb = backing.to(flag_gems.device), mask_backing.to(flag_gems.device)
        if variant == "strided":
            inp, mask = db[1:5, 1:7:2], mb[1:5, 1:7:2]
        elif variant == "transposed":
            inp, mask = db.t(), mb.t()
        else:
            inp, mask = db[:1, :], mb[:, :1]
    else:
        inp, mask = inp.to(flag_gems.device), mask.to(flag_gems.device)
    out = flag_gems.masked_select(inp, mask)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    if variant in ("strided", "transposed", "broadcast"):
        torch.testing.assert_close(db.cpu(), backing, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(mb.cpu(), mask_backing, rtol=0, atol=0)


def test_masked_select_hygon_registered_empty(monkeypatch):
    inp = torch.zeros((0, 3), device=flag_gems.device)
    mask = torch.ones((1, 3), dtype=torch.bool, device=flag_gems.device)
    monkeypatch.setattr(hygon_masked_select, "_generic_masked_select", _fail_generic)
    with flag_gems.use_gems(include=["masked_select"]):
        out = torch.masked_select(inp, mask)
    assert out.shape == (0,) and out.dtype == inp.dtype
