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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend addressing"
)


@pytest.mark.parametrize(
    "layout", ["transpose", "offset_singleton", "holes", "expanded"]
)
def test_zero_storage_layout(layout):
    base = torch.ones((8, 12), device=flag_gems.device)
    if layout == "transpose":
        view = base.t()
    elif layout == "offset_singleton":
        view = base[2:6].t().unsqueeze(1)
    elif layout == "holes":
        view = base[:, ::3]
    else:
        view = base[2:3].expand(4, 12)
    expected = torch.ones((8, 12))
    if layout == "transpose":
        expected.t().zero_()
    elif layout == "offset_singleton":
        expected[2:6].t().unsqueeze(1).zero_()
    elif layout == "holes":
        expected[:, ::3].zero_()
    else:
        expected[2:3].zero_()
    assert flag_gems.zero_(view) is view
    assert torch.equal(base.cpu(), expected)


@pytest.mark.parametrize("paired", [False, True])
def test_index_put_large_invalid_indices_do_not_wrap(paired):
    """Vendor masked OOB lanes leave storage untouched; this is not a Torch oracle."""
    x = torch.zeros((8, 4), device=flag_gems.device)
    indices = torch.tensor([2**32 + 1, -(2**32) - 1, -1], device=x.device)
    expected = torch.zeros((8, 4))
    if paired:
        cols = torch.tensor([0, 1, -1], device=x.device)
        value = torch.tensor(3.0, device=x.device)
        flag_gems.index_put_(x, [indices, cols], value)
        expected[-1, -1] = 3
    else:
        values = torch.full((3, 4), 3.0, device=x.device)
        flag_gems.index_put_(x, [indices], values)
        expected[-1] = 3
    assert torch.equal(x.cpu(), expected)


@pytest.mark.parametrize("mode", ["replicate", "reflect", "circular"])
def test_pad_address_span_guard_and_wide_fallback(monkeypatch, mode):
    import importlib

    module = importlib.import_module(flag_gems.pad.__module__)
    guard = module._pad_can_use_int32
    assert guard((2, 3), (3, 1), (4, 5), (1, 1), (1, 1), 20, 1024)
    assert not guard((2,), (2**31,), (2,), (0,), (0,), 2, 1024)
    assert not guard((2**31 - 1,), (1,), (2**31 - 1,), (0,), (0,), 2**31 - 1, 1024)
    monkeypatch.setattr(module, "_pad_can_use_int32", lambda *args: False)
    x = torch.arange(24, device=flag_gems.device).reshape(2, 3, 4).transpose(1, 2)
    expected = torch.nn.functional.pad(x.cpu(), (-1, 2), mode=mode)
    assert torch.equal(flag_gems.pad(x, (-1, 2), mode=mode).cpu(), expected)


@pytest.mark.parametrize("shape", [(8,), (8, 7), (8, 2053)])
def test_index_put_row_tile_tails(shape):
    x = torch.zeros(shape, device=flag_gems.device)
    index = torch.tensor([1, 3, -1], device=x.device)
    values = torch.arange(3 * x[0].numel(), dtype=x.dtype, device=x.device).reshape(
        3, *shape[1:]
    )
    expected = torch.zeros(shape)
    expected[[1, 3, -1]] = values.cpu()
    assert flag_gems.index_put_(x, [index], values) is x
    assert torch.equal(x.cpu(), expected)


def test_index_scalar_plain_tile():
    x = torch.arange(12, device=flag_gems.device).reshape(3, 4)
    indices = [
        torch.tensor([0, -1, 1], device=x.device),
        torch.tensor([-1, 0, 2], device=x.device),
    ]
    expected = x.cpu()[indices[0].cpu(), indices[1].cpu()]
    assert torch.equal(flag_gems.index(x, indices).cpu(), expected)


@pytest.mark.pad
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend addressing")
def test_ascend_pad_large_reflect_exact_coordinates():
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    x_cpu = torch.randn((64, 64, 64, 64), dtype=torch.float16, generator=generator)
    padding = (3, 5, 4, 2, 1, 7)
    expected = torch.nn.functional.pad(x_cpu.float(), padding, mode="reflect").to(x_cpu.dtype)
    assert expected.numel() > 2**24
    actual = flag_gems.pad(x_cpu.to(flag_gems.device), padding, mode="reflect")
    assert torch.equal(actual.cpu(), expected)
