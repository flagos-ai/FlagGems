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

pytestmark = [
    pytest.mark.linear,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.nn.functional.linear


def _inputs(shape, dtype, device):
    m, k, n = shape
    x = torch.randn((m, k), device=device, dtype=dtype)
    w = torch.randn((n, k), device=device, dtype=dtype)
    return (x, w, torch.randn(n, dtype=dtype, device=device) if m == 8 else None), {}


@pytest.mark.parametrize("shape", [(1, 4096, 64), (1, 128, 128), (8, 6144, 128)])
def test_linear_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.linear(*args, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.016)


def test_public_dispatch(monkeypatch):
    import sys

    module = sys.modules[flag_gems.linear.__module__]
    original = module._launch
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_launch", counted)
    x = torch.randn((1, 4096), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.randn((64, 4096), dtype=x.dtype, device=x.device)
    expected = torch.nn.functional.linear(x, weight)
    with flag_gems.use_gems(include=["linear"]):
        actual = torch.nn.functional.linear(x, weight)
    assert calls
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.03)
