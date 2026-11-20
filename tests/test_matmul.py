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
    pytest.mark.matmul,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.matmul


def _inputs(shape, dtype, device):
    m, k, n = shape
    a = torch.randn((m, 1, k), device=device, dtype=dtype)
    b = torch.randn((n, k), device=device, dtype=dtype).T
    return (a, b), {}


@pytest.mark.parametrize("shape", [(1024, 128, 128), (1, 16384, 4)])
def test_matmul_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.matmul(*args, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.016)


def test_public_dispatch(monkeypatch):
    import sys

    module = sys.modules[flag_gems.matmul.__module__]
    original = module._launch
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_launch", counted)
    x = torch.randn((1024, 1, 128), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.randn((128, 128), dtype=x.dtype, device=x.device)
    expected = torch.matmul(x, weight)
    with flag_gems.use_gems(include=["matmul"]):
        actual = torch.matmul(x, weight)
    assert calls
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.03)


@pytest.mark.parametrize("shapes", [((64,), (64,)), ((1, 3, 5, 16), (4, 1, 16, 8))])
def test_matmul_rank_and_broadcast(shapes):
    a, b = (
        torch.randn(shape, dtype=torch.bfloat16, device=flag_gems.device)
        for shape in shapes
    )
    actual = flag_gems.matmul(a, b)
    torch.testing.assert_close(actual, torch.matmul(a, b), atol=0.03, rtol=0.016)
