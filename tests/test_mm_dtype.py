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
    pytest.mark.mm_dtype,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.aten.mm.dtype


def _inputs(shape, dtype, device):
    m, k, n = shape
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((n, k), device=device, dtype=dtype).T
    return (a, b, torch.float32), {}


@pytest.mark.parametrize("shape", [(1, 6144, 256), (4, 128, 64)])
def test_mm_dtype_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.mm_dtype(*args, **kwargs)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)


def test_a5_fast_path(monkeypatch):
    import sys

    module = sys.modules[flag_gems.mm_dtype.__module__]
    original = module._launch
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_launch", counted)
    x = torch.randn((1, 6144), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.randn((256, 6144), dtype=x.dtype, device=x.device).T
    expected = torch.ops.aten.mm.dtype(x, weight, torch.float32)
    actual = flag_gems.mm_dtype(x, weight, torch.float32)
    assert calls
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.03)
