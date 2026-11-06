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
    pytest.mark.native_layer_norm,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon fast path"),
]


def device():
    if hasattr(torch, "flagos") and torch.flagos.is_available():
        return "flagos:0"
    return flag_gems.device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [20, 320, 1025, 4096])
@pytest.mark.parametrize("entrypoint", ["layer_norm", "native_layer_norm"])
def test_strided_native_outputs(dtype, width, entrypoint):
    base = torch.randn(3, 5, width, dtype=dtype)
    x = base.to(device()).permute(1, 0, 2)[1:]
    w = torch.randn(width, dtype=dtype, device=x.device)
    b = torch.randn(width, dtype=dtype, device=x.device)
    before = x.cpu().clone()
    reference = torch.ops.aten.native_layer_norm.default(
        x.cpu().float(), (width,), w.cpu().float(), b.cpu().float(), 1e-5
    )
    result = getattr(flag_gems, entrypoint)(x, (width,), w, b, 1e-5)
    tolerance = (
        1e-4 if dtype == torch.float32 else (3e-3 if dtype == torch.float16 else 3e-2)
    )
    for index, (actual, expected) in enumerate(zip(result, reference)):
        output_dtype = torch.float32 if index else dtype
        assert actual.dtype == output_dtype and actual.shape == expected.shape
        torch.testing.assert_close(
            actual.cpu(), expected.to(output_dtype), rtol=tolerance, atol=tolerance
        )
    torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)


def test_warm_plan_uses_current_data_and_epsilon():
    x = torch.randn(2, 3, 320, device=device())
    first = flag_gems.layer_norm(x, (320,), None, None, 1e-5)
    second = flag_gems.layer_norm(x + 3, (320,), None, None, 1e-2)
    assert len({v.untyped_storage().data_ptr() for v in (*first, *second)}) == 6
    reference = torch.ops.aten.native_layer_norm.default(
        x.cpu() + 3, (320,), None, None, 1e-2
    )
    for actual, expected in zip(second, reference):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_last_axis_stride_generic_path():
    x = torch.randn(2, 3, 640, device=device())[..., ::2]
    result = flag_gems.layer_norm(x, (320,))
    reference = torch.ops.aten.native_layer_norm.default(
        x.cpu(), (320,), None, None, 1e-5
    )
    for actual, expected in zip(result, reference):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
