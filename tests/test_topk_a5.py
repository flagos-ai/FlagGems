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
    pytest.mark.topk,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.aten.topk.default


def _inputs(shape, dtype, device):
    x = torch.randn(
        (*shape[:-1], max(shape[-1], 64)), dtype=torch.float32, device=device
    )[..., : shape[-1]]
    return (x, 16), dict(dim=-1, largest=shape[0] == 1, sorted=True)


@pytest.mark.parametrize("shape", [(1, 1024, 16), (1, 8, 64), (2, 3, 32)])
def test_topk_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.topk(*args, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(
        args[0].gather(-1, actual[1]), actual[0], atol=0, rtol=0, equal_nan=True
    )
    assert not actual[1].sort(-1).values.diff(dim=-1).eq(0).any()
