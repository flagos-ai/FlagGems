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
    pytest.mark.foreach_add_list,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.aten._foreach_add.List


def _inputs(shape, dtype, device):
    a = torch.randn(shape, device=device, dtype=dtype)
    return ([a, torch.randn_like(a)], [torch.randn_like(a), torch.randn_like(a)]), {}


@pytest.mark.parametrize("shape", [(1, 4096), (1, 6144), (1024, 4096)])
def test_foreach_add_list_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.foreach_add_list(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
