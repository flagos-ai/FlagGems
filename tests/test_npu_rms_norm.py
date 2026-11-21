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
    pytest.mark.npu_rms_norm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.npu.npu_rms_norm.default


def _inputs(shape, dtype, device):
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape[-1], dtype=dtype, device=device),
    ), {}


@pytest.mark.parametrize("shape", [(1, 2048), (1024, 6144), (1, 64, 512)])
def test_npu_rms_norm_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.npu_rms_norm(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
