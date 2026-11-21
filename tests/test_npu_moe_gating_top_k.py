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
    pytest.mark.npu_moe_gating_top_k,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.npu.npu_moe_gating_top_k.default


def _inputs(shape, dtype, device):
    rows, experts, k = shape
    x = torch.randn(rows, experts, dtype=torch.float32, device=device)
    bias = torch.arange(experts, dtype=torch.float32, device=device) / experts
    return (x, k), dict(
        bias=bias, k_group=1, group_count=1, group_select_mode=1, norm_type=1
    )


@pytest.mark.parametrize(
    "shape", [(1, 256, 8), (1024, 256, 8), (2, 128, 4), (256, 128, 4)]
)
def test_npu_moe_gating_top_k_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.npu_moe_gating_top_k(*args, **kwargs)
    assert len(actual) == len(expected) == 3
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    # With out_flag=False the third native output has no defined element values.
    assert actual[2].shape == expected[2].shape and actual[2].dtype == expected[2].dtype
