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


def make_inputs(shape, device):
    m, experts = shape
    logits = torch.randn((m, experts), device=device)
    return (logits, 8, False, 0, 2.5, 0), dict(bias=torch.randn(experts, device=device))


@pytest.mark.moe_topk_gate_forward
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
@pytest.mark.parametrize("shape", [(512, 256), (8001, 256)])
def test_moe_topk_gate_forward(shape):
    reference = pytest.importorskip(
        "tile_kernels.moe.moe_topk_gate_forward_kernel"
    ).moe_topk_gate_forward
    args, kwargs = make_inputs(shape, flag_gems.device)
    expected = reference(*args, **kwargs)
    actual = flag_gems.moe_topk_gate_forward(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
