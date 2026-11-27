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
    m, h = shape
    x = torch.randn((m, 4, h), device=device, dtype=torch.bfloat16)
    kv = torch.randn((m, 5, h), device=device, dtype=torch.bfloat16)
    weight = torch.randn((4, h), device=device)
    return (x, kv, weight, 1e-20, 1e-6), dict(save_for_backward=True)


@pytest.mark.engram_gate_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
@pytest.mark.parametrize("shape", [(512, 128), (512, 4096)])
def test_engram_gate_fwd(shape):
    oracle = pytest.importorskip("tile_kernels.torch.engram").engram_gate_ref
    args, kwargs = make_inputs(shape, flag_gems.device)
    x, kv, weight, eps, clamp = args
    expected = oracle(
        x, kv, weight, torch.ones_like(weight), clamp, eps, save_for_backward=True
    )
    actual = flag_gems.engram_gate_fwd(*args, **kwargs)
    # Use the upstream tolerances for output, dot and gate score, respectively.
    for result, ref, tol in zip(actual[:3], expected[:3], (8e-3, 2e-4, 1e-4)):
        torch.testing.assert_close(result, ref, atol=tol, rtol=tol)
    torch.testing.assert_close(actual[3:], expected[3:])
