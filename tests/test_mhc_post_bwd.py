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
    x = torch.randn((1, m, h), device=device, dtype=torch.bfloat16)
    residual = torch.randn((1, m, 4, h), device=device, dtype=torch.bfloat16)
    post = torch.randn((1, m, 4, 1), device=device)
    comb = torch.randn((1, m, 4, 4), device=device)
    args = (x, residual, post, comb)
    return (*args, torch.randn_like(residual)), {}


@pytest.mark.mhc_post_bwd
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
@pytest.mark.parametrize("shape", [(512, 4096), (512, 7168)])
def test_mhc_post_bwd(shape):
    reference = pytest.importorskip("tile_kernels.mhc.post_kernel").mhc_post_bwd
    args, kwargs = make_inputs(shape, flag_gems.device)
    expected = reference(*args, **kwargs)
    actual = flag_gems.mhc_post_bwd(*args, **kwargs)
    torch.testing.assert_close(actual[:2], expected[:2])
    torch.testing.assert_close(actual[2:], expected[2:], rtol=1e-4, atol=1e-4)
