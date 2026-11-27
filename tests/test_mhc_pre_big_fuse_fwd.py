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
    residual = torch.randn((m, 4, h), device=device, dtype=torch.bfloat16)
    partials = torch.randn((1, m, 24), device=device)
    squares = residual.float().square().sum(dim=(1, 2)).reshape(1, m)
    scale = torch.randn(3, device=device) * 0.1
    bias = torch.randn(24, device=device) * 0.1
    post = torch.empty((m, 4), device=device)
    comb = torch.empty((m, 16), device=device)
    layer = torch.empty((m, h), device=device, dtype=torch.bfloat16)
    return (partials, squares, scale, bias, residual, post, comb, layer), dict(
        rms_eps=1e-6,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=1.0,
        sinkhorn_repeat=10,
        n_splits=1,
    )


@pytest.mark.mhc_pre_big_fuse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
@pytest.mark.parametrize("shape", [(512, 4096), (512, 7168)])
def test_mhc_pre_big_fuse_fwd(shape):
    reference = pytest.importorskip(
        "tile_kernels.mhc.pre_big_fuse_kernel"
    ).mhc_pre_big_fuse_fwd
    args, kwargs = make_inputs(shape, flag_gems.device)
    ref_outputs = tuple(torch.empty_like(t) for t in args[5:8])
    reference(*args[:5], *ref_outputs, **kwargs)
    assert flag_gems.mhc_pre_big_fuse_fwd(*args, **kwargs) is None
    torch.testing.assert_close(args[5:8], ref_outputs)
