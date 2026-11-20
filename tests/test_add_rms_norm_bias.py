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
    pytest.mark.add_rms_norm_bias,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


@pytest.mark.parametrize("rows,width", [(1, 6144), (1024, 6144), (32, 129)])
@pytest.mark.parametrize("with_beta", [True, False])
def test_add_rms_norm_bias(rows, width, with_beta):
    # A padded second input exercises the independent row strides.
    x1 = torch.randn((rows, width), device=flag_gems.device, dtype=torch.bfloat16)
    x2 = torch.randn((rows, width + 8), device=x1.device, dtype=x1.dtype)[:, :width]
    gamma = torch.randn((width,), device=x1.device, dtype=x1.dtype)
    beta = torch.randn_like(gamma) if with_beta else None
    epsilon = 3e-5
    y, rstd, residual = flag_gems.add_rms_norm_bias(x1, x2, gamma, beta, epsilon)
    summed = x1.cpu().float() + x2.cpu().float()
    inv_rms = (summed.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    expected = summed * inv_rms * gamma.cpu().float()
    if beta is not None:
        expected += beta.cpu().float()
    torch.testing.assert_close(residual.cpu(), summed.to(x1.dtype), rtol=0, atol=0)
    torch.testing.assert_close(rstd.cpu(), inv_rms, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(y.cpu(), expected.to(x1.dtype), rtol=0.008, atol=0.0001)


@pytest.mark.parametrize("rows", [1, 1024])
def test_add_rms_norm_bias_native(rows):
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    pytest.importorskip("vllm_ascend.vllm_ascend_C")
    x1 = torch.randn((rows, 6144), dtype=torch.bfloat16, device=flag_gems.device)
    x2 = torch.randn_like(x1)
    gamma = torch.randn(6144, dtype=x1.dtype, device=x1.device)
    beta = torch.randn_like(gamma)
    expected = torch.ops._C_ascend.npu_add_rms_norm_bias.default(x1, x2, gamma, beta)
    actual = flag_gems.add_rms_norm_bias(x1, x2, gamma, beta)
    torch.testing.assert_close(actual, expected)
