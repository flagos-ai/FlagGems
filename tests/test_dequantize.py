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

from . import accuracy_utils as utils

# Dequantize operator tests
# Dequantize input is always a quantized tensor (torch.qint8);
# output is always torch.float32. No FLOAT_DTYPES parametrization needed.
# Typical shapes for quantized tensor dequantization testing
QUANT_SHAPES = [(4, 4), (16, 32), (32, 64), (64, 128), (1024, 1024)]


@pytest.mark.dequantize
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("scale", [0.1, 0.01, 0.5])
@pytest.mark.parametrize("zero_point", [0, 10, -20])
def test_dequantize(shape, scale, zero_point):
    # Create quantized tensor
    torch.manual_seed(0)
    fp_tensor = torch.randn(shape, device="cpu")
    q_tensor = torch.quantize_per_tensor(
        fp_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8
    )
    if flag_gems.vendor_name == "ascend":
        host = q_tensor
        q_tensor = torch._empty_affine_quantized(
            host.shape,
            scale=host.q_scale(),
            zero_point=host.q_zero_point(),
            dtype=host.dtype,
            device=flag_gems.device,
        )
        q_tensor.untyped_storage().copy_(host.untyped_storage())
    else:
        q_tensor = q_tensor.to(flag_gems.device)

    if flag_gems.vendor_name == "ascend" and utils.TO_CPU:
        # Observe the actual target bytes: quantized .to("cpu") is unsupported.
        integers = torch.empty(0, dtype=torch.int8).set_(
            q_tensor.untyped_storage().cpu(),
            q_tensor.storage_offset(),
            q_tensor.shape,
            q_tensor.stride(),
        )
        ref_q_tensor = torch._make_per_tensor_quantized_tensor(
            integers, q_tensor.q_scale(), q_tensor.q_zero_point()
        )
    else:
        ref_q_tensor = utils.to_reference(q_tensor)

    # Reference dequantize, upcast after dequantization for comparison.
    ref_out = utils.to_reference(torch.dequantize(ref_q_tensor), True)

    # GEMS dequantize
    res_out = flag_gems.dequantize(q_tensor)

    if flag_gems.vendor_name == "ascend":
        assert res_out.device == q_tensor.device
        assert res_out.dtype == torch.float32 and res_out.shape == q_tensor.shape
        assert res_out.untyped_storage()._cdata != q_tensor.untyped_storage()._cdata

    # Output is always float32
    utils.gems_assert_close(res_out, ref_out, dtype=torch.float32)
