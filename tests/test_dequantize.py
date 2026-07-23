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
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("scale", [0.1, 0.01, 0.5])
@pytest.mark.parametrize("zero_point", [0, 10, -20])
def test_dequantize(shape, scale, zero_point):
    # Create quantized tensor
    fp_tensor = torch.randn(shape, device="cpu")
    q_tensor = torch.quantize_per_tensor(
        fp_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8
    ).to(flag_gems.device)

    ref_q_tensor = utils.to_reference(q_tensor)

    # Reference dequantize
    ref_out = torch.dequantize(ref_q_tensor)

    # GEMS dequantize
    with flag_gems.use_gems():
        res_out = torch.dequantize(q_tensor)

    # Output is always float32
    utils.gems_assert_close(res_out, ref_out, dtype=torch.float32)
