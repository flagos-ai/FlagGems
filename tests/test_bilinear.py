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

# (batch_dims, in1_features, in2_features, out_features)
BILINEAR_SHAPES = [
    ((8,), 16, 24, 32),
    ((4,), 1, 1, 1),
    ((128,), 64, 48, 16),
    ((2, 8), 32, 32, 64),
    ((16,), 17, 13, 7),
    ((3, 5, 4), 8, 6, 10),
]


@pytest.mark.bilinear
@pytest.mark.parametrize("batch_dims, in1, in2, out", BILINEAR_SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bilinear(batch_dims, in1, in2, out, with_bias, dtype):
    input1 = torch.randn((*batch_dims, in1), dtype=dtype, device=flag_gems.device)
    input2 = torch.randn((*batch_dims, in2), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((out, in1, in2), dtype=dtype, device=flag_gems.device)
    bias = (
        torch.randn((out,), dtype=dtype, device=flag_gems.device) if with_bias else None
    )

    ref_input1 = utils.to_reference(input1, True)
    ref_input2 = utils.to_reference(input2, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.bilinear(ref_input1, ref_input2, ref_weight, ref_bias)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.bilinear(input1, input2, weight, bias)

    # Output reduces over in1 * in2 elements.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in1 * in2)
