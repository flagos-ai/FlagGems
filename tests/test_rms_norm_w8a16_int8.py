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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

GROUP_SIZE = 128


def _quantize_int8_weight(weight, group_size=GROUP_SIZE):
    grouped_weight = weight.float().reshape(-1, group_size)
    scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    weight_q = (
        (grouped_weight / scale)
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
        .reshape_as(weight)
        .contiguous()
    )
    return weight_q, scale.squeeze(-1).to(weight.dtype).contiguous()


W8A16_SHAPES = [
    (1, 4096),
    (128, 4096),
    (512, 4096),
    (64, 8192),
    (1, 16384),
    (1, 32768),
]


def _run_rms_norm_w8a16_test(shape, quantize_weight, op):
    dtype = torch.bfloat16
    m, n = shape
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, (m, n)).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, (n,)).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device)
    weight_q, weight_scale = quantize_weight(weight)
    dequant_weight = (
        (weight_q.float().reshape(-1, GROUP_SIZE) * weight_scale.float().unsqueeze(-1))
        .reshape_as(weight)
        .to(dtype)
    )

    eps = 1e-5
    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(dequant_weight)
    ref_out = torch.nn.functional.rms_norm(ref_inp, (n,), ref_weight, eps=eps)
    res_out = op(
        inp,
        (n,),
        weight_q,
        weight_scale,
        eps=eps,
        group_size=GROUP_SIZE,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rms_norm_w8a16_int8
@pytest.mark.parametrize("shape", W8A16_SHAPES)
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="RMSNorm W8A16 INT8 is only available on Ascend",
)
def test_rms_norm_w8a16_int8(shape):
    _run_rms_norm_w8a16_test(
        shape, _quantize_int8_weight, flag_gems.rms_norm_w8a16_int8
    )
