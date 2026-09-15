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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils

# Register marker for operator name starting with underscore
setattr(
    pytest.mark,
    "_transform_bias_rescale_qkv",
    MarkDecorator(
        Mark("_transform_bias_rescale_qkv", (), {}, _ispytest=True), _ispytest=True
    ),
)


@pytest.mark._transform_bias_rescale_qkv
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 16, 32])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__transform_bias_rescale_qkv(batch, seq_len, num_heads, head_dim, dtype):
    hidden_dim = num_heads * head_dim
    total_dim = 3 * hidden_dim

    res_qkv = torch.randn(
        batch, seq_len, total_dim, dtype=dtype, device=flag_gems.device
    )
    res_bias = torch.randn(total_dim, dtype=dtype, device=flag_gems.device)
    ref_qkv = utils.to_reference(res_qkv, True)
    ref_bias = utils.to_reference(res_bias, True)

    ref_q, ref_k, ref_v = torch._transform_bias_rescale_qkv(
        ref_qkv, ref_bias, num_heads
    )
    res_q, res_k, res_v = flag_gems._transform_bias_rescale_qkv(
        res_qkv, res_bias, num_heads
    )

    utils.gems_assert_close(res_q, ref_q, dtype)
    utils.gems_assert_close(res_k, ref_k, dtype)
    utils.gems_assert_close(res_v, ref_v, dtype)
