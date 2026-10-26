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

SDPA_MATH = torch.ops.aten._scaled_dot_product_attention_math
CASES = [
    (1, 2, 8, 8, 16, False, False, None),
    (2, 4, 16, 16, 32, False, True, None),
    (2, 4, 8, 16, 32, False, False, 0.5),
    (2, 4, 16, 16, 32, True, False, 1.0),
]


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.underscore_scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch,heads,q_len,kv_len,head_dim,use_mask,is_causal,scale", CASES
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math(
    batch, heads, q_len, kv_len, head_dim, use_mask, is_causal, scale, dtype
):
    torch.manual_seed(0)
    query = torch.randn(
        batch, heads, q_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    key = torch.randn(
        batch, heads, kv_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    value = torch.randn(
        batch, heads, kv_len, head_dim, dtype=dtype, device=flag_gems.device
    )
    attn_mask = (
        torch.randn(q_len, kv_len, dtype=dtype, device=flag_gems.device)
        if use_mask
        else None
    )

    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)
    ref_mask = utils.to_reference(attn_mask, True)
    ref_output, ref_weights = SDPA_MATH(
        ref_query,
        ref_key,
        ref_value,
        attn_mask=ref_mask,
        is_causal=is_causal,
        scale=scale,
    )

    output, weights = flag_gems._scaled_dot_product_attention_math(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )

    assert output.shape == (batch, heads, q_len, head_dim)
    assert weights.shape == (batch, heads, q_len, kv_len)
    utils.gems_assert_close(output, ref_output, dtype, atol=3e-2)
    utils.gems_assert_close(weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.underscore_scaled_dot_product_attention_math
def test_scaled_dot_product_attention_math_rejects_dropout():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 8, 16, device=flag_gems.device)
    with pytest.raises(AssertionError, match="dropout_p"):
        flag_gems._scaled_dot_product_attention_math(query, query, query, dropout_p=0.1)
