# Copyright 2026, The FlagOS Contributors.
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
from .accuracy_utils import gems_assert_close


@pytest.mark.scaled_dot_product_attention_math_for_mps
@pytest.mark.underscore_scaled_dot_product_attention_math_for_mps
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("num_head", [4, 8])
@pytest.mark.parametrize("seq_len", [16, 32, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_attention_math_for_mps(
    batch, num_head, seq_len, head_dim, dtype, is_causal
):
    """Test accuracy of _scaled_dot_product_attention_math_for_mps."""
    device = flag_gems.device
    q = torch.randn(
        batch, num_head, seq_len, head_dim, dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    k = torch.randn(
        batch, num_head, seq_len, head_dim, dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    v = torch.randn(
        batch, num_head, seq_len, head_dim, dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)

    # Reference: the math definition returns both the attention output and the
    # full softmax attention-weight matrix. The aten op only dispatches on
    # MPS/Meta, so compute the reference math directly.
    scale = 1.0 / (head_dim**0.5)
    ref_q, ref_k, ref_v = (utils.to_reference(x).float() for x in (q, k, v))
    scores = torch.matmul(ref_q, ref_k.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=ref_q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    ref_weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(ref_weights, ref_v)
    ref_output = utils.to_reference(ref_output.to(dtype), True)
    ref_weights = utils.to_reference(ref_weights.to(dtype), True)

    expected_inputs = tuple((x.device, x.dtype, tuple(x.shape)) for x in (q, k, v))
    output, attn_weights = flag_gems._scaled_dot_product_attention_math_for_mps(
        q, k, v, is_causal=is_causal
    )

    for inp, expected in zip((q, k, v), expected_inputs):
        assert (inp.device, inp.dtype, tuple(inp.shape)) == expected
    device, expected_dtype, query_shape = expected_inputs[0]
    key_shape = expected_inputs[1][2]
    assert output.device == device and output.dtype == expected_dtype
    assert output.shape == query_shape
    assert attn_weights.device == device and attn_weights.dtype == expected_dtype
    assert attn_weights.shape == (*query_shape[:-1], key_shape[-2])

    # output and attn_weights both reduce over the key/sequence dimension
    # (softmax normalization + weighted value sum), so scale atol accordingly.
    gems_assert_close(output, ref_output, dtype, reduce_dim=seq_len)
    gems_assert_close(attn_weights, ref_weights, dtype, reduce_dim=seq_len)
