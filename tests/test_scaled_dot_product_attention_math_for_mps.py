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
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    ref_weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(ref_weights, v.float())
    ref_output = utils.to_reference(ref_output.to(dtype))
    ref_weights = utils.to_reference(ref_weights.to(dtype))

    output, attn_weights = flag_gems._scaled_dot_product_attention_math_for_mps(
        q, k, v, is_causal=is_causal
    )

    # output and attn_weights both reduce over the key/sequence dimension
    # (softmax normalization + weighted value sum), so scale atol accordingly.
    gems_assert_close(output, ref_output, dtype, reduce_dim=seq_len)
    gems_assert_close(attn_weights, ref_weights, dtype, reduce_dim=seq_len)


@pytest.mark.scaled_dot_product_attention_math_for_mps
@pytest.mark.parametrize("seq_len", [16, 32, 70])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math_for_mps_all_negative_logits(seq_len, dtype):
    """Rows whose logits are all far below zero must still be a plain softmax.

    Regression: with every logit at -16 the result must be the uniform
    distribution (output = 1, weight = 1 / seq_len). Backends that normalize
    against ``max(rowmax, 0)`` instead of the true row maximum push the whole
    row past their exponential cut-off and return output 0 / weights inf.
    """
    device = flag_gems.device
    head_dim = 64
    q = torch.ones((1, 1, seq_len, head_dim), dtype=dtype, device=device)
    k = -2 * q
    v = q

    output, attn_weights = flag_gems._scaled_dot_product_attention_math_for_mps(q, k, v)

    assert torch.isfinite(attn_weights).all(), "attention weights must be finite"
    ref_output = utils.to_reference(torch.ones_like(output))
    ref_weights = utils.to_reference(torch.full_like(attn_weights, 1.0 / seq_len))
    gems_assert_close(output, ref_output, dtype, reduce_dim=seq_len)
    gems_assert_close(attn_weights, ref_weights, dtype, reduce_dim=seq_len)


@pytest.mark.scaled_dot_product_attention_math_for_mps
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math_for_mps_weights_match_output(
    dtype, is_causal
):
    """The returned weights must reproduce the returned output.

    Regression: the output and the weights were normalized with different
    truncation rules, so weights that the output pass did accumulate were
    dropped from the returned matrix and ``weights @ value`` went to 0 while
    the output stayed around 0.75.
    """
    device = flag_gems.device
    seq_len, head_dim = 128, 64
    half = seq_len // 2
    # The first half of the keys sits ~9.5 logits below the second half: small
    # enough to be truncated by a log-sum-exp based cut, large enough to carry
    # the whole output because only those keys hold a non-zero value.
    q = torch.ones((1, 1, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.zeros_like(q)
    k[:, :, :half, :] = -1.1875
    v = torch.zeros_like(q)
    v[:, :, :half, :] = 10000.0

    output, attn_weights = flag_gems._scaled_dot_product_attention_math_for_mps(
        q, k, v, is_causal=is_causal
    )

    # ``weights @ value`` is the reference, evaluated on CPU in float64 because
    # the device matmul may itself run at reduced precision.  The comparison is
    # relative: the output is accumulated with a reduced-precision dot product,
    # and the small weights are rounded to ``dtype`` on the way out (they are
    # subnormal in float16), so only the relative agreement is meaningful here.
    # The truncation bug this guards against turned the reference into 0.0.
    ref_output = torch.matmul(
        attn_weights.detach().double().cpu(), v.detach().double().cpu()
    )
    rtol = 5e-3 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(
        output.detach().double().cpu(), ref_output, rtol=rtol, atol=1e-3
    )
