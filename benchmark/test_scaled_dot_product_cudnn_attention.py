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

from . import base
from .consts import FLOAT_DTYPES

# cuDNN attention only supports fp16/bf16, not fp32
CUDNN_ATTENTION_DTYPES = [dt for dt in FLOAT_DTYPES if dt != torch.float32]


class AttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing four elements:
        # (batch, num_heads, seq_len, head_size).
        return []


def torch_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    scale=None,
):
    # Use PyTorch's scaled_dot_product_attention as reference
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def gems_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    scale=None,
):
    result = flag_gems.ops._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        compute_log_sumexp=compute_log_sumexp,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )
    return result[0]  # Return only the output tensor


@pytest.mark.scaled_dot_product_cudnn_attention
@pytest.mark.skipif(
    flag_gems.device != "cuda",
    reason="_scaled_dot_product_cudnn_attention is a CUDA-only operator (cuDNN)",
)
@pytest.mark.parametrize("is_causal", [True, False])
def test_scaled_dot_product_cudnn_attention(is_causal):
    """Benchmark for _scaled_dot_product_cudnn_attention."""

    def scaled_dot_product_cudnn_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        head_size = shape[-1]
        scale = 1.0 / (head_size**0.5)
        yield query, key, value, None, True, 0.0, is_causal, False, scale

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_cudnn_attention",
        input_fn=scaled_dot_product_cudnn_attention_kwargs,
        torch_op=torch_cudnn_attention,
        gems_op=gems_cudnn_attention,
        dtypes=CUDNN_ATTENTION_DTYPES,
    )
    bench.run()
