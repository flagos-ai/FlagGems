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

import math

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base


class CrossAttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _torch_attention(query, key, value, attn_mask, scale):
    allowed_mask = None if attn_mask is None else ~(attn_mask != 0)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=allowed_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


def _input_fn(shape, dtype, device):
    batch, heads, query_len, head_dim, key_len = shape
    query = torch.randn(batch, heads, query_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, heads, key_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, heads, key_len, head_dim, device=device, dtype=dtype)
    yield query, key, value, None, 1.0 / math.sqrt(head_dim)


def _masked_input_fn(shape, dtype, device):
    batch, heads, query_len, head_dim, key_len = shape
    query = torch.randn(batch, heads, query_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, heads, key_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, heads, key_len, head_dim, device=device, dtype=dtype)
    mask = torch.zeros(batch, 1, query_len, key_len, device=device, dtype=torch.bool)
    mask[..., (key_len * 3) // 4 :] = True
    yield query, key, value, mask, 1.0 / math.sqrt(head_dim)


@pytest.mark.cross_attention
@pytest.mark.parametrize(
    "op_name,input_fn",
    (("cross_attention", _input_fn), ("cross_attention_mask", _masked_input_fn)),
)
def test_cross_attention_benchmark(op_name, input_fn):
    bench = CrossAttentionBenchmark(
        op_name=op_name,
        torch_op=_torch_attention,
        gems_op=flag_gems.cross_attention,
        input_fn=input_fn,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
