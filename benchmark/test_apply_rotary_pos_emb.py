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

from typing import Optional

import pytest
import torch

import flag_gems

from . import base, consts


class InplaceRopeBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (tokens, q_heads, k_heads, head_dim, interleaved)
            for tokens in [1, 128, 2048, 8192]
            for q_heads, k_heads in [(1, 1), (8, 1), (16, 4)]
            for head_dim in [64, 128, 256]
            for interleaved in [False, True]
        ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.set_more_shapes()
        self.shape_desc = "tokens, q_heads, k_heads, head_dim, interleaved"


def inplace_rope_input_fn(shape, dtype, device):
    tokens, q_heads, k_heads, head_dim, interleaved = shape
    q = torch.randn(tokens, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(tokens, k_heads, head_dim, device=device, dtype=dtype)
    # Exact quarter turns avoid overflow/underflow during repeated in-place runs.
    cos = torch.zeros(4096, head_dim // 2, device=device)
    sin = torch.ones_like(cos)
    positions = torch.randint(0, 4096, (tokens,), device=device)
    yield q, k, cos, sin, positions, {
        "rotary_interleaved": interleaved,
        "inplace": True,
    }


def torch_inplace_rope(q, k, cos, sin, positions, rotary_interleaved, inplace):
    c = cos[positions].unsqueeze(1)
    s = sin[positions].unsqueeze(1)

    def rotate(x):
        if rotary_interleaved:
            a, b = x.float()[..., ::2], x.float()[..., 1::2]
            return torch.stack((a * c - b * s, b * c + a * s), dim=-1).flatten(-2)
        a, b = x.float().chunk(2, dim=-1)
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1)

    q.copy_(rotate(q))
    k.copy_(rotate(k))
    return q, k


@pytest.mark.apply_rotary_pos_emb
def test_apply_rotary_pos_emb_inplace():
    bench = InplaceRopeBenchmark(
        input_fn=inplace_rope_input_fn,
        op_name="apply_rotary_pos_emb",
        torch_op=torch_inplace_rope,
        gems_op=flag_gems.apply_rotary_pos_emb,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class RopeBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return []


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def rope_input_fn(shape, dtype, device):
    batch_size = 4
    q_heads = 8
    k_heads = 1
    head_dim = 64

    seq_len = shape[0]
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device
    )
    cos, sin = get_rope_cos_sin(seq_len, head_dim, dtype, device=device)
    yield q, k, cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_fn(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    cos = cos[None, : q.size(-3), None, :]
    sin = sin[None, : q.size(-3), None, :]
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
    sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


@pytest.mark.apply_rotary_pos_emb
def test_apply_rotary_pos_emb():
    bench = RopeBenchmark(
        input_fn=rope_input_fn,
        op_name="apply_rotary_pos_emb",
        torch_op=torch_apply_rotary_pos_emb,
        gems_op=flag_gems.apply_rotary_pos_emb,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
