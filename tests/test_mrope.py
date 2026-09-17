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

MAX_POSITION = 32768
BASE = 1000000.0

# (num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim,
#  section_t, section_h, section_w, is_neox_style,
#  is_interleaved, is_interleaved_glm)
DECODE_CONFIGS = [
    (1, 28, 4, 128, 128, 24, 20, 20, True, False, False),
    (1, 28, 4, 128, 128, 24, 20, 20, True, True, False),
    (1, 28, 4, 128, 64, 8, 12, 12, True, True, True),
]

PREFILL_CONFIGS = [
    (512, 28, 4, 128, 128, 24, 20, 20, True, False, False),
    (8192, 28, 4, 128, 128, 24, 20, 20, True, False, False),
    (4096, 40, 8, 128, 128, 24, 20, 20, True, False, False),
    (2048, 32, 4, 128, 64, 8, 12, 12, False, False, False),
    (512, 28, 4, 128, 128, 24, 20, 20, True, True, False),
    (8192, 28, 4, 128, 128, 24, 20, 20, True, True, False),
    (512, 28, 4, 128, 64, 8, 12, 12, True, True, True),
    (2048, 16, 16, 128, 64, 8, 12, 12, True, True, True),
]

ALL_CONFIGS = DECODE_CONFIGS + PREFILL_CONFIGS


def _compute_cos_sin_cache(max_position, rotary_dim, base, dtype):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    t = torch.arange(max_position, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return cos_sin_cache.to(dtype)


def _compute_axis_map(mrope_section, rotary_dim):
    num_pairs = rotary_dim // 2
    axis_map = torch.empty(num_pairs, dtype=torch.long)
    counts = [0, 0, 0]
    for i in range(num_pairs):
        current_ax = i % 3
        while counts[current_ax] >= mrope_section[current_ax]:
            current_ax = (current_ax + 1) % 3
        axis_map[i] = current_ax
        counts[current_ax] += 1
    return axis_map


def _apply_rotary_emb(x, cos, sin, is_neox_style):
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1).to(orig_dtype)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2).to(orig_dtype)


def _apply_interleaved_rope(x, mrope_section):
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def torch_mrope(
    query,
    key,
    cos_sin_cache,
    positions,
    section_t,
    section_h,
    section_w,
    head_size,
    rotary_dim,
    is_interleaved=False,
    is_interleaved_glm=False,
    is_neox_style=True,
    axis_map=None,
):
    mrope_section = [section_t, section_h, section_w]
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    if is_interleaved:
        if is_interleaved_glm:
            idx = axis_map[None, None, :].expand(1, cos.shape[1], -1)
            cos = torch.gather(cos, 0, idx).squeeze(0)
            sin = torch.gather(sin, 0, idx).squeeze(0)
        else:
            cos = _apply_interleaved_rope(cos, mrope_section)
            sin = _apply_interleaved_rope(sin, mrope_section)
    else:
        cos = torch.cat(
            [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1,
        )

    num_tokens = query.shape[0]
    query_shape = query.shape
    key_shape = key.shape

    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = _apply_rotary_emb(key_rot, cos, sin, is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


def _build_inputs(config):
    (
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_size,
        rotary_dim,
        section_t,
        section_h,
        section_w,
        is_neox_style,
        is_interleaved,
        is_interleaved_glm,
    ) = config

    torch.manual_seed(42)
    dtype = torch.bfloat16
    positions = torch.randint(0, MAX_POSITION, (3, num_tokens), device=flag_gems.device)
    query = torch.randn(
        num_tokens, num_q_heads * head_size, device=flag_gems.device, dtype=dtype
    )
    key = torch.randn(
        num_tokens, num_kv_heads * head_size, device=flag_gems.device, dtype=dtype
    )
    cos_sin_cache = _compute_cos_sin_cache(MAX_POSITION, rotary_dim, BASE, dtype).to(
        flag_gems.device
    )
    axis_map = None
    if is_interleaved_glm:
        axis_map = _compute_axis_map([section_t, section_h, section_w], rotary_dim).to(
            flag_gems.device
        )
    return positions, query, key, cos_sin_cache, axis_map


@pytest.mark.mrope
@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_mrope_vs_torch(config):
    (
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_size,
        rotary_dim,
        section_t,
        section_h,
        section_w,
        is_neox_style,
        is_interleaved,
        is_interleaved_glm,
    ) = config

    positions, query, key, cos_sin_cache, axis_map = _build_inputs(config)

    ref_axis_map = utils.to_reference(axis_map) if axis_map is not None else None

    ref_q, ref_k = torch_mrope(
        utils.to_reference(query.clone()),
        utils.to_reference(key.clone()),
        utils.to_reference(cos_sin_cache),
        utils.to_reference(positions),
        section_t,
        section_h,
        section_w,
        head_size,
        rotary_dim,
        is_interleaved,
        is_interleaved_glm,
        is_neox_style,
        ref_axis_map,
    )
    from flag_gems.fused.mrope import mrope as gems_mrope

    res_q, res_k = gems_mrope(
        query,
        key,
        cos_sin_cache,
        positions,
        section_t,
        section_h,
        section_w,
        head_size,
        rotary_dim,
        is_interleaved,
        is_interleaved_glm,
        is_neox_style,
        axis_map,
    )

    utils.gems_assert_close(res_q, ref_q, torch.bfloat16, reduce_dim=head_size)
    utils.gems_assert_close(res_k, ref_k, torch.bfloat16, reduce_dim=head_size)
