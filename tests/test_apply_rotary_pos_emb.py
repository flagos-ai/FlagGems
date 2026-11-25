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

import random
import time
from typing import Optional

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    if position_ids is None:
        cos = cos[None, : q.size(-3), None, :]
        sin = sin[None, : q.size(-3), None, :]
    else:
        cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
        sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


def _get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    return cos, sin


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("batch_size", [2] if cfg.TO_CPU or cfg.QUICK_MODE else [4, 8])
@pytest.mark.parametrize(
    "max_seq_len", [16] if cfg.TO_CPU or cfg.QUICK_MODE else [512, 2048]
)
@pytest.mark.parametrize(
    "q_heads,k_heads",
    [(8, 1)] if cfg.QUICK_MODE else [(8, 1), (6, 2), (1, 1), (8, 8)],
)
@pytest.mark.parametrize(
    "head_dim", [8] if cfg.TO_CPU or cfg.QUICK_MODE else [64, 96, 128, 256]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [True, False])
@pytest.mark.parametrize("has_pos_id", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_apply_rotary_pos_emb(
    batch_size,
    max_seq_len,
    q_heads,
    k_heads,
    head_dim,
    dtype,
    has_pos_id,
    rotary_interleaved,
    inplace,
):
    seq_len = torch.randint(1, max_seq_len, (1,)).item()
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=flag_gems.device
    )

    position_ids = torch.randint(
        0, max_seq_len, (batch_size, seq_len), device=flag_gems.device
    )
    cos, sin = _get_rope_cos_sin(max_seq_len, head_dim, dtype, device=flag_gems.device)

    ref_q = utils.to_reference(q, True)
    ref_k = utils.to_reference(k, True)
    ref_cos = utils.to_reference(cos, True)
    ref_sin = utils.to_reference(sin, True)
    ref_position_ids = utils.to_reference(position_ids)

    q_embed_ref, k_embed_ref = _torch_apply_rotary_pos_emb(
        q=ref_q,
        k=ref_k,
        cos=ref_cos,
        sin=ref_sin,
        position_ids=ref_position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
    )
    q_embed_out, k_embed_out = flag_gems.apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )

    if inplace:
        assert q_embed_out.data_ptr() == q.data_ptr()
        assert k_embed_out.data_ptr() == k.data_ptr()
    utils.gems_assert_close(q_embed_out, q_embed_ref, dtype)
    utils.gems_assert_close(k_embed_out, k_embed_ref, dtype)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("head_dim", [128] if cfg.QUICK_MODE else [128, 160, 192, 256])
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_apply_rotary_pos_emb_inplace_race(head_dim, rotary_interleaved):
    # Keep the issue-sized workload even with --quick/--ref cpu: the old race
    # needs concurrent tokens and a head dimension spanning multiple warps.
    dtype = torch.bfloat16 if utils.bf16_is_supported else torch.float16
    n_tokens, q_heads, k_heads, max_position = 2048, 16, 4, 4096
    generator = torch.Generator().manual_seed(9344)
    q_cpu = torch.randn(n_tokens, q_heads, head_dim, generator=generator).to(dtype)
    k_cpu = torch.randn(n_tokens, k_heads, head_dim, generator=generator).to(dtype)
    positions_cpu = torch.randint(max_position, (n_tokens,), generator=generator)
    inv_freq = 1.0 / (
        1e6 ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float64), inv_freq)
    cos_cpu, sin_cpu = freqs.cos().float(), freqs.sin().float()
    cache = torch.cat((cos_cpu, sin_cpu), dim=-1).to(flag_gems.device)
    cos, sin = cache.chunk(2, dim=-1)
    positions_storage = torch.empty(
        n_tokens * 2, device=flag_gems.device, dtype=torch.int64
    )
    positions = positions_storage[::2]
    positions.copy_(positions_cpu)
    c = cos_cpu[positions_cpu].double().unsqueeze(1)
    s = sin_cpu[positions_cpu].double().unsqueeze(1)

    def reference(x):
        x = x.double()
        if rotary_interleaved:
            a, b = x[..., ::2], x[..., 1::2]
            return torch.stack((a * c - b * s, b * c + a * s), dim=-1).flatten(-2)
        a, b = x.chunk(2, dim=-1)
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1)

    ref_q, ref_k = reference(q_cpu), reference(k_cpu)
    # The corruption is timing-dependent. Each invocation starts with the same
    # original inputs and checks every Q and K element against CPU FP64.
    for _ in range(3):
        q = q_cpu.to(flag_gems.device)
        k = k_cpu.to(flag_gems.device)
        out_q, out_k = flag_gems.apply_rotary_pos_emb(
            q, k, cos, sin, positions, rotary_interleaved, inplace=True
        )
        assert out_q.data_ptr() == q.data_ptr()
        assert out_k.data_ptr() == k.data_ptr()
        torch.testing.assert_close(out_q.cpu().double(), ref_q, atol=0.032, rtol=0)
        torch.testing.assert_close(out_k.cpu().double(), ref_k, atol=0.032, rtol=0)
