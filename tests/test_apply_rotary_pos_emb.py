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
    )

    utils.gems_assert_close(q_embed_out, q_embed_ref, dtype)
    utils.gems_assert_close(k_embed_out, k_embed_ref, dtype)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("q_heads,k_heads", [(8, 2), (1, 1)])
@pytest.mark.parametrize(
    "rotary_interleaved,has_pos_id", [(False, False), (True, True)]
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_apply_rotary_pos_emb_inplace(
    dtype, q_heads, k_heads, rotary_interleaved, has_pos_id
):
    batch_size = 2
    head_dim = 96
    max_seq_len = 64
    cos, sin = _get_rope_cos_sin(max_seq_len, head_dim, dtype, device=flag_gems.device)

    # Both token counts map to the same tuning bucket. This also checks that a
    # cached kernel still launches the exact grid required by the current input.
    for seq_len in (17, 31):
        q = torch.randn(
            (batch_size, seq_len, q_heads, head_dim),
            dtype=dtype,
            device=flag_gems.device,
        )
        k = torch.randn(
            (batch_size, seq_len, k_heads, head_dim),
            dtype=dtype,
            device=flag_gems.device,
        )
        position_ids = torch.randint(
            0, max_seq_len, (batch_size, seq_len), device=flag_gems.device
        )

        q_embed_ref, k_embed_ref = _torch_apply_rotary_pos_emb(
            q=utils.to_reference(q, True),
            k=utils.to_reference(k, True),
            cos=utils.to_reference(cos, True),
            sin=utils.to_reference(sin, True),
            position_ids=utils.to_reference(position_ids) if has_pos_id else None,
            rotary_interleaved=rotary_interleaved,
        )
        q_data_ptr = q.data_ptr()
        k_data_ptr = k.data_ptr()

        q_embed_out, k_embed_out = flag_gems.apply_rotary_pos_emb(
            q=q,
            k=k,
            cos=cos,
            sin=sin,
            position_ids=position_ids if has_pos_id else None,
            rotary_interleaved=rotary_interleaved,
            inplace=True,
        )

        assert q_embed_out.data_ptr() == q_data_ptr
        assert k_embed_out.data_ptr() == k_data_ptr
        utils.gems_assert_close(q_embed_out, q_embed_ref, dtype)
        utils.gems_assert_close(k_embed_out, k_embed_ref, dtype)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.skipif(not utils.bf16_is_supported, reason="Requires BF16 support")
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_apply_rotary_pos_emb_inplace_long_sequence_deterministic():
    # Reproduce the long-input race reported on AMD gfx1201. Fresh clones are
    # required: applying RoPE repeatedly to an already rotated input is different.
    seq_len = 300 if cfg.TO_CPU or cfg.QUICK_MODE else 1336
    dtype = torch.bfloat16
    q = torch.randn((seq_len, 64, 64), dtype=dtype, device=flag_gems.device)
    k = torch.randn((seq_len, 8, 64), dtype=dtype, device=flag_gems.device)
    position_ids = torch.arange(seq_len, device=flag_gems.device)
    cos, sin = _get_rope_cos_sin(seq_len, 64, dtype, base=150000)
    q_ref, k_ref = _torch_apply_rotary_pos_emb(
        utils.to_reference(q, True),
        utils.to_reference(k, True),
        utils.to_reference(cos, True),
        utils.to_reference(sin, True),
        utils.to_reference(position_ids),
    )

    first_result = None
    for _ in range(3 if cfg.QUICK_MODE else 6):
        q_input, k_input = q.clone(), k.clone()
        q_output, k_output = flag_gems.apply_rotary_pos_emb(
            q_input, k_input, cos, sin, position_ids, inplace=True
        )
        assert q_output.data_ptr() == q_input.data_ptr()
        assert k_output.data_ptr() == k_input.data_ptr()
        utils.gems_assert_close(q_output, q_ref, dtype)
        utils.gems_assert_close(k_output, k_ref, dtype)
        if first_result is None:
            first_result = (q_output, k_output)
        else:
            torch.testing.assert_close(q_output, first_result[0], rtol=0, atol=0)
            torch.testing.assert_close(k_output, first_result[1], rtol=0, atol=0)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.skipif(not utils.bf16_is_supported, reason="Requires BF16 support")
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("nvidia", "amd"),
    reason="PT2 Triton compilation requires a CUDA/ROCm Torch build",
)
def test_apply_rotary_pos_emb_inplace_pt2(rotary_interleaved):
    from flag_gems.pt2.rotary_embedding import (
        rotary_embedding_inplace,
        supports_pt2_triton,
    )

    if not hasattr(torch, "compile") or not supports_pt2_triton():
        pytest.skip("Requires torch.compile and torch.library Triton APIs")

    compiled = torch.compile(rotary_embedding_inplace, fullgraph=True)
    dtype = torch.bfloat16
    cos, sin = _get_rope_cos_sin(64, 96, dtype)
    # Exercise the direct raw-kernel launch and the grid after the token count
    # changes. The public eager autotuner does not cover this PT2 entry point.
    for seq_len in ((17,) if cfg.TO_CPU or cfg.QUICK_MODE else (17, 31)):
        q = torch.randn((seq_len, 8, 96), dtype=dtype, device=flag_gems.device)
        k = torch.randn((seq_len, 2, 96), dtype=dtype, device=flag_gems.device)
        position_ids = torch.arange(seq_len, device=flag_gems.device)
        q_ref, k_ref = _torch_apply_rotary_pos_emb(
            utils.to_reference(q, True),
            utils.to_reference(k, True),
            utils.to_reference(cos, True),
            utils.to_reference(sin, True),
            utils.to_reference(position_ids),
            rotary_interleaved,
        )
        q_eager, k_eager = q.clone(), k.clone()
        rotary_embedding_inplace(
            q_eager, k_eager, cos, sin, position_ids, rotary_interleaved
        )
        q_compiled, k_compiled = q.clone(), k.clone()
        q_data_ptr, k_data_ptr = q_compiled.data_ptr(), k_compiled.data_ptr()
        compiled(q_compiled, k_compiled, cos, sin, position_ids, rotary_interleaved)

        assert q_compiled.data_ptr() == q_data_ptr
        assert k_compiled.data_ptr() == k_data_ptr
        utils.gems_assert_close(q_eager, q_ref, dtype)
        utils.gems_assert_close(k_eager, k_ref, dtype)
        torch.testing.assert_close(q_compiled, q_eager, rtol=0, atol=0)
        torch.testing.assert_close(k_compiled, k_eager, rtol=0, atol=0)
