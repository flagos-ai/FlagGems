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
from flag_gems.fused import fused_deepseek_v4_qnorm_rope_kv_rope_insert

device = flag_gems.device

# DeepSeek-V4 model constants
HEAD_DIM = 512  # hardcoded in kernel
ROPE_DIM = 64  # hardcoded in kernel
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448

RTOL_BF16 = 1.6e-2
ATOL_Q = 1e-4 * HEAD_DIM  # 0.0512, scaled by head_dim for FP accumulation
ATOL_KV = 1e-2


# ── PyTorch reference ──────────────────────────────────────────────────


def _rms_norm_no_weight(x, eps):
    """RMSNorm without learnable weight, matching kernel behavior."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return x_normed.to(torch.bfloat16)


def _gptj_rope(x, cos_sin_cache, position_ids, nope_dim=448):
    """GPT-J style RoPE on the last ROPE_DIM=64 dims (interleaved pairs)."""
    rope_dim = x.shape[-1] - nope_dim
    half_rope = rope_dim // 2

    cos = cos_sin_cache[position_ids, :half_rope]
    sin = cos_sin_cache[position_ids, half_rope:]

    if x.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rope_part = x[..., nope_dim:]
    x_even = rope_part[..., 0::2].float()
    x_odd = rope_part[..., 1::2].float()

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    result = x.clone()
    result[..., nope_dim::2] = out_even.to(torch.bfloat16)
    result[..., nope_dim + 1 :: 2] = out_odd.to(torch.bfloat16)
    return result


def _reference_impl(
    q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, eps, cache_block_size
):
    """PyTorch reference: modifies q and k_cache in-place."""
    N_insert = slot_mapping.shape[0]

    # Q side: RMSNorm + RoPE
    q_normed = _rms_norm_no_weight(q, eps)
    q_result = _gptj_rope(q_normed, cos_sin_cache, position_ids)
    q.copy_(q_result)

    # KV side: RoPE + bf16 cache insert
    kv_roped = _gptj_rope(kv[:N_insert], cos_sin_cache, position_ids[:N_insert])
    for i in range(N_insert):
        slot_id = slot_mapping[i].item()
        if slot_id < 0:
            continue
        block_idx = slot_id // cache_block_size
        pos_in_block = slot_id % cache_block_size
        k_cache[block_idx, pos_in_block, :] = kv_roped[i]


# ── Helpers ────────────────────────────────────────────────────────────


def _make_cos_sin_cache(max_pos, rope_dim, device):
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _make_inputs(N, H, cache_block_size=16, max_pos=4096):
    q = torch.randn(N, H, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(N, HEAD_DIM, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(N, device=device, dtype=torch.int64)
    slot_mapping = torch.arange(N, device=device, dtype=torch.int64)
    num_blocks = (N + cache_block_size - 1) // cache_block_size + 1
    k_cache = torch.zeros(
        num_blocks, cache_block_size, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, device)
    return q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_insert
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
@pytest.mark.parametrize("n_heads", [8, 64])
def test_q_path(num_tokens, n_heads):
    """Q path: RMSNorm + GPT-J RoPE correctness."""
    torch.manual_seed(0)
    eps, cbs = 1e-6, 16
    q, kv, k_cache, _, position_ids, cos_sin_cache = _make_inputs(
        num_tokens, n_heads, cbs
    )
    slot_dummy = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)

    q_ref = q.clone()
    kc_ref = k_cache.clone()
    _reference_impl(
        q_ref, kv, kc_ref, slot_dummy, position_ids, cos_sin_cache, eps, cbs
    )

    q_tri = q.clone()
    kc_tri = k_cache.clone()
    fused_deepseek_v4_qnorm_rope_kv_rope_insert(
        q_tri, kv, kc_tri, slot_dummy, position_ids, cos_sin_cache, eps, cbs
    )

    torch.testing.assert_close(q_tri, q_ref, rtol=RTOL_BF16, atol=ATOL_Q)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_insert
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
@pytest.mark.parametrize("block_size", [16, 64])
def test_kv_path(num_tokens, block_size):
    """KV path: GPT-J RoPE + bf16 cache insert correctness."""
    torch.manual_seed(1)
    eps = 1e-6
    q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(
        num_tokens, 1, block_size
    )

    q_ref = q.clone()
    kc_ref = k_cache.clone()
    _reference_impl(
        q_ref,
        kv,
        kc_ref,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        block_size,
    )

    q_tri = q.clone()
    kc_tri = k_cache.clone()
    fused_deepseek_v4_qnorm_rope_kv_rope_insert(
        q_tri,
        kv,
        kc_tri,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        block_size,
    )

    torch.testing.assert_close(kc_tri, kc_ref, rtol=RTOL_BF16, atol=ATOL_KV)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_insert
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
@pytest.mark.parametrize("n_heads", [8, 128])
@pytest.mark.parametrize("block_size", [16, 64])
def test_combined(num_tokens, n_heads, block_size):
    """Combined Q + KV paths in a single fused call."""
    torch.manual_seed(2)
    eps = 1e-6
    q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(
        num_tokens, n_heads, block_size
    )

    q_ref = q.clone()
    kc_ref = k_cache.clone()
    _reference_impl(
        q_ref,
        kv,
        kc_ref,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        block_size,
    )

    q_tri = q.clone()
    kc_tri = k_cache.clone()
    fused_deepseek_v4_qnorm_rope_kv_rope_insert(
        q_tri,
        kv,
        kc_tri,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        block_size,
    )

    torch.testing.assert_close(q_tri, q_ref, rtol=RTOL_BF16, atol=ATOL_Q)
    torch.testing.assert_close(kc_tri, kc_ref, rtol=RTOL_BF16, atol=ATOL_KV)
