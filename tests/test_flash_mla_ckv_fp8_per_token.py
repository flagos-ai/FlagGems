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

from flag_gems.fused.flash_mla_ckv_fp8_per_token import (
    HAS_TLE,
    flash_mla_ckv_fp8_per_token,
    prepare_flash_mla_ckv_fp8_per_token,
    quantize_k_ckv_per_token,
    quantize_q_ckv_per_token,
)

from . import conftest as cfg


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


pytestmark = pytest.mark.skipif(
    not (HAS_TLE and _is_hopper()),
    reason="requires an NVIDIA Hopper GPU and FlagTree TLE support",
)


def _make_inputs(batch, h_q, seqlen):
    device = torch.device("cuda")
    pages_per_row = math.ceil(seqlen / 64)
    total_pages = batch * pages_per_row
    torch.manual_seed(42)

    q = torch.randn(batch, 1, h_q, 576, dtype=torch.bfloat16, device=device) * 0.1
    blocked_k = (
        torch.randn(total_pages, 64, 576, dtype=torch.bfloat16, device=device) * 0.1
    )
    q_nope, q_rope, q_scale = quantize_q_ckv_per_token(q)
    k_lora, k_rope, k_scale = quantize_k_ckv_per_token(blocked_k)
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, pages_per_row
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    return {
        "q": q,
        "blocked_k": blocked_k,
        "q_nope": q_nope,
        "q_rope": q_rope,
        "q_scale": q_scale,
        "k_lora": k_lora,
        "k_rope": k_rope,
        "k_scale": k_scale,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
        "lengths": (seqlen,) * batch,
    }


def _reference(inputs):
    q = inputs["q"].float()
    blocked_k = inputs["blocked_k"].float()
    block_table = inputs["block_table"]
    cache_seqlens = inputs["cache_seqlens"]
    batch, _, h_q, _ = q.shape
    out = torch.empty(batch, 1, h_q, 512, dtype=torch.float32, device=q.device)
    lse = torch.empty(batch, h_q, 1, dtype=torch.float32, device=q.device)
    softmax_scale = 576**-0.5

    for batch_idx in range(batch):
        seqlen = int(cache_seqlens[batch_idx].item())
        page_count = math.ceil(seqlen / 64)
        page_ids = block_table[batch_idx, :page_count].long()
        kv = blocked_k.index_select(0, page_ids).reshape(-1, 576)[:seqlen]
        scores = torch.matmul(q[batch_idx, 0], kv.transpose(0, 1))
        scores *= softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        out[batch_idx, 0] = torch.matmul(probabilities, kv[:, :512])
        lse[batch_idx, :, 0] = torch.logsumexp(scores, dim=-1)
    return out, lse


def _assert_close(out, lse, ref_out, ref_lse):
    out_f32 = out.float()
    rel_l2 = torch.linalg.vector_norm(out_f32 - ref_out) / torch.linalg.vector_norm(
        ref_out
    ).clamp_min(1e-12)
    cosine_distance = 1.0 - torch.nn.functional.cosine_similarity(
        out_f32.flatten(), ref_out.flatten(), dim=0
    )
    lse_max_abs = (lse.float() - ref_lse).abs().max()
    assert rel_l2.item() <= 5e-2
    assert cosine_distance.item() <= 1e-3
    assert lse_max_abs.item() <= 2e-2


CASES = [(1, 64, 128)] if cfg.QUICK_MODE else [(1, 64, 128), (2, 64, 640)]


@pytest.mark.parametrize("batch,h_q,seqlen", CASES)
def test_flash_mla_ckv_fp8_per_token_accuracy(batch, h_q, seqlen):
    inputs = _make_inputs(batch, h_q, seqlen)
    out, lse = flash_mla_ckv_fp8_per_token(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
    )
    ref_out, ref_lse = _reference(inputs)
    _assert_close(out, lse, ref_out, ref_lse)


def test_flash_mla_ckv_fp8_per_token_prepared_outputs_are_deterministic():
    inputs = _make_inputs(1, 64, 128)
    handle, (fresh_out, fresh_lse) = prepare_flash_mla_ckv_fp8_per_token(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    out = torch.empty_like(fresh_out)
    lse = torch.empty_like(fresh_lse)
    caller_out, caller_lse = handle(out=out, lse=lse)
    expected_out = caller_out.clone()
    expected_lse = caller_lse.clone()
    replay_out, replay_lse = handle(out=out, lse=lse)

    assert torch.equal(fresh_out, expected_out)
    assert torch.equal(fresh_lse, expected_lse)
    assert torch.equal(replay_out, expected_out)
    assert torch.equal(replay_lse, expected_lse)
