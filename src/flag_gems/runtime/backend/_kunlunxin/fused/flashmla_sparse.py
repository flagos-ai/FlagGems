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

from typing import Optional, Tuple

import torch


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    safe_indices = indices.squeeze(1).clone()
    if topk_length is not None:
        length_mask = torch.arange(TOPK, device=q.device).unsqueeze(0) >= (
            topk_length.unsqueeze(1)
        )
        safe_indices.masked_fill_(length_mask, -1)
    invalid_mask = (safe_indices < 0) | (safe_indices >= SKV)
    safe_indices.masked_fill_(invalid_mask, 0)

    gathered_kv = (
        kv.index_select(0, safe_indices.flatten()).reshape(SQ, TOPK, DQK).float()
    )
    logits = torch.matmul(q.float(), gathered_kv.transpose(1, 2)) * sm_scale
    logits = logits.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))
    original_lse = torch.logsumexp(logits, dim=-1)
    max_logits = torch.max(logits, dim=-1).values

    lse_for_output = original_lse
    if attn_sink is not None:
        sink_lse = attn_sink.unsqueeze(0).expand(SQ, HQ)
        lse_for_output = torch.logsumexp(
            torch.stack((original_lse, sink_lse), dim=0), dim=0
        )
    lse_for_output = torch.where(
        torch.isneginf(lse_for_output),
        torch.full_like(lse_for_output, float("inf")),
        lse_for_output,
    )
    probabilities = torch.exp(logits - lse_for_output.unsqueeze(-1))
    output = torch.matmul(probabilities, gathered_kv[..., :d_v])
    lse = torch.where(
        torch.isneginf(original_lse),
        torch.full_like(original_lse, float("inf")),
        original_lse,
    )
    return output.to(torch.bfloat16), max_logits, lse
