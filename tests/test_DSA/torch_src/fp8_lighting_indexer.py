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

# ruff: noqa
import torch


def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    # logits[m, n] = sum_h relu(q[m,h,:] @ k[n,:]) * weights[m,h].
    # Accumulate head-by-head so we never materialize the [H, S, SKV] score
    # tensor (its fp32 peak is ~8.6 GiB at the largest test shapes and OOMs
    # the GPU; issue #2353). Per-iteration peak is a single [S, SKV] block.
    S, H, _ = q.shape
    logits = torch.zeros((S, seq_len_kv), dtype=torch.float32, device="cuda")
    for h in range(H):
        score_h = torch.einsum("md,nd->mn", q[:, h, :], k)
        logits += score_h.relu() * weights[:, h].unsqueeze(-1)
    logits = logits.masked_fill(~mask, float("-inf"))

    cost = mask.sum()
    return logits, cost
