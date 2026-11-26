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

"""Accuracy and batch-invariance tests for fused.paged_decode_attention.

Batch invariance: every (seq, kv_head, segment) is processed independently and
the segment length only depends on that sequence's ``seq_lens``, so a sequence
must produce the *same bits* no matter which other sequences share the batch.
That property is what makes the operator reproducible in serving and keeps A/B
comparisons clean; it is asserted here bitwise.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import paged_decode_attention

from . import accuracy_utils as utils

HQ, HKV, D, BS = 16, 2, 128, 16


def _make_cache(num_pages, dtype, device):
    k = torch.randn(num_pages, BS, HKV, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(num_pages, BS, HKV, D, dtype=dtype, device=device) * 0.1
    return k, v


def _ref(q, k_cache, v_cache, block_table, seq_lens, scale):
    """fp32 reference: gather pages per sequence, GQA softmax, weighted sum."""
    B, _, _ = q.shape
    gqa = HQ // HKV
    outs = []
    for b in range(B):
        n = int(seq_lens[b])
        nblk = (n + BS - 1) // BS
        pages = block_table[b, :nblk].long()
        kk = k_cache[pages].reshape(-1, HKV, D)[:n].float()
        vv = v_cache[pages].reshape(-1, HKV, D)[:n].float()
        qg = q[b].float().view(HKV, gqa, D)
        s = torch.einsum("hgd,khd->hgk", qg, kk) * scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hgk,khd->hgd", p, vv).reshape(HQ, D))
    return torch.stack(outs)


@pytest.mark.paged_decode_attention
@pytest.mark.parametrize("kv_len", [1024, 4096])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_decode_attention_accuracy(kv_len, ragged, dtype):
    device = flag_gems.device
    B = 8
    torch.manual_seed(0)
    nblk = kv_len // BS
    k_cache, v_cache = _make_cache(B * nblk, dtype, device)
    block_table = torch.arange(B * nblk, dtype=torch.int32,
                               device=device).view(B, nblk).contiguous()
    lens = torch.full((B,), kv_len, dtype=torch.int32, device=device)
    if ragged:
        lens = (lens - torch.randint(0, 64, (B,), device=device).to(torch.int32))
    q = torch.randn(B, HQ, D, dtype=dtype, device=device)
    scale = D ** -0.5
    out = torch.empty_like(q)

    paged_decode_attention(q, k_cache, v_cache, out, block_table, lens, scale,
                           num_segments=32, block_n=BS, num_warps=1)
    torch.cuda.synchronize()

    ref = _ref(q, k_cache, v_cache, block_table, lens, scale)
    utils.gems_assert_close(out, ref, dtype, atol=2e-2)


@pytest.mark.paged_decode_attention
def test_paged_decode_attention_batch_invariant():
    """Same sequences, different batch compositions -> bitwise identical rows."""
    device = flag_gems.device
    dtype = torch.bfloat16
    B, max_kv = 12, 2048
    torch.manual_seed(1)
    nblk = max_kv // BS
    k_cache, v_cache = _make_cache(B * nblk, dtype, device)
    block_table = torch.arange(B * nblk, dtype=torch.int32,
                               device=device).view(B, nblk).contiguous()
    # ragged lengths so segment boundaries differ per sequence
    lens = torch.tensor([max_kv - 137 * i for i in range(B)],
                        dtype=torch.int32, device=device)
    q = torch.randn(B, HQ, D, dtype=dtype, device=device)
    scale = D ** -0.5

    def run(idx):
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        qs = q.index_select(0, idx_t)
        bt = block_table.index_select(0, idx_t)
        sl = lens.index_select(0, idx_t)
        out = torch.empty_like(qs)
        paged_decode_attention(qs, k_cache, v_cache, out, bt, sl, scale,
                               num_segments=32, block_n=BS, num_warps=1)
        torch.cuda.synchronize()
        return out

    full = run(list(range(B)))
    # batch of one, two mid-size batches, and a permuted full batch
    for idx in ([0], [B - 1], [3, 7, 11], list(range(B // 2)),
                list(reversed(range(B)))):
        out = run(idx)
        back = run(idx)  # same composition twice: must be deterministic too
        for row, seq in enumerate(idx):
            utils.gems_assert_equal(out[row], full[seq])
            utils.gems_assert_equal(back[row], out[row])
