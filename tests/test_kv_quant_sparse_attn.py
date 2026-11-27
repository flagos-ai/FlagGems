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

"""Tests for kv_quant_sparse_flash_attention (Ascend-only, MLA-absorb + fp8 per-tile KV).

Tests the Triton replacement for torch_npu.npu_kv_quant_sparse_flash_attention
in the decode form used by vllm-ascend.

Categories:
  - Accuracy vs CANN baseline (model shapes)
  - Accuracy vs torch fp32 reference
  - Boundary shapes (small sparse, single token, edge splits)
  - Input validation (bad dtypes, shapes, missing args)
  - Cache view correctness
"""

import os

import pytest
import torch

import flag_gems

NOPE = 512
ROPE = 64
TILE = 128
NTILE = NOPE // TILE
SLOT = NOPE + ROPE * 2 + NTILE * 4   # 656

ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="kv_quant_sparse_flash_attention targets Ascend 950",
)

pytestmark = [pytest.mark.kv_quant_sparse_flash_attention, ascend_only]

QUICK_MODE = os.environ.get("QUICK_MODE", "0") == "1"

# ─── helpers ───────────────────────────────────────────────────────────

def _build_inputs(heads=64, kv_len=8192, sparse=2048, block_size=128,
                  block_num=None, batch=1, seed=0, fill_neg1=0,
                  device=None):
    """Build fp8 per-tile quantized packed KV cache + sparse indices.

    Args:
        fill_neg1: number of -1 sentinels to place at the end of sparse_indices.
    Returns:
        query, cache, sparse_indices, block_table, actual_seq_lengths_kv, meta
    """
    if device is None:
        device = flag_gems.device
    if block_num is None:
        block_num = max(kv_len // block_size + 16, kv_len // block_size)

    g = torch.Generator().manual_seed(seed)
    n_blk = kv_len // block_size

    # query: [batch, heads, 576] bf16
    q_nope = (torch.randn(batch, heads, NOPE, generator=g) * 1.0).to(torch.bfloat16)
    q_rope = (torch.randn(batch, heads, ROPE, generator=g) * 1.0).to(torch.bfloat16)
    query = torch.cat([q_nope, q_rope], dim=-1).to(device)

    # logical KV
    k_nope = (torch.randn(kv_len, NOPE, generator=g) * 2.0).to(torch.float8_e4m3fn)
    k_rope = (torch.randn(kv_len, ROPE, generator=g)).to(torch.bfloat16)
    scales = (torch.rand(kv_len, NTILE, generator=g) * 0.1 + 0.05).float()

    # packed cache: [block_num, block_size, 1, 656] as uint8
    cache_u8 = torch.zeros(block_num, block_size, 1, SLOT, dtype=torch.uint8)
    flat = cache_u8.view(-1, SLOT)

    # random page table
    perm = torch.randperm(block_num, generator=g)[:n_blk].to(torch.int32)
    block_table = perm.reshape(1, n_blk)
    if batch > 1:
        block_table = block_table.expand(batch, -1).contiguous()

    rows = (perm.long()[torch.arange(kv_len) // block_size] * block_size
            + torch.arange(kv_len) % block_size)

    flat[rows, 0:NOPE] = k_nope.view(torch.uint8)
    flat[rows, NOPE:NOPE + ROPE * 2] = k_rope.contiguous().view(torch.uint8)
    flat[rows, NOPE + ROPE * 2:SLOT] = scales.contiguous().view(torch.uint8)

    cache = cache_u8.view(torch.float8_e4m3fn).to(device)

    # sparse_indices
    valid_count = sparse - fill_neg1
    assert valid_count > 0, "must have at least 1 valid index"
    idx_valid = torch.randperm(kv_len, generator=g)[:valid_count].to(torch.int32)
    if fill_neg1 > 0:
        idx_pad = torch.full((fill_neg1,), -1, dtype=torch.int32)
        idx = torch.cat([idx_valid, idx_pad])
    else:
        idx = idx_valid
    sparse_indices = idx.reshape(1, 1, sparse)
    if batch > 1:
        sparse_indices = sparse_indices.expand(batch, -1, -1).contiguous()
    sparse_indices = sparse_indices.to(device)

    actual_seq_lengths_kv = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    meta = dict(
        q_nope=q_nope, q_rope=q_rope, k_nope=k_nope, k_rope=k_rope,
        scales=scales, idx=idx, heads=heads, kv_len=kv_len,
        block_size=block_size, valid_count=valid_count,
    )
    return query, cache, sparse_indices, block_table.to(device), actual_seq_lengths_kv, meta


def _torch_reference(meta, batch_idx=0):
    """CPU fp32 reference matching test_a5_c8_lse."""
    q = torch.cat([meta["q_nope"], meta["q_rope"]], dim=-1)[batch_idx].float()
    sel = meta["idx"].long()
    sel = sel[:meta["valid_count"]]
    sel = sel[sel >= 0]
    deq = (meta["k_nope"].float() * meta["scales"].repeat_interleave(TILE, dim=-1)).to(torch.bfloat16)
    k = torch.cat([deq, meta["k_rope"]], dim=-1)[sel].float()
    scores = q @ k.T * ((NOPE + ROPE) ** -0.5)
    return torch.softmax(scores, dim=-1) @ deq[sel].float()   # [H, 512]


def _cann_baseline(query, cache, sparse_indices, block_table, kv_len, batch=1):
    """Run CANN baseline for comparison."""
    import torch_npu  # noqa: F811
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query, key=cache, value=cache,
        sparse_indices=sparse_indices,
        scale_value=(NOPE + ROPE) ** -0.5,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=torch.ones(batch, dtype=torch.int32, device=query.device),
        actual_seq_lengths_kv=torch.full((batch,), kv_len, dtype=torch.int32, device=query.device),
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2,
        quant_scale_repo_mode=1, tile_size=TILE,
        key_quant_mode=2, value_quant_mode=2,
        rope_head_dim=ROPE,
    )


# ─── accuracy vs CANN baseline ────────────────────────────────────────

MODEL_SHAPES = [
    # (name, heads, kv_len, sparse, block_size, block_num)
    ("GLM-W4", 64, 8192, 2048, 128, 4194),
    ("GLM-W8", 64, 8192, 2048, 128, 8778),
]

@pytest.mark.parametrize(
    "name,heads,kv_len,sparse,block_size,block_num",
    MODEL_SHAPES if not QUICK_MODE else MODEL_SHAPES[:1],
    ids=[s[0] for s in (MODEL_SHAPES if not QUICK_MODE else MODEL_SHAPES[:1])],
)
def test_accuracy_vs_cann(name, heads, kv_len, sparse, block_size, block_num):
    """Triton output matches CANN baseline within bf16 tolerance."""
    query, cache, si, bt, seqkv, meta = _build_inputs(
        heads=heads, kv_len=kv_len, sparse=sparse,
        block_size=block_size, block_num=block_num,
    )
    got = flag_gems.kv_quant_sparse_flash_attention(
        query, cache, si, bt, actual_seq_lengths_kv=seqkv,
    )
    base = _cann_baseline(query, cache, si, bt, kv_len)
    g = got[0].cpu().float()
    b = base[0].cpu().float()
    diff = (g - b).abs()
    rel = (diff.max() / b.abs().max().clamp(min=1e-6)).item()
    assert rel <= 2e-2, f"{name}: max_rel vs CANN = {rel:.3e} > 2e-2"


# ─── accuracy vs torch reference ──────────────────────────────────────

REF_SHAPES = [
    # (heads, kv_len, sparse, block_size, splits, block_h, block_s)
    (64, 8192, 2048, 128, 64, 64, 32),
    (64, 4096, 1024, 64, 32, 32, 32),
]

@pytest.mark.parametrize(
    "heads,kv_len,sparse,block_size,splits,block_h,block_s",
    REF_SHAPES if not QUICK_MODE else REF_SHAPES[:1],
)
def test_accuracy_vs_torch_ref(heads, kv_len, sparse, block_size, splits, block_h, block_s):
    """Triton output matches CPU fp32 reference within bf16 tolerance."""
    query, cache, si, bt, seqkv, meta = _build_inputs(
        heads=heads, kv_len=kv_len, sparse=sparse, block_size=block_size,
    )
    got = flag_gems.kv_quant_sparse_flash_attention(
        query, cache, si, bt, actual_seq_lengths_kv=seqkv,
        splits=splits, block_h=block_h, block_s=block_s,
    )
    ref = _torch_reference(meta)
    g = got[0].cpu().float()
    diff = (g - ref).abs()
    rel = (diff.max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel <= 2e-2, f"max_rel vs torch ref = {rel:.3e} > 2e-2"


# ─── boundary shapes ──────────────────────────────────────────────────

BOUNDARY_SHAPES = [
    # (heads, kv_len, sparse, block_size, splits, block_h, block_s)
    # Fewer splits: chunk=2048/32=64, block_s=32 → 2 iters, block_h=16 fits UB
    (64, 8192, 2048, 128, 32, 16, 32),
    # block_size=64, splits=32 → chunk=32, block_s=32 → 1 iter
    (64, 4096, 1024, 64, 32, 32, 32),
    # Different block_h: splits=64 block_h=32 → CHUNK=32=block_s → 1 iter
    (64, 8192, 2048, 128, 64, 32, 32),
]

@pytest.mark.parametrize(
    "heads,kv_len,sparse,block_size,splits,block_h,block_s",
    BOUNDARY_SHAPES if not QUICK_MODE else BOUNDARY_SHAPES[:2],
)
def test_accuracy_boundary_shapes(heads, kv_len, sparse, block_size, splits, block_h, block_s):
    """Boundary configurations produce correct results vs torch reference."""
    query, cache, si, bt, seqkv, meta = _build_inputs(
        heads=heads, kv_len=kv_len, sparse=sparse, block_size=block_size,
    )
    got = flag_gems.kv_quant_sparse_flash_attention(
        query, cache, si, bt, actual_seq_lengths_kv=seqkv,
        splits=splits, block_h=block_h, block_s=block_s,
    )
    ref = _torch_reference(meta)
    g = got[0].cpu().float()
    diff = (g - ref).abs()
    rel = (diff.max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel <= 2e-2, f"max_rel = {rel:.3e} > 2e-2 (splits={splits} bh={block_h} bs={block_s})"


# ─── neg-1 sentinel handling ──────────────────────────────────────────

def test_neg1_sentinel():
    """Slots with -1 indices are correctly ignored."""
    heads, kv_len, sparse, bs = 64, 8192, 2048, 128
    # half valid, half -1
    query, cache, si, bt, seqkv, meta = _build_inputs(
        heads=heads, kv_len=kv_len, sparse=sparse, block_size=bs,
        fill_neg1=sparse // 2,
    )
    got = flag_gems.kv_quant_sparse_flash_attention(
        query, cache, si, bt, actual_seq_lengths_kv=seqkv,
    )
    ref = _torch_reference(meta)
    g = got[0].cpu().float()
    diff = (g - ref).abs()
    rel = (diff.max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel <= 2e-2, f"max_rel with -1 sentinels = {rel:.3e} > 2e-2"


# ─── deq_f32 path ─────────────────────────────────────────────────────

def test_deq_f32_path():
    """DEQ_F32 dequantization path produces correct results."""
    query, cache, si, bt, seqkv, meta = _build_inputs(
        heads=64, kv_len=8192, sparse=2048, block_size=128,
    )
    # DEQ_F32 requires smaller block_h due to UB constraints
    got = flag_gems.kv_quant_sparse_flash_attention(
        query, cache, si, bt, actual_seq_lengths_kv=seqkv,
        deq_f32=True, deq3d=False, block_h=16, splits=64, block_s=32,
    )
    ref = _torch_reference(meta)
    g = got[0].cpu().float()
    diff = (g - ref).abs()
    rel = (diff.max() / ref.abs().max().clamp(min=1e-6)).item()
    assert rel <= 2e-2, f"max_rel DEQ_F32 = {rel:.3e} > 2e-2"


# ─── input validation ─────────────────────────────────────────────────

def test_rejects_non_bf16_query():
    query, cache, si, bt, seqkv, _ = _build_inputs()
    with pytest.raises(AssertionError, match="bf16"):
        flag_gems.kv_quant_sparse_flash_attention(
            query.half(), cache, si, bt, actual_seq_lengths_kv=seqkv,
        )

def test_rejects_non_fp8_cache():
    query, cache, si, bt, seqkv, _ = _build_inputs()
    with pytest.raises(AssertionError, match="fp8_e4m3fn"):
        flag_gems.kv_quant_sparse_flash_attention(
            query, cache.view(torch.uint8), si, bt, actual_seq_lengths_kv=seqkv,
        )

def test_rejects_wrong_query_dim():
    query, cache, si, bt, seqkv, _ = _build_inputs()
    with pytest.raises(AssertionError):
        flag_gems.kv_quant_sparse_flash_attention(
            query.squeeze(0), cache, si, bt, actual_seq_lengths_kv=seqkv,
        )

def test_rejects_missing_seq_lengths():
    query, cache, si, bt, seqkv, _ = _build_inputs()
    with pytest.raises(AssertionError, match="actual_seq_lengths_kv"):
        flag_gems.kv_quant_sparse_flash_attention(
            query, cache, si, bt,
        )

def test_rejects_non_power_of_2_splits():
    query, cache, si, bt, seqkv, _ = _build_inputs(sparse=2048)
    with pytest.raises(AssertionError, match="power of 2"):
        flag_gems.kv_quant_sparse_flash_attention(
            query, cache, si, bt, actual_seq_lengths_kv=seqkv,
            splits=3,
        )

def test_rejects_t_ne_b():
    """T != B should be rejected."""
    query, cache, si, bt, seqkv, _ = _build_inputs()
    # block_table is [1, max_blocks], query is [1, 64, 576]
    # Create a 2-batch query but keep 1-batch block_table
    query2 = query.expand(2, -1, -1).contiguous()
    with pytest.raises(AssertionError, match="T==B"):
        flag_gems.kv_quant_sparse_flash_attention(
            query2, cache, si, bt, actual_seq_lengths_kv=seqkv,
        )


# ─── cache view correctness ───────────────────────────────────────────

def test_cache_views():
    """_cache_views returns views with correct dtypes and shapes."""
    from flag_gems.runtime.backend._ascend.ops.kv_quant_sparse_attn import _cache_views
    block_num, block_size = 10, 128
    cache_u8 = torch.zeros(block_num, block_size, 1, SLOT, dtype=torch.uint8, device=flag_gems.device)
    cache = cache_u8.view(torch.float8_e4m3fn)
    nope, rope, scl = _cache_views(cache)
    nrow = block_num * block_size
    assert nope.shape == (nrow, NOPE)
    assert nope.dtype == torch.float8_e4m3fn
    assert rope.shape == (nrow, ROPE)
    assert rope.dtype == torch.bfloat16
    assert scl.shape == (nrow, NTILE)
    assert scl.dtype == torch.float32
