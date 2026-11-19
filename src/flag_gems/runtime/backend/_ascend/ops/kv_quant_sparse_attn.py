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

"""Ascend KvQuantSparseFlashAttention (MLA-absorb + fp8 per-tile quantized KV).

Replaces ``torch_npu.npu_kv_quant_sparse_flash_attention`` in the decode
call form used by vllm-ascend ``_execute_kv_quant_sparse_flash_attention_vllm``
(``device_op.py:634``)::

    torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query, key=kv, value=kv, sparse_indices=topk_indices,
        scale_value=sfa_impl.scale, sparse_block_size=1, block_table=block_table,
        actual_seq_lengths_query=..., actual_seq_lengths_kv=...,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2, quant_scale_repo_mode=1,
        tile_size=128, key_quant_mode=2, value_quant_mode=2, rope_head_dim=64,
    )

Numerical definition (matches ``test_a5_c8_lse`` reference)::

    k_deq  = (k_nope_fp8.float() * scale.repeat_interleave(128)).to(bf16)
    k      = cat(k_deq, k_rope)
    scores = q.float() @ k.T * scale_value
    out    = softmax(scores) @ k_deq.float()

Key implementation notes:

1. V is the dequantized k_nope (MLA-absorb: KV share the same data).
2. Dequantization multiply in fp32, then round to bf16.
3. KV cache slot is 656 bytes: 512 fp8 nope + 128B bf16 rope + 16B fp32 scale.
4. Flash-decoding splits the sparse dimension for parallelism on 32 AI cores.
5. Two-pass combine kernel merges partial results via online softmax.
6. Division and modulo on runtime values replaced by bit-shift/mask (17-37x faster).

Coverage: decode (T==B), attention_mode=2, sparse_block_size=1,
key/value_quant_mode=2, quant_scale_repo_mode=1, tile_size=128,
rope_head_dim=64, sparse_mode=3, layout TND/PA_BSND, fp8_e4m3fn KV.
"""

from __future__ import annotations

import logging

import torch
import torch_npu  # noqa: F401  register NPU backend
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

__all__ = ["kv_quant_sparse_flash_attention"]

NOPE = 512
ROPE = 64
TILE = 128                       # tile_size attribute, only 128 supported
NTILE = NOPE // TILE             # 4 per-tile scales
SLOT = NOPE + ROPE * 2 + NTILE * 4   # 656 bytes


@triton.jit
def _qsfa_kernel(
    q_ptr,                       # [T, Q_N, 576] bf16
    k_nope_ptr,                  # fp8  view [NROW, 512]
    k_rope_ptr,                  # bf16 view [NROW, 64]
    k_scl_ptr,                   # fp32 view [NROW, 4]
    sparse_ptr,                  # [T, 1, S] int32
    bt_ptr,                      # [B, MAXBLK] int32
    seqkv_ptr,                   # [B] int32  actual_seq_lengths_kv
    out_ptr,                     # [T, Q_N, 512] bf16   (SPLITS==1)
    po_ptr, pm_ptr, pl_ptr,      # partials [SPLITS,T,Q_N,512] / [SPLITS,T,Q_N]
    scale_value,
    QN: tl.constexpr, S: tl.constexpr, MAXBLK: tl.constexpr,
    SLOT_N: tl.constexpr, SLOT_R: tl.constexpr, SLOT_S: tl.constexpr,
    LOG2_BS: tl.constexpr, BS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr,
    SPLITS: tl.constexpr, LOG2_SPLITS: tl.constexpr,
    NOPE_C: tl.constexpr, ROPE_C: tl.constexpr,
    TILE_C: tl.constexpr, NTILE_C: tl.constexpr,
    DEQ_F32: tl.constexpr, DEQ3D: tl.constexpr, PART_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.program_id(1)                 # batch / query token
    hb = pid >> LOG2_SPLITS
    sp = pid & (SPLITS - 1)

    CHUNK: tl.constexpr = S // SPLITS

    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, NOPE_C)
    offs_r = tl.arange(0, ROPE_C)
    offs_t = tl.arange(0, NTILE_C)

    qrow = q_ptr + t * QN * (NOPE_C + ROPE_C) + offs_h[:, None] * (NOPE_C + ROPE_C)
    q_nope = tl.load(qrow + offs_d[None, :])
    q_rope = tl.load(qrow + NOPE_C + offs_r[None, :])

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, NOPE_C], dtype=tl.float32)

    valid_cnt = tl.minimum(tl.load(seqkv_ptr + t), S)

    for s0 in range(0, CHUNK, BLOCK_S):
        offs_s = sp * CHUNK + s0 + tl.arange(0, BLOCK_S)
        idx = tl.load(sparse_ptr + t * S + offs_s)

        valid = (idx >= 0) & (offs_s < valid_cnt)
        idxc = tl.maximum(idx, 0)

        logical = idxc >> LOG2_BS
        off = idxc & (BS - 1)
        phys = tl.load(bt_ptr + t * MAXBLK + logical)
        row = phys * BS + off

        k_raw = tl.load(k_nope_ptr + row[:, None] * SLOT_N + offs_d[None, :])
        k_rope = tl.load(k_rope_ptr + row[:, None] * SLOT_R + offs_r[None, :])
        sc4 = tl.load(k_scl_ptr + row[:, None] * SLOT_S + offs_t[None, :])

        if DEQ_F32:
            sc_e = tl.reshape(tl.broadcast_to(sc4[:, :, None], (BLOCK_S, NTILE_C, TILE_C)),
                              (BLOCK_S, NOPE_C))
            k_deq = (k_raw.to(tl.float32) * sc_e).to(tl.bfloat16)
        elif DEQ3D:
            k3 = tl.reshape(k_raw.to(tl.bfloat16), (BLOCK_S, NTILE_C, TILE_C))
            k_deq = tl.reshape(k3 * sc4.to(tl.bfloat16)[:, :, None], (BLOCK_S, NOPE_C))
        else:
            sc_b = tl.reshape(
                tl.broadcast_to(sc4.to(tl.bfloat16)[:, :, None], (BLOCK_S, NTILE_C, TILE_C)),
                (BLOCK_S, NOPE_C))
            k_deq = k_raw.to(tl.bfloat16) * sc_b

        s = tl.dot(q_nope, tl.trans(k_deq), out_dtype=tl.float32)
        s += tl.dot(q_rope, tl.trans(k_rope), out_dtype=tl.float32)
        s = s * scale_value
        s = tl.where(valid[None, :], s, float("-inf"))

        # online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_deq, out_dtype=tl.float32)
        m_i = m_new

    if SPLITS == 1:
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / safe_l[:, None]
        tl.store(out_ptr + t * QN * NOPE_C + offs_h[:, None] * NOPE_C + offs_d[None, :],
                 o.to(tl.bfloat16))
    else:
        base = sp * (QN * NOPE_C) + t * (SPLITS * QN * NOPE_C)
        if PART_BF16:
            tl.store(po_ptr + base + offs_h[:, None] * NOPE_C + offs_d[None, :],
                     acc.to(tl.bfloat16))
        else:
            tl.store(po_ptr + base + offs_h[:, None] * NOPE_C + offs_d[None, :], acc)
        mbase = sp * QN + t * (SPLITS * QN)
        tl.store(pm_ptr + mbase + offs_h, m_i)
        tl.store(pl_ptr + mbase + offs_h, l_i)


@triton.jit
def _qsfa_combine_kernel(
    po_ptr, pm_ptr, pl_ptr,
    out_ptr,
    QN: tl.constexpr, NOPE_C: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_H: tl.constexpr,
    PART_BF16: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    """Merge SPLITS unnormalized partial sums via two-pass weighted reduction."""
    hb = tl.program_id(0)
    t = tl.program_id(1)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, NOPE_C)
    offs_a = tl.arange(0, SPLITS)

    # Pass 1: read all (m, l), compute global M and weights
    ml_base = t * (SPLITS * QN) + offs_a[:, None] * QN + offs_h[None, :]
    pm_all = tl.load(pm_ptr + ml_base)
    pl_all = tl.load(pl_ptr + ml_base)

    M = tl.max(pm_all, axis=0)
    M_safe = tl.where(M == float("-inf"), 0.0, M)
    w = tl.where(pm_all == float("-inf"), 0.0, tl.exp(pm_all - M_safe[None, :]))
    l_tot = tl.sum(pl_all * w, axis=0)

    # Pass 2: weighted reduction over partials in blocks of BLOCK_SP
    acc = tl.zeros([BLOCK_H, NOPE_C], dtype=tl.float32)
    for s0 in range(0, SPLITS, BLOCK_SP):
        offs_s = s0 + tl.arange(0, BLOCK_SP)
        po = tl.load(po_ptr + t * (SPLITS * QN * NOPE_C)
                     + offs_s[:, None, None] * (QN * NOPE_C)
                     + offs_h[None, :, None] * NOPE_C
                     + offs_d[None, None, :])
        wb = tl.load(pm_ptr + t * (SPLITS * QN)
                     + offs_s[:, None] * QN + offs_h[None, :])
        wb = tl.where(wb == float("-inf"), 0.0, tl.exp(wb - M_safe[None, :]))
        acc += tl.sum(po.to(tl.float32) * wb[:, :, None], axis=0)

    safe_l = tl.where(l_tot > 0.0, l_tot, 1.0)
    o = acc / safe_l[:, None]
    tl.store(out_ptr + t * QN * NOPE_C + offs_h[:, None] * NOPE_C + offs_d[None, :],
             o.to(tl.bfloat16))


def _cache_views(cache):
    """Build three zero-copy views on the packed KV cache: fp8 nope / bf16 rope / fp32 scale.

    Uses ``Tensor.set_()`` instead of ``.view(dtype)`` because the slot mixes
    three dtypes and each segment is not a suffix of the full tensor.
    """
    st = cache.untyped_storage()
    nrow = cache.numel() // SLOT
    dev = cache.device

    def mk(dtype, elem_off, width, slot_elems):
        t = torch.empty(0, dtype=dtype, device=dev)
        t.set_(st, storage_offset=elem_off, size=(nrow, width), stride=(slot_elems, 1))
        return t

    nope = mk(torch.float8_e4m3fn, 0, NOPE, SLOT)
    rope = mk(torch.bfloat16, NOPE // 2, ROPE, SLOT // 2)
    scl = mk(torch.float32, (NOPE + ROPE * 2) // 4, NTILE, SLOT // 4)
    return nope, rope, scl


def kv_quant_sparse_flash_attention(
    query, cache, sparse_indices, block_table,
    actual_seq_lengths_kv=None,
    scale_value=None,
    splits=64,
    block_h=64,
    block_s=32,
    deq_f32=False,
    deq3d=True,
    part_bf16=True,
    launch_num_stages=None,
    combine_h=1,
    block_sp=32,
    num_cores=None,
):
    """Triton KvQuantSparseFlashAttention (decode path).

    Parameters match ``torch_npu.npu_kv_quant_sparse_flash_attention`` where
    applicable.

    Args:
        query: ``[T, Q_N, 576]`` bf16 (TND, 576 = 512 nope + 64 rope).
        cache: ``[block_num, block_size, 1, 656]`` fp8_e4m3fn (PA_BSND).
        sparse_indices: ``[T, 1, sparse]`` int32, valid values left-aligned, -1 padding.
        block_table: ``[B, max_blocks]`` int32.
        actual_seq_lengths_kv: ``[B]`` int32, required.
        scale_value: attention scale, defaults to (512+64)^-0.5.
        splits: flash-decoding KV splits (power of 2).
        block_h: head block size.
        block_s: KV token block size.
        deq_f32: dequantize in fp32 (uses more UB).
        deq3d: 3D reshape dequantization (default, saves UB).
        part_bf16: store partials as bf16 (halves combine traffic).
        launch_num_stages: override backend auto-multi-buffer.
        combine_h: head block for combine kernel.
        block_sp: split block for combine kernel.
        num_cores: override core count.

    Returns:
        ``[T, Q_N, 512]`` bf16.
    """
    assert query.dtype == torch.bfloat16, f"query must be bf16, got {query.dtype}"
    assert cache.dtype == torch.float8_e4m3fn, f"cache must be fp8_e4m3fn, got {cache.dtype}"
    assert query.dim() == 3 and query.shape[-1] == NOPE + ROPE, \
        f"query must be [T, Q_N, {NOPE + ROPE}], got {tuple(query.shape)}"
    assert cache.dim() == 4 and cache.shape[-1] == SLOT and cache.shape[-2] == 1, \
        f"cache must be [block_num, block_size, 1, {SLOT}], got {tuple(cache.shape)}"

    T, QN, _ = query.shape
    block_size = cache.shape[1]
    S = sparse_indices.shape[-1]
    B, MAXBLK = block_table.shape

    assert block_size & (block_size - 1) == 0, \
        f"block_size={block_size} must be power of 2"
    assert splits & (splits - 1) == 0, f"splits={splits} must be power of 2"
    assert S % splits == 0 and (S // splits) % block_s == 0, \
        f"sparse={S} must be divisible by splits={splits}, and chunk {S // splits} by block_s={block_s}"
    assert QN % block_h == 0, f"Q_N={QN} must be divisible by block_h={block_h}"
    assert T == B, f"decode requires T==B, got T={T} B={B}"
    assert actual_seq_lengths_kv is not None, \
        "actual_seq_lengths_kv is required"
    seqkv = actual_seq_lengths_kv.reshape(-1).to(torch.int32).contiguous()
    assert seqkv.numel() == B, \
        f"actual_seq_lengths_kv must have B={B} elements, got {seqkv.numel()}"
    if scale_value is None:
        scale_value = (NOPE + ROPE) ** -0.5

    with torch_device_fn.device(query.device):
        q = query.contiguous()
        si = sparse_indices.contiguous()
        bt = block_table.contiguous()
        k_nope, k_rope, k_scl = _cache_views(cache)

        out = torch.empty(T, QN, NOPE, device=query.device, dtype=torch.bfloat16)
        nh = QN // block_h

        if splits == 1:
            po = pm = pl = out
        else:
            po = torch.empty(T, splits, QN, NOPE, device=query.device,
                             dtype=torch.bfloat16 if part_bf16 else torch.float32)
            pm = torch.empty(T, splits, QN, device=query.device, dtype=torch.float32)
            pl = torch.empty(T, splits, QN, device=query.device, dtype=torch.float32)

        kw = {}
        if num_cores is not None:
            kw["num_cores"] = num_cores
        if launch_num_stages is not None:
            kw["num_stages"] = launch_num_stages

        _qsfa_kernel[(nh * splits, T)](
            q, k_nope, k_rope, k_scl, si, bt, seqkv, out, po, pm, pl,
            scale_value,
            QN=QN, S=S, MAXBLK=MAXBLK,
            SLOT_N=SLOT, SLOT_R=SLOT // 2, SLOT_S=SLOT // 4,
            LOG2_BS=block_size.bit_length() - 1, BS=block_size,
            BLOCK_H=block_h, BLOCK_S=block_s,
            SPLITS=splits, LOG2_SPLITS=splits.bit_length() - 1,
            NOPE_C=NOPE, ROPE_C=ROPE, TILE_C=TILE, NTILE_C=NTILE,
            DEQ_F32=deq_f32, DEQ3D=deq3d, PART_BF16=part_bf16,
            **kw,
        )

        if splits > 1:
            ch = combine_h if combine_h is not None else block_h
            assert QN % ch == 0, f"Q_N={QN} must be divisible by combine_h={ch}"
            _qsfa_combine_kernel[(QN // ch, T)](
                po, pm, pl, out,
                QN=QN, NOPE_C=NOPE, SPLITS=splits, BLOCK_H=ch, PART_BF16=part_bf16,
                BLOCK_SP=min(block_sp, splits),
            )

    return out
