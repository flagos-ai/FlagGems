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
"""GCU400 specialization of the DeepSeek sparse-MLA lightning-indexer logits.

Same math as ``flag_gems.fused.DSA.indexer_k_tiled``:
``logits[q, n] = sum_h relu(q[q, h, :] . kv[n, :]) * w[q, h]`` for ``n`` inside
the per-query ``[ks[q], ke[q])`` window and ``-inf`` outside.

What changed for GCU400:

* multiple queries per program (``BQ``) so the MMA M dimension is
  ``BQ * H`` (128/256) instead of ``H``; the key tile ``BK`` is then sized so
  ``BQ * H * BK <= 32768`` elements, which is the largest tile that still fits
  the per-workgroup DSM budget of GCU400;
* the ``sum_h`` reduction uses ``acc.reshape(BQ, H, BK).sum(1)`` instead of the
  double transpose ``acc.trans().reshape(BK, BQ, H).sum(-1).trans()``;
* the key axis is processed in a single program dimension (``TK = K``) whenever
  there are enough query programs to fill the device, which removes the
  duplicated per-chunk prologue;
* ``warp_specialize`` (an NVIDIA-only hint) is dropped and the tile/launch
  parameters are chosen explicitly instead of by runtime autotuning.
"""

import torch
import triton
import triton.language as tl

# Largest (BQ * H) * BK product that fits the GCU400 per-workgroup DSM budget
# (measured: 32768 elements is accepted, 65536 raises OutOfResources).
_TILE_ELEM_LIMIT = 32768
# Below this many query programs the key axis is split as well, otherwise the
# device would not be filled with enough programs.
_MIN_QUERY_PROGRAMS = 96
_KEY_CHUNK_FALLBACK = 2048


@triton.jit
def _indexer_k_tiled_kernel(
    q_index,
    k_index,
    cu_bg_seqlens,
    cu_ed_seqlens,
    weights,
    logits,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_wh,
    stride_lm,
    stride_ln,
    Q: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    CU: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    i_sh, i_k = tl.program_id(0), tl.program_id(1)

    # Per-query [ks, ke) window, clipped to the key chunk handled by this program.
    offs_cu = tl.arange(0, BQ) + i_sh * BQ
    mask_cu = offs_cu < CU
    bos_vec, eos_vec = tl.load(
        cu_bg_seqlens + offs_cu, mask_cu, 1000000000
    ) + i_k * TK, tl.load(cu_ed_seqlens + offs_cu, mask_cu, -1000000000)
    eos_vec = tl.minimum(eos_vec, bos_vec + (i_k + 1) * TK)
    # Union of the BQ windows: the dot product is evaluated on it, the per-query
    # masks below drop everything outside each single window again.
    bos, eos = max(bos_vec.min(0), 0), min(eos_vec.max(0), K)
    CK = eos - bos
    if CK > 0:
        q_base = q_index
        k_base = k_index + bos * stride_kn
        w_base = weights
        o_base = logits + bos * stride_ln
        offs_bq = tl.arange(0, BQ * H) + i_sh * (BQ * H)
        offs_d = tl.arange(0, D)
        mask_bq = offs_bq < Q * H
        mask_d = offs_d < D

        q_ptr = q_base + offs_bq[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q_blk = tl.load(q_ptr, mask_bq[:, None] & mask_d[None, :], 0.0).to(tl.float16)

        w_ptr = w_base + offs_bq * stride_wh
        w_blk = tl.load(w_ptr, mask_bq, 0.0).to(tl.float16)

        CK = tl.cdiv(CK, BK)
        for ck in range(CK):
            offs_bk = ck * BK + tl.arange(0, BK)
            mask_bk = bos + offs_bk < eos
            k_ptr = k_base + offs_d[:, None] * stride_kd + offs_bk[None, :] * stride_kn
            k_blk = tl.load(k_ptr, mask_d[:, None] & mask_bk[None, :], 0.0).to(
                tl.float16
            )
            acc = tl.dot(q_blk, k_blk, out_dtype=tl.float16)
            # relu + per-head weight, then reduce the H heads that share a query.
            acc = tl.maximum(acc, 0.0) * w_blk[:, None]
            out_blk = acc.reshape(BQ, H, BK).sum(1)
            pos = offs_bk[None, :] + bos
            out_ptr = (
                o_base + offs_cu[:, None] * stride_lm + offs_bk[None, :] * stride_ln
            )
            out_msk = (
                (offs_cu[:, None] < Q)
                & mask_bk[None, :]
                & (bos_vec[:, None] <= pos)
                & (eos_vec[:, None] > pos)
            )
            tl.store(out_ptr, out_blk.to(tl.float16), out_msk)


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _select_tiles(Q, H, D, K):
    """Choose (BQ, BK, TK) for the given problem size.

    ``BQ * H`` is kept at 256 lanes for the MMA M dimension (128 for tiny head
    counts), ``BK`` fills the remaining DSM budget, and the key axis stays in a
    single program as long as the grid is wide enough.
    """
    # tl.arange() needs power-of-two lengths, which the original kernel also
    # relies on for ``arange(0, H)``.
    if not _is_pow2(H):
        return 1, 64, min(K, _KEY_CHUNK_FALLBACK)

    target_m = 256 if H >= 32 else 128
    bq = max(1, target_m // H)
    while bq > 1 and (not _is_pow2(bq * H) or bq > Q):
        bq //= 2
    elem_limit = _TILE_ELEM_LIMIT if D <= 128 else _TILE_ELEM_LIMIT // 2
    bk = min(256, elem_limit // (bq * H))
    bk = max(bk, 64)

    num_query_programs = triton.cdiv(Q, bq)
    if num_query_programs >= _MIN_QUERY_PROGRAMS or K <= _KEY_CHUNK_FALLBACK:
        tk = K
    else:
        tk = _KEY_CHUNK_FALLBACK
    return bq, bk, tk


def triton_lighting_indexer_k_tiled_interface(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke
):
    Q, H, D = q.shape[0], q.shape[1], q.shape[2]
    K = kv.shape[0]
    CU = cu_seqlen_ks.shape[0]
    logits = torch.full([Q, K], float("-inf"), device=q.device, dtype=torch.float32)

    BQ, BK, TK = _select_tiles(Q, H, D, K)
    grid = (triton.cdiv(Q, BQ), triton.cdiv(K, TK))
    _indexer_k_tiled_kernel[grid](
        q,
        kv,
        cu_seqlen_ks,
        cu_seqlen_ke,
        weights,
        logits,
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        Q,
        H,
        K,
        TK,
        D,
        CU,
        BQ,
        BK,
        num_warps=4,
        num_stages=1,
    )
    return logits
