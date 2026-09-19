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
#
# Kunlunxin (XPU) backend specialization of the DSA lighting indexer kernel.
# Rewrite rationale (each item addresses a construct the SDNN pipeline cannot
# lower or executes incorrectly on XPU):
#   1) window bounds load as scalars (BQ == 1) - a tensor load followed by a
#      min/max reduce feeding a dot kernel trips TritonSDNNLegalize;
#   2) a single store per program: the store mask is `offs < hi` (slt with a
#      scalar rhs) and lanes below the window start are turned into -inf by
#      the additive division `x - below / (1 - below)`.  A second store for
#      the lower bound would write the same columns twice (the compiler emits
#      the two DMA writes without a WAW wait, observed as sporadic 64-column
#      corruption), and a compare->select cannot be used (its EW lowering
#      corrupts the first 32-lane chunk for a dynamic threshold);
#   3) no runtime inner loop - a dot inside a runtime loop produced
#      non-deterministic results on the released compiler;
#   4) bf16 operands feed tl.dot directly (bf16 -> f16 DMA conversion is not
#      supported by the DMAi lowering);
#   5) head-sum via expand_dims(acc.sum(0), 0) under the BQ == 1 contract -
#      the trans/reshape/3-D-reduce chain fails TritonSDNNLegalize.
# ik_fix8.py — loop-free restructure: grid = (Q rows, K/BK column blocks); one dot per program.
# Rewrites vs upstream indexer_k_tiled.py: no inner kv loop (race-free), scalar window loads,
# slt-only masks + value-level where for the lower bound, expand_dims(acc.sum(0)) for the head sum.
import torch
import triton
import triton.language as tl

@triton.jit
def fishdv(
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
    i_sh = tl.program_id(0)  # q row (BQ == 1)
    i_bk = tl.program_id(1)  # kv block of BK columns

    # BQ == 1 contract: per-row window bounds load as scalars (no DMA-load reduce:
    # tensor-load + min/max reduce feeding a dot kernel trips SDNN Legalize).
    bc = tl.load(cu_bg_seqlens + i_sh, mask=i_sh < CU, other=1000000000)  # ks
    ec = tl.load(cu_ed_seqlens + i_sh, mask=i_sh < CU, other=-1000000000)  # ke
    bos_v = bc
    eos_v = ec
    lo_g = tl.maximum(bos_v, 0)  # window start (global kv index)
    hi_g = tl.minimum(eos_v, K)  # window end (global kv index)

    offs_bk = i_bk * BK + tl.arange(0, BK)  # global kv indices of this block
    mask_k = offs_bk < K  # load guard (last block may exceed K)

    offs_bq = tl.arange(0, BQ * H) + i_sh * (BQ * H)
    offs_d = tl.arange(0, D)
    offs_boq = tl.arange(0, BQ) + i_sh * BQ
    mask_boq = offs_boq < Q

    q_ptr = q_index + offs_bq[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_blk = tl.load(q_ptr)  # [BQ*H, D] bf16 (bf16->f16 dmai unsupported)
    w_blk = tl.load(weights + offs_bq * stride_wh)  # [BQ*H] f32

    k_ptr = k_index + offs_d[:, None] * stride_kd + offs_bk[None, :] * stride_kn
    k_blk = tl.load(k_ptr, mask_k[None, :], other=0.0)  # bf16

    acc = tl.dot(q_blk, k_blk, out_dtype=tl.float16)  # [BQ*H, BK]
    acc = tl.maximum(acc, 0.0).to(tl.float32) * w_blk[:, None]
    # BQ == 1: acc rows are h in [0,H); axis-0 sum == head-sum.
    out_blk = tl.expand_dims(acc.sum(0), 0)  # [1, BK] == [BQ, BK]

    o_ptr = (
        logits
        + offs_boq[:, None] * stride_lm
        + offs_bk[None, :] * stride_ln
    )
    out_msk = mask_boq[:, None] & (offs_bk[None, :] < hi_g)
    # store only in [lo_g, hi_g); lanes below keep -inf (buffer pre-filled)
    below = (offs_bk < lo_g).to(tl.float32)[None, :]  # 1.0 below window start
    out_val = out_blk - below / (1.0 - below)  # below: x - inf = -inf; above: x - 0 = x
    tl.store(o_ptr, out_val, out_msk)  # f32 store


def triton_lighting_indexer_k_tiled_interface(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke
):
    Q, H, D = q.shape[0], q.shape[1], q.shape[2]
    K = kv.shape[0]
    CU = cu_seqlen_ks.shape[0]
    logits = torch.full([Q, K], float("-inf"), device="cuda", dtype=torch.float32)
    BQ = 1
    BK = 512
    TK = 2048
    NQ = triton.cdiv(Q, BQ)
    NK = triton.cdiv(K, BK)
    grid = (NQ, NK)
    fishdv[grid](
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
        num_warps=8,
        num_stages=2,
    )
    return logits
