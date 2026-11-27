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

import torch
import triton
import triton.language as tl


@triton.jit
def _engram_hash_kernel(
    tok_ptr,
    mult_ptr,
    vocab_ptr,
    off_ptr,
    mask_ptr,
    out_ptr,
    num_tokens,
    M: tl.constexpr,
    TAB: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    # Token rows are the innermost (vectorized) axis, so a 64-lane int32 vector
    # group (256 B) is fully populated and the per-column quantities (v, o,
    # 2**15 % v, 2**30 % v) are scalars of the major axis.
    #
    # Each output column group is emitted as its own TAB-wide tile.  C is
    # (M-1)*TAB and TAB is a power of two, so every tile is exactly full: no
    # masked padding columns, no per-column limb select, and the three folded
    # limbs stay on the narrow [BT] row vector instead of the wide tile.
    pid_t = tl.program_id(0)
    pid_l = tl.program_id(1)
    base = pid_t * BT

    rows = base + tl.arange(0, BT)
    rmask = rows < num_tokens
    tloc = tl.arange(0, TAB)

    if HAS_MASK:
        mv = tl.load(mask_ptr + rows, mask=rmask, other=0) != 0
    else:
        mv = rmask

    # Output column group g (0..M-2) uses the hash accumulated over n-gram
    # positions 0..g+1: progressive XOR, exactly as the reference kernel.
    h = tl.zeros([BT], dtype=tl.int64)
    for i in tl.static_range(M):
        t_i = tl.load(tok_ptr + rows * M + i, mask=rmask, other=0).to(tl.int64)
        m_i = tl.load(mult_ptr + pid_l * M + i)
        h = h ^ (t_i * m_i)
        if i > 0:
            cols = (i - 1) * TAB + tloc
            vg = tl.load(vocab_ptr + pid_l * C + cols)
            o = tl.load(off_ptr + pid_l * C + cols)

            # h = d0 + d1*2**15 + d2*2**30, so
            # h % v == (d0 + d1*(2**15 % v) + d2*(2**30 % v)) % v.
            # d1*(2**15 % v) <= 32767*32768 and d2*(2**30 % v) <= 15*(v-1), so
            # the fold is bounded by ~1.09e9 for v <= 1e6 and never overflows
            # int32; all operands are non-negative, so the single 32-bit
            # remainder is exact.
            lo = h.to(tl.int32)
            hi = (h >> 32).to(tl.int32)
            d0 = lo & 32767
            d1 = (lo >> 15) & 32767
            d2 = ((lo >> 30) & 3) | (hi << 2)
            r1 = 32768 % vg
            r2 = 1073741824 % vg
            s = d0[None, :] + d1[None, :] * r1[:, None] + d2[None, :] * r2[:, None]
            # Exact s % v without the opaque out-of-line int32 vmod call.
            # s <= 1.09e9 and 1e5 <= v < 1e6, so s/v <= 1.1e4 and the relative
            # error of float32(s) * (1/v) is under ~3e-7: at most 0.003
            # absolute on the quotient, i.e. the truncated quotient is within 1
            # of floor(s/v).  One fixup in each direction therefore makes
            # s - q*v exact on [0, v).  q*v <= s + v never overflows int32.
            rcp = 1.0 / vg.to(tl.float32)
            q = (s.to(tl.float32) * rcp[:, None]).to(tl.int32)
            r = s - q * vg[:, None]
            r = tl.where(r < 0, r + vg[:, None], r)
            r = tl.where(r >= vg[:, None], r - vg[:, None], r)
            r = r + o[:, None]

            if HAS_MASK:
                r = tl.where(mv[None, :], -1, r)

            # Row-major (layer, token, col) output: the returned tensor must
            # keep the reference strides, so this store is transposing.
            out_off = pid_l * num_tokens * C + rows[None, :] * C + cols[:, None]
            tl.store(out_ptr + out_off, r, mask=rmask[None, :])


def _pick_bt(num_tokens, n_layers):
    # 32 physical vector cores, 64 blocks total -> two waves with slack.
    per_layer = max(1, 32 // max(n_layers, 1))
    bt = triton.next_power_of_2(triton.cdiv(num_tokens, per_layer))
    if bt < 32:
        bt = 32
    if bt > 256:
        bt = 256
    return bt


def engram_hash(
    ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask=None
):
    """Compute per-layer n-gram embedding table indices."""
    layers, order = multipliers.shape
    if (
        ngram_token_ids.ndim != 2
        or ngram_token_ids.shape[1] != order
        or vocab_sizes.shape[:2] != (layers, order - 1)
        or offsets.shape != (layers, (order - 1) * vocab_sizes.shape[2])
    ):
        raise ValueError("incompatible n-gram or embedding table dimensions")
    if not all(
        t.is_contiguous() for t in (ngram_token_ids, multipliers, vocab_sizes, offsets)
    ):
        raise ValueError("expected contiguous n-gram inputs")
    num_tokens = ngram_token_ids.shape[0]
    n_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]
    tab = vocab_sizes.shape[2]
    C = (max_ngram_size - 1) * tab

    out = torch.empty(
        (n_layers, num_tokens, C),
        dtype=torch.int32,
        device=ngram_token_ids.device,
    )
    if num_tokens == 0 or n_layers == 0 or C == 0:
        return out

    bt = _pick_bt(num_tokens, n_layers)
    has_mask = image_token_mask is not None
    mask_arg = image_token_mask.view(torch.int8) if has_mask else ngram_token_ids
    grid = (triton.cdiv(num_tokens, bt), n_layers)
    _engram_hash_kernel[grid](
        ngram_token_ids,
        multipliers,
        vocab_sizes,
        offsets,
        mask_arg,
        out,
        num_tokens,
        M=max_ngram_size,
        TAB=tab,
        C=C,
        BT=bt,
        HAS_MASK=has_mask,
        num_warps=4,
        num_stages=1,
    )
    return out
