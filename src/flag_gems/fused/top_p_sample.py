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

"""Fused top-p + temperature + Gumbel-Max sampler tail.

The eager vLLM sampler tail on MetaX costs ~2.0 ms per decode step at
[64, 130560]:
    cast bf16->fp32        (~38 us)
    div_ by temperature    (~139 us)
    top-p filter           (~1613 us; one program per row, 23 row passes:
                            stats + 20-iteration bisection + mask)
    gumbel argmax          (~218 us)

This module replaces the whole chain with a split-parallel pipeline that
reads the bf16 lm_head output directly and never materializes fp32 logits:

    1. row stats:   split-parallel online max + partition sum of x/T
                    (partial kernel + tiny reduce kernel)
    2. top-p zoom:  mass-histogram threshold search over the logit range.
                    Default: a single 1024-bin round (tau precision
                    ~0.06 logits; statistically identical to the 2x256
                    two-round variant at 2.6x lower kernel time).
                    tau is taken as the crossing bin's LOWER edge, so the
                    kept set always carries >= p * Z mass (never over-trims).
    3. fused sample: split-parallel pass computing
                        score = (x/T >= tau) ? x + T*g : -inf
                    i.e. mask + temperature-folded Gumbel-Max in one read;
                    rows with T < 1e-5 get T := 0 and degenerate to a plain
                    argmax (vLLM's greedy-row semantics).

Numerics: bf16 -> fp32 conversion is exact, and the division x/T happens in
fp32 exactly like the eager path, so the only behavioral delta vs the eager
tail is the top-p threshold granularity above.

Randomness uses Triton Philox with scalar seed/step: for the non-captured
sampler tail only -- do NOT capture in a CUDA graph.
"""

import torch
import triton
import triton.language as tl

_SAMPLING_EPS = 1e-5
_NBINS = 256
_RANGE = 60.0  # exp(-60) is far below fp32 softmax resolution
_MAX_SPLITS = 32


# ---------------------------------------------------------------------------
# 1. row stats: per-row max m and partition sum Z of x/T
# ---------------------------------------------------------------------------
@triton.jit
def _row_stats_partial(
    LOGITS,
    TEMP,
    M_PART,
    Z_PART,
    stride_row,
    vocab_size,
    tiles_per_split,
    num_splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK)
    t = tl.load(TEMP + row).to(tl.float32)
    t = tl.where(t < 1e-5, 1.0, t)  # greedy rows: /1.0, same as stock

    m = float("-inf")
    Z = 0.0
    start = split * tiles_per_split * BLOCK
    for tile in range(tiles_per_split):
        idx = start + tile * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32) / t
        tile_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, tile_max)
        # an empty split (start >= vocab) yields new_m = -inf; -inf - -inf is NaN,
        # so rescale only when the running max is finite and skip non-finite lanes.
        keep = x > float("-inf")
        add = tl.sum(tl.where(keep, tl.exp(x - new_m), 0.0), axis=0)
        Z = tl.where(m > float("-inf"), Z * tl.exp(m - new_m), 0.0) + add
        m = new_m
    tl.store(M_PART + row * num_splits + split, m)
    tl.store(Z_PART + row * num_splits + split, Z)


@triton.jit
def _row_stats_reduce(
    M_PART,
    Z_PART,
    M_OUT,
    Z_OUT,
    LO,
    HI,
    num_splits,
    RANGE: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, MAX_SPLITS)
    mask = offs < num_splits
    ms = tl.load(M_PART + row * num_splits + offs, mask=mask, other=float("-inf"))
    zs = tl.load(Z_PART + row * num_splits + offs, mask=mask, other=0.0)
    m = tl.max(ms, axis=0)
    diff = tl.where(ms > float("-inf"), ms - m, 0.0)
    Z = tl.sum(zs * tl.exp(diff), axis=0)
    tl.store(M_OUT + row, m)
    tl.store(Z_OUT + row, Z)
    # initial zoom range: [m - RANGE, m]; exp(-RANGE) ~ 0 in fp32 softmax
    tl.store(LO + row, m - RANGE)
    tl.store(HI + row, m)


# ---------------------------------------------------------------------------
# 2. top-p threshold via histogram zoom
# ---------------------------------------------------------------------------
@triton.jit
def _hist_zoom(
    LOGITS,
    TEMP,
    M_ROW,
    LO,
    HI,
    HIST,
    stride_row,
    vocab_size,
    tiles_per_split,
    num_splits,
    NBINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK)
    t = tl.load(TEMP + row).to(tl.float32)
    t = tl.where(t < 1e-5, 1.0, t)
    m = tl.load(M_ROW + row)
    lo = tl.load(LO + row)
    hi = tl.load(HI + row)
    scale = NBINS / (hi - lo)

    start = split * tiles_per_split * BLOCK
    for tile in range(tiles_per_split):
        idx = start + tile * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32) / t
        b = ((x - lo) * scale).to(tl.int32)
        b = tl.minimum(tl.maximum(b, 0), NBINS - 1)
        w = tl.exp(x - m)
        # elements below the range carry ~zero mass; skip them
        in_range = (x >= lo) & mask
        tl.atomic_add(HIST + row * NBINS + b, tl.where(in_range, w, 0.0))


@triton.jit
def _hist_walk(
    HIST,
    Z_ROW,
    P,
    LO,
    HI,
    TAU,
    NBINS: tl.constexpr,
    FINAL: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NBINS)
    # load bins in reversed (top-first) order; cumsum then gives, at reversed
    # position r, the mass of bins >= NBINS-1-r
    mass_rev = tl.load(HIST + row * NBINS + (NBINS - 1 - offs))
    cum = tl.cumsum(mass_rev, axis=0)
    z = tl.load(Z_ROW + row)
    p = tl.load(P + row).to(tl.float32)
    lo = tl.load(LO + row)
    hi = tl.load(HI + row)
    w = (hi - lo) / NBINS

    target = p * z
    reach = cum >= target
    # crossing bin b*: cum-from-top first reaches target there;
    # sum(reach) == b* + 1 (reversed positions r*..NBINS-1 all reach)
    b = tl.sum(reach.to(tl.int32), axis=0) - 1
    b = tl.minimum(tl.maximum(b, 0), NBINS - 1)
    # conservative: tau = lower edge of crossing bin => mass{x>=tau} >= target
    new_lo = lo + b * w
    new_hi = new_lo + w
    if FINAL:
        tl.store(TAU + row, new_lo)
    else:
        tl.store(LO + row, new_lo)
        tl.store(HI + row, new_hi)


# ---------------------------------------------------------------------------
# 3. mask + temperature-folded Gumbel-Max in one pass
# ---------------------------------------------------------------------------
@triton.jit
def _masked_sample_partial(
    LOGITS,
    TEMP,
    TAU,
    PVAL,
    PIDX,
    stride_row,
    vocab_size,
    vocab_padded,
    seed,
    step,
    tiles_per_split,
    num_splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK)
    t = tl.load(TEMP + row).to(tl.float32)
    t_eff = tl.where(t < 1e-5, 1.0, t)
    noise = tl.where(t < 1e-5, 0.0, t)  # greedy rows: pure argmax
    tau = tl.load(TAU + row)
    row_seed = seed + row

    start = split * tiles_per_split * BLOCK
    best = float("-inf")
    best_idx = 0
    for tile in range(tiles_per_split):
        idx = start + tile * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        xs = x / t_eff
        keep = xs >= tau
        u = tl.rand(row_seed, step * vocab_padded + idx.to(tl.int32))
        u = tl.minimum(tl.maximum(u, 1e-20), 1.0 - 1e-7)
        g = -tl.log(-tl.log(u))
        score = tl.where(keep, x + noise * g, float("-inf"))
        tile_best = tl.max(score, axis=0)
        tile_arg = tl.argmax(score, axis=0)
        take = tile_best > best
        best_idx = tl.where(take, start + tile * BLOCK + tile_arg, best_idx)
        best = tl.maximum(best, tile_best)
    tl.store(PVAL + row * num_splits + split, best)
    tl.store(PIDX + row * num_splits + split, best_idx.to(tl.int32))


@triton.jit
def _sample_reduce(
    PVAL,
    PIDX,
    OUT,
    num_splits,
    MAX_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, MAX_SPLITS)
    mask = offs < num_splits
    vals = tl.load(PVAL + row * num_splits + offs, mask=mask, other=float("-inf"))
    idxs = tl.load(PIDX + row * num_splits + offs, mask=mask, other=0)
    first = tl.argmax(vals, axis=0)  # left-most split wins ties
    tok = tl.sum(tl.where(offs == first, idxs, 0))
    tl.store(OUT + row, tok)


_STEP_COUNTER = {"value": 0}
_BUF_CACHE = {}


def _get_buffers(batch, vocab, dev, nbins=_NBINS):
    key = (batch, vocab, dev, nbins)
    bufs = _BUF_CACHE.get(key)
    if bufs is None:
        BLOCK = 4096
        tiles_total = triton.cdiv(vocab, BLOCK)
        num_splits = min(_MAX_SPLITS, max(1, 2048 // max(batch, 1)), tiles_total)
        tiles_per_split = triton.cdiv(tiles_total, num_splits)
        f32 = dict(device=dev, dtype=torch.float32)
        bufs = dict(
            num_splits=num_splits,
            tiles_per_split=tiles_per_split,
            vocab_padded=num_splits * tiles_per_split * BLOCK,
            m_part=torch.empty(batch, num_splits, **f32),
            z_part=torch.empty(batch, num_splits, **f32),
            m_row=torch.empty(batch, **f32),
            z_row=torch.empty(batch, **f32),
            lo=torch.empty(batch, **f32),
            hi=torch.empty(batch, **f32),
            tau=torch.empty(batch, **f32),
            hist=torch.empty(batch, nbins, **f32),
            pval=torch.empty(batch, num_splits, **f32),
            pidx=torch.empty(batch, num_splits, device=dev, dtype=torch.int32),
            out=torch.empty(batch, device=dev, dtype=torch.int32),
        )
        _BUF_CACHE[key] = bufs
    return bufs


def fused_top_p_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    seed: int = 0,
    zoom_rounds: int = 1,
    nbins: int = 0,
) -> torch.Tensor:
    """Sample one token per row: top-p filter + temperature + Gumbel-Max.

    Args:
        logits: [batch, vocab] bf16/fp16/fp32 raw lm_head output.
        temperature: [batch] float tensor; rows < 1e-5 are greedy (argmax).
        top_p: [batch] float tensor of nucleus thresholds.
        seed: Philox base seed.
        zoom_rounds: histogram zoom rounds.  1 (default) with 1024 bins is
            fastest and statistically identical to 2x256 on TV/boundary
            tests (test_zoom_variants.py); tau is always the crossing bin's
            LOWER edge so the kept set never under-covers p*Z.
        nbins: histogram bins per round; 0 => 1024 (1 round) or 256
            (2+ rounds).

    Returns:
        [batch] int32 tensor of token ids.
    """
    if nbins == 0:
        nbins = 1024 if zoom_rounds == 1 else _NBINS
    batch, vocab = logits.shape
    dev = logits.device
    b = _get_buffers(batch, vocab, dev, nbins)
    ns = b["num_splits"]
    grid2d = (batch, ns)

    _row_stats_partial[grid2d](
        logits, temperature, b["m_part"], b["z_part"],
        logits.stride(0), vocab, b["tiles_per_split"], ns,
        BLOCK=4096, num_warps=2,
    )
    _row_stats_reduce[(batch,)](
        b["m_part"], b["z_part"], b["m_row"], b["z_row"], b["lo"], b["hi"],
        ns, RANGE=_RANGE, MAX_SPLITS=_MAX_SPLITS, num_warps=1,
    )

    for round_no in range(zoom_rounds, 0, -1):
        b["hist"].zero_()
        _hist_zoom[grid2d](
            logits, temperature, b["m_row"], b["lo"], b["hi"], b["hist"],
            logits.stride(0), vocab, b["tiles_per_split"], ns,
            NBINS=nbins, BLOCK=4096, num_warps=4,
        )
        _hist_walk[(batch,)](
            b["hist"], b["z_row"], top_p, b["lo"], b["hi"], b["tau"],
            NBINS=nbins, FINAL=(round_no == 1), num_warps=1,
        )

    step = _STEP_COUNTER["value"]
    _STEP_COUNTER["value"] += 1
    _masked_sample_partial[grid2d](
        logits, temperature, b["tau"], b["pval"], b["pidx"],
        logits.stride(0), vocab, b["vocab_padded"], seed, step,
        b["tiles_per_split"], ns,
        BLOCK=4096, num_warps=4,
    )
    _sample_reduce[(batch,)](
        b["pval"], b["pidx"], b["out"], ns, MAX_SPLITS=_MAX_SPLITS, num_warps=1
    )
    return b["out"]
