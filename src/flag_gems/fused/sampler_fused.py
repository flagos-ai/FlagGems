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

"""Fused sampler tail: bf16 logits -> (temperature fold) -> token id.

vLLM's eager sampler tail runs three full-vocab passes per decode step:
  1. ``logits.to(torch.float32)``   (bf16 -> fp32 cast, 2x bytes out)
  2. ``logits.div_(temperature)``   (fp32 read + write)
  3. Gumbel-Max / argmax            (fp32 read)

This kernel fuses all three into a single read of the *bf16* lm_head output:

- SAMPLING=True: Gumbel-Max with temperature folded in.  For a row with
  temperature T > 0,

      argmax_i(x_i / T + g_i)  ==  argmax_i(x_i + T * g_i),  g_i ~ Gumbel(0,1)

  because dividing every score by the same positive T never changes the
  argmax.  Rows with T < 1e-5 (greedy requests in a mixed batch) get T := 0,
  which cancels the noise term and degenerates to a plain argmax -- exactly
  the semantics of vLLM's ``torch.where(temp < eps, greedy, random)`` tail.

- SAMPLING=False: pure argmax (no RNG at all), deterministic, left-most index
  wins on ties, matching ``torch.argmax``.

Parallelism: a one-program-per-row grid (batch = 64) starves the C500, so the
vocab dimension is split across programs.  Pass 1 writes per-split
(score, index) partials; pass 2 (one program per row, SPLITS lanes) reduces
them with the same left-most tie-break.

Traffic per step for [64, 130560] bf16: 16.5 MB read once, vs ~264 MB across
the eager cast + div + sample passes.

The Philox seed/step are plain scalars: this kernel is for the non-captured
(outside cudagraph) sampler tail.  Do NOT capture it in a CUDA graph.

Returned token ids are int32 to match ``SamplerOutput.sampled_token_ids``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sample_partial(
    LOGITS,
    TEMP,
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
    SAMPLING: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK)

    if SAMPLING:
        t = tl.load(TEMP + row).to(tl.float32)
        # Greedy rows (temperature < eps): zero temperature cancels the noise,
        # score == logits, i.e. a plain argmax for that row.
        t = tl.where(t < 1e-5, 0.0, t)
        row_seed = seed + row

    start = split * tiles_per_split * BLOCK
    best = float("-inf")
    best_idx = 0
    for tile in range(tiles_per_split):
        idx = start + tile * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if SAMPLING:
            u = tl.rand(row_seed, step * vocab_padded + idx.to(tl.int32))
            # clamp away from 0/1 so log(-log(u)) stays finite
            u = tl.minimum(tl.maximum(u, 1e-20), 1.0 - 1e-7)
            g = -tl.log(-tl.log(u))
            score = x + t * g
        else:
            score = x
        tile_best = tl.max(score, axis=0)
        tile_arg = tl.argmax(score, axis=0)
        take = tile_best > best
        best_idx = tl.where(take, start + tile * BLOCK + tile_arg, best_idx)
        best = tl.maximum(best, tile_best)

    tl.store(PVAL + row * num_splits + split, best)
    tl.store(PIDX + row * num_splits + split, best_idx.to(tl.int32))


@triton.jit
def _fused_sample_reduce(
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
    # left-most split wins ties, matching the within-split tie-break
    first = tl.argmax(vals, axis=0)
    tok = tl.sum(tl.where(offs == first, idxs, 0))
    tl.store(OUT + row, tok)


_STEP_COUNTER = {"value": 0}

_MAX_SPLITS = 32
_BUF_CACHE = {}


def fused_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Sample (or argmax) one token per row straight from low-precision logits.

    Args:
        logits: [batch, vocab] tensor, bf16 / fp16 / fp32.  NOT required to be
            temperature-scaled -- pass ``temperature`` instead.
        temperature: None for pure argmax, else a [batch] float tensor.
            Rows with temperature < 1e-5 are sampled greedily (argmax).
        seed: base seed for the Philox generator (sampling mode only).

    Returns:
        [batch] int32 tensor of token ids.  NOTE: the buffer is reused across
        calls with the same shape -- consume it before the next call.
    """
    batch, vocab = logits.shape
    key = (batch, vocab, logits.device)
    bufs = _BUF_CACHE.get(key)
    if bufs is None:
        BLOCK = 4096
        tiles_total = triton.cdiv(vocab, BLOCK)
        # aim for ~2048 programs in pass 1 without exceeding the tile count
        # (swept on C500 [64, 130560]: 32 splits / 4 warps is the sweet spot)
        num_splits = min(_MAX_SPLITS, max(1, 2048 // max(batch, 1)), tiles_total)
        tiles_per_split = triton.cdiv(tiles_total, num_splits)
        bufs = dict(
            num_splits=num_splits,
            tiles_per_split=tiles_per_split,
            vocab_padded=num_splits * tiles_per_split * BLOCK,
            pval=torch.empty(batch, num_splits, device=logits.device, dtype=torch.float32),
            pidx=torch.empty(batch, num_splits, device=logits.device, dtype=torch.int32),
            out=torch.empty(batch, device=logits.device, dtype=torch.int32),
        )
        _BUF_CACHE[key] = bufs

    sampling = temperature is not None
    step = _STEP_COUNTER["value"]
    _STEP_COUNTER["value"] += 1
    grid = (batch, bufs["num_splits"])
    _fused_sample_partial[grid](
        logits,
        temperature if sampling else logits,  # TEMP unused when not sampling
        bufs["pval"],
        bufs["pidx"],
        logits.stride(0),
        vocab,
        bufs["vocab_padded"],
        seed,
        step,
        bufs["tiles_per_split"],
        bufs["num_splits"],
        BLOCK=4096,
        SAMPLING=sampling,
        num_warps=4,
    )
    _fused_sample_reduce[(batch,)](
        bufs["pval"],
        bufs["pidx"],
        bufs["out"],
        bufs["num_splits"],
        MAX_SPLITS=_MAX_SPLITS,
        num_warps=1,
    )
    return bufs["out"]
