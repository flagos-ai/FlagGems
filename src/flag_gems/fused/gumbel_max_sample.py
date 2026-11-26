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

"""Fused Gumbel-Max sampling: single-pass categorical sampling from logits.

After top-k / top-p filtering (e.g. ``flag_gems.fused.top_k_top_p``), vLLM's
native sampler tail materializes softmax probs, exponential noise, a division
and an argmax — four full-vocab passes plus four kernel launches per decode
step.  This kernel replaces that tail with a single pass using the Gumbel-Max
identity:

    argmax_i(x_i + g_i),  g_i ~ Gumbel(0, 1)   ≡   sample ~ softmax(x)

- ``-inf`` logits (masked by top-k/top-p) are excluded automatically.
- Randomness comes from Triton's Philox generator; ``seed``/``step`` are plain
  scalars because this kernel is intended for the non-captured (outside
  cudagraph) sampler tail.  Do NOT capture it in a CUDA graph — the scalar
  seed would freeze; for graph use, load seed/offset from device pointers
  instead.
- Returned token ids are int64 to match torch.argmax semantics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gumbel_argmax_kernel(
    LOGITS,
    OUT,
    stride_row,
    vocab_size,
    vocab_padded,
    seed,
    step,
    num_tiles,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK)
    row_seed = seed + row

    best = float("-inf")
    best_idx = 0
    for t in range(num_tiles):
        idx = t * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        u = tl.rand(row_seed, step * vocab_padded + idx.to(tl.int32))
        # clamp away from 0/1 so log(-log(u)) stays finite
        u = tl.minimum(tl.maximum(u, 1e-20), 1.0 - 1e-7)
        g = -tl.log(-tl.log(u))
        score = x + g
        tile_best = tl.max(score, axis=0)
        tile_arg = tl.argmax(score, axis=0)
        take = tile_best > best
        best_idx = tl.where(take, t * BLOCK + tile_arg, best_idx)
        best = tl.maximum(best, tile_best)

    tl.store(OUT + row, best_idx.to(tl.int64))


_STEP_COUNTER = {"value": 0}


def gumbel_max_sample(logits: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Sample one token per row from softmax(logits) via the Gumbel-Max trick.

    Args:
        logits: [batch, vocab] float32 (or castable) tensor, already
            temperature-scaled and top-k/top-p masked.
        seed: base seed for the Philox generator.

    Returns:
        [batch] int64 tensor of sampled token ids.
    """
    batch, vocab = logits.shape
    BLOCK = 4096
    vocab_padded = triton.cdiv(vocab, BLOCK) * BLOCK
    out = torch.empty(batch, device=logits.device, dtype=torch.int64)
    step = _STEP_COUNTER["value"]
    _STEP_COUNTER["value"] += 1
    _gumbel_argmax_kernel[(batch,)](
        logits,
        out,
        logits.stride(0),
        vocab,
        vocab_padded,
        seed,
        step,
        triton.cdiv(vocab, BLOCK),
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out
