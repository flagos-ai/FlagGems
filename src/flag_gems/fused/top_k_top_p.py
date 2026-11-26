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

"""Sort-free fused top-k / top-p (nucleus) logits filtering.

vLLM's Qrita top-k/top-p Triton kernel fails to compile on MetaX
(PassManager::run failed in ttgir stage), so serving falls back to the eager
PyTorch path which fully sorts the vocab per row per decode step.  For a
130k-token vocab at batch 64 that costs ~25 ms per step — over a third of
decode time.

This kernel produces the same masked logits as
``vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p_pytorch``
(logits kept where they pass the filters, ``-inf`` elsewhere) without any
sort:

1. one online pass computes the row max ``m`` and partition sum
   ``Z = sum(exp(x - m))``;
2. bisection over the logit range finds the largest threshold ``tau`` such
   that the set ``{x >= tau}`` contains at most ``k`` elements and carries at
   least ``p * Z`` softmax mass;
3. a final pass masks everything below ``tau`` with ``-inf``.

Each bisection round re-reads the row (from L2 after the first pass), so the
whole filter costs ~23 row-passes regardless of vocab size, instead of an
O(V log V) sort.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _top_k_top_p_kernel(
    LOGITS,
    K,
    P,
    stride_row,
    vocab_size,
    num_tiles,
    k_enabled: tl.constexpr,
    p_enabled: tl.constexpr,
    BISECT_ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = LOGITS + row * stride_row
    offs = tl.arange(0, BLOCK)

    # ---- pass 1: online max + partition sum -------------------------------
    m = float("-inf")
    Z = 0.0
    for t in range(num_tiles):
        idx = t * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, tile_max)
        # rescale accumulator when the running max moves
        Z = Z * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    # load per-row k / p; disabled filters use neutral values
    k_limit = vocab_size
    if k_enabled:
        k_limit = tl.load(K + row).to(tl.int32)

    # ---- top-k: threshold = k-th largest value ----------------------------
    # largest tau with count{x >= tau} >= k  (keeps ties, like the reference)
    tau_k = m - 60.0  # exp(-60) is far below fp32 softmax resolution
    if k_enabled:
        lo_k = m - 60.0
        hi_k = m
        for _ in range(BISECT_ITERS):
            mid = (lo_k + hi_k) / 2.0
            cnt = 0
            for t in range(num_tiles):
                idx = t * BLOCK + offs
                mask = idx < vocab_size
                x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
                cnt += tl.sum((x >= mid).to(tl.int32), axis=0)
            if cnt >= k_limit:
                lo_k = mid
            else:
                hi_k = mid
        tau_k = lo_k

    # ---- top-p: threshold on softmax mass within the k-kept set -----------
    tau = tau_k
    if p_enabled:
        # reference renormalizes over the top-k-kept set first
        zkept = 0.0
        for t in range(num_tiles):
            idx = t * BLOCK + offs
            mask = idx < vocab_size
            x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
            zkept += tl.sum(tl.where(x >= tau_k, tl.exp(x - m), 0.0), axis=0)
        target = tl.load(P + row) * zkept
        # largest tau in [tau_k, m] with mass{x >= tau} >= target
        lo_p = tau_k
        hi_p = m
        for _ in range(BISECT_ITERS):
            mid = (lo_p + hi_p) / 2.0
            mass = 0.0
            for t in range(num_tiles):
                idx = t * BLOCK + offs
                mask = idx < vocab_size
                x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
                mass += tl.sum(tl.where(x >= mid, tl.exp(x - m), 0.0), axis=0)
            if mass >= target:
                lo_p = mid
            else:
                hi_p = mid
        tau = lo_p

    # ---- pass 3: apply the mask -------------------------------------------
    for t in range(num_tiles):
        idx = t * BLOCK + offs
        mask = idx < vocab_size
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        x = tl.where(x >= tau, x, float("-inf"))
        tl.store(base + idx, x.to(LOGITS.dtype.element_ty), mask=mask)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    bisect_iters: int = 20,
) -> torch.Tensor:
    """Apply top-k / top-p filtering to ``logits`` in-place (same semantics
    and return convention as vLLM's ``apply_top_k_top_p_pytorch``).

    Args:
        logits: [batch, vocab] tensor, modified in place and returned.
        k: optional int32/int64 [batch] tensor of per-row top-k limits.
        p: optional float [batch] tensor of per-row nucleus thresholds.
        bisect_iters: bisection rounds; 20 gives ~1e-4 relative threshold
            precision over a 60-wide logit range.
    """
    if k is None and p is None:
        return logits
    batch, vocab = logits.shape
    BLOCK = 4096
    num_tiles = triton.cdiv(vocab, BLOCK)
    _top_k_top_p_kernel[(batch,)](
        logits,
        k if k is not None else logits,  # dummy ptr when disabled
        p if p is not None else logits,
        logits.stride(0),
        vocab,
        num_tiles,
        k_enabled=k is not None,
        p_enabled=p is not None,
        BISECT_ITERS=bisect_iters,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return logits
