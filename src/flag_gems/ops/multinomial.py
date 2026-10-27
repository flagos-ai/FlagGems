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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops import normed_cumsum
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset, uniform

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["K", "N", "philox_seed", "philox_offset"])
def multinomial_with_replacement(
    cdf_ptr, out_ptr, K, N, philox_seed, philox_offset, NBLOCK: tl.constexpr = 128
):
    # The computation is arranged in a 2d grid of blocks, each producing
    # a batch of samples for a particular distribution.
    #            <------------------- grid.x --------------------->
    #           |   dist0.batch0 | dist0.batch1 | dist0.batch2 ...
    #   grid.y  |   dist1.batch0 | dist1.batch1 | dist1.batch2 ...
    #           |   dist2.batch0 | dist2.batch1 | dist2.batch2 ...
    y_off = tl.program_id(1) * N
    n = tl.program_id(0) * NBLOCK + tl.arange(0, NBLOCK)
    rv, _, _, _ = uniform(philox_seed, philox_offset, y_off + n)

    # Do a binary search for each random number on the cumulative
    # probabilities, selecting the leftmost index whose cumulative probability
    # is strictly greater than the random value (the textbook inverse-CDF
    # sample). The strict comparison makes zero-probability categories --
    # including a zero-probability leading category, even when the tossed
    # random value is zero -- impossible to select, so no perturbation of the
    # random value is needed. Perturbing it (e.g. rv += eps) would bias
    # sampling against small-probability categories whose cumulative mass falls
    # below the perturbation.
    cdf_ptr += tl.program_id(1) * K
    start = tl.zeros((NBLOCK,), dtype=tl.int32)
    end = tl.zeros((NBLOCK,), dtype=tl.int32) + K - 1
    steps = tl.math.log2(K.to(tl.float32)).to(tl.int32) + 1
    for _ in range(steps):
        mid = start + (end - start) // 2
        x = tl.load(cdf_ptr + mid, mask=n < N)
        start = tl.where(x <= rv, mid + 1, start)
        end = tl.where(x <= rv, end, mid)

    # Returns the last index in case of an overflow
    start = tl.where(start >= K, K - 1, start)

    tl.store(out_ptr + y_off + n, start, mask=n < N)


def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    logger.debug("GEMS MULTINOMIAL")
    assert prob.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    assert 0 < prob.dim() <= 2, "prob_dist must be 1 or 2 dim"
    n_categories = prob.size(-1)
    assert n_categories <= (1 << 24), "number of categories cannot exceed 2^24"
    assert (
        with_replacement or n_samples <= n_categories
    ), "cannot sample n_samples > prob.size(-1) samples without replacement."

    # Sampling without replacement
    if (not with_replacement) or n_samples == 1:
        # Gumbel-max trick: s = argmax( p / q ) where q ~ Exp(1)
        # IMPORTANT: Both prob and q must be computed in float32 to avoid:
        #   1. fp16/bf16 overflow when small q values cause prob/q -> inf
        #   2. Multiple inf values making topk return duplicate indices
        prob_f32 = prob.float()
        prob_f32.clamp_(min=1e-12)  # avoid 0/q=0 ties for zero-prob categories
        q = torch.empty(
            prob.shape, dtype=torch.float32, device=prob.device
        ).exponential_(1.0)
        q.clamp_(min=1e-20)  # prevent division overflow
        s = prob_f32 / q
        if n_samples == 1:
            return torch.argmax(s, dim=-1, keepdim=True).to(torch.int64)
        else:
            # Use sort instead of topk to avoid potential duplicates from
            # FlagGems' topk kernel (which uses approximate sorting that can
            # produce duplicate indices for very close float32 values).
            _, indices = torch.sort(s, dim=-1, descending=True, stable=True)
            return indices[..., :n_samples].to(torch.int64)

    cum_prob = normed_cumsum(prob, dim=-1)

    if cum_prob.dim() == 1:
        n_dist = 1
        out = torch.empty((n_samples,), device=prob.device, dtype=torch.int64)
    else:
        n_dist = cum_prob.size(0)
        out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int64)
    # The CTA level parallelism is framed in a 2d grid of blocks with grid.y
    # indexing into distributions and grid.x output sample batches
    increment = n_dist * n_samples
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)
    grid = lambda META: (triton.cdiv(n_samples, META["NBLOCK"]), n_dist)
    multinomial_with_replacement[grid](
        cum_prob, out, n_categories, n_samples, philox_seed, philox_offset
    )
    return out
