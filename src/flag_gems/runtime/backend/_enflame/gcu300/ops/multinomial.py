import logging

import torch
import triton
import triton.language as tl

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
    y_off = tl.program_id(0) * N
    n = tl.program_id(1) * NBLOCK + tl.arange(0, NBLOCK)
    rv, _, _, _ = uniform(philox_seed, philox_offset, y_off + n)

    # Do a binary search for each random number on the cumulative probabilities.
    # Each random number always selects the leftmost index of the data greater
    # than or equal to itself. However, this is likely to give a wrong result
    # in case the first probability is zero which is not expected to selected.
    # This error happens when the tossed random number is also zero. To avoid
    # this mistake, we simply perturb random variable with a small number.
    rv += 0.0001
    rv = tl.where(rv > 0.9999, 0.9999, rv)

    cdf_ptr += tl.program_id(0) * K
    start = tl.zeros((NBLOCK,), dtype=tl.int32)
    end = tl.zeros((NBLOCK,), dtype=tl.int32) + K - 1
    steps = tl.math.log2(K.to(tl.float32)).to(tl.int32) + 1
    for _ in range(steps):
        mid = start + (end - start) // 2
        x = tl.load(cdf_ptr + mid, mask=n < N)
        start = tl.where(x < rv, mid + 1, start)
        end = tl.where(x < rv, end, mid)

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
        # In case of with_replacement, sampling is approximated by selecing
        # the top k indices over sorted probabilities with an exponential pertubation
        # s = argmax( p / q ) where q ~ Exp(1)
        q = torch.empty_like(prob).exponential_(1.0)
        s = torch.div(prob, q, out=q)
        if n_samples == 1:
            return torch.argmax(s, dim=-1, keepdim=True).to(torch.int64)
        else:
            vals, indices = torch.topk(s, n_samples, dim=-1)
            return indices.to(torch.int64)

    from .cumsum import normed_cumsum

    cum_prob = normed_cumsum(prob, dim=-1)

    if cum_prob.dim() == 1:
        n_dist = 1
        # out = torch.empty((n_samples,), device=prob.device, dtype=torch.int64)
        out = torch.empty((n_samples,), device=prob.device, dtype=torch.int32)
    else:
        n_dist = cum_prob.size(0)
        # gcu300: don't support int64
        # out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int64)
        out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int32)
    # The CTA level parallelism is framed in a 2d grid of blocks with grid.y
    # indexing into distributions and grid.x output sample batches
    increment = n_dist * n_samples
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)

    # gcu300: don't support int64, convert int64 to int32
    philox_tensor = torch.tensor([philox_seed, philox_offset], dtype=torch.int64)
    philox_tensor = philox_tensor.to(torch.int32)
    philox_list = philox_tensor.tolist()
    philox_seed = philox_list[0]
    philox_offset = philox_list[1]

    grid = lambda META: (n_dist, triton.cdiv(n_samples, META["NBLOCK"]))
    multinomial_with_replacement[grid](
        cum_prob, out, n_categories, n_samples, philox_seed, philox_offset
    )
    return out.to(torch.int64)
