import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset, uint_to_uniform_float

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(
    do_not_specialize=[
        "K",
        "N",
        "philox_seed_lo",
        "philox_seed_hi",
        "philox_offset_lo",
        "philox_offset_hi",
    ]
)
def multinomial_with_replacement(
    cdf_ptr,
    out_ptr,
    K,
    N,
    philox_seed_lo: tl.uint32,
    philox_seed_hi: tl.uint32,
    philox_offset_lo: tl.uint32,
    philox_offset_hi: tl.uint32,
    NBLOCK: tl.constexpr = 512,
):
    y_off = tl.program_id(1) * N
    n = tl.program_id(0) * NBLOCK + tl.arange(0, NBLOCK)
    c0 = philox_offset_lo + y_off + n
    c1 = philox_offset_hi
    _O = c0 * 0
    rv, _, _, _ = tl.philox(philox_seed_lo, philox_seed_hi, c0, c1, _O, _O)
    rv = uint_to_uniform_float(rv)

    rv += 0.0001
    rv = tl.where(rv > 0.9999, 0.9999, rv)

    cdf_ptr += tl.program_id(1) * K
    start = tl.zeros((NBLOCK,), dtype=tl.int32)
    end = tl.zeros((NBLOCK,), dtype=tl.int32) + K - 1
    steps = tl.math.log2(K.to(tl.float32)).to(tl.int32) + 1
    for _ in range(steps):
        mid = start + (end - start) // 2
        x = tl.load(cdf_ptr + mid, mask=n < N)
        start = tl.where(x < rv, mid + 1, start)
        end = tl.where(x < rv, end, mid)

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

    if not with_replacement:
        if n_categories == 1:
            return torch.zeros((*prob.shape[:-1], n_samples), dtype=torch.int64)
        return torch.multinomial(prob.cpu(), n_samples, False, generator=gen)

    if n_samples == 1:
        q = torch.empty_like(prob).exponential_(1.0)
        s = torch.div(prob, q, out=q)
        if n_samples == 1:
            return torch.argmax(s, dim=-1, keepdim=True).to(torch.int64)
        vals, indices = torch.topk(s, n_samples, dim=-1)
        return indices.to(torch.int64)

    prob_cpu = prob.cpu()
    cum_prob = torch.cumsum(prob_cpu, dim=-1) / torch.sum(prob_cpu, dim=-1, keepdim=True)
    cum_prob = cum_prob.to(prob.device)

    if cum_prob.dim() == 1:
        n_dist = 1
        out = torch.empty((n_samples,), device=prob.device, dtype=torch.int32)
    else:
        n_dist = cum_prob.size(0)
        out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int32)
    increment = n_dist * n_samples
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)
    philox_seed_hi = (philox_seed >> 32) & 0xFFFFFFFF
    philox_seed_lo = philox_seed & 0xFFFFFFFF
    philox_offset_hi = (philox_offset >> 32) & 0xFFFFFFFF
    philox_offset_lo = philox_offset & 0xFFFFFFFF
    grid = lambda META: (triton.cdiv(n_samples, META["NBLOCK"]), n_dist)
    multinomial_with_replacement[grid](
        cum_prob,
        out,
        n_categories,
        n_samples,
        philox_seed_lo,
        philox_seed_hi,
        philox_offset_lo,
        philox_offset_hi,
    )
    return out
