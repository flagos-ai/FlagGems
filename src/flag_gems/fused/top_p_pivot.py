"""Guarded weighted-pivot top-p for MetaX C500.

Uses one online-statistics pass, a safeguarded weighted-quantile pivot search
(one full row scan per pivot), then one in-place mask pass.  The Gaussian
logit model supplies only the initial guess; brackets and value/tie checks
retain the selection rule. Numerically ambiguous rows use the original
sort/softmax/cumsum rule; graph capture also keeps that device-only reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _normal_inv_cdf(q):
    # Acklam's inverse-normal approximation; avoids host synchronization and
    # is used only as a starting point for the safeguarded exact pivot search.
    q = tl.minimum(tl.maximum(q, 1.0e-7), 1.0 - 1.0e-7)
    a1 = -39.69683028665376
    a2 = 220.9460984245205
    a3 = -275.9285104469687
    a4 = 138.3577518672690
    a5 = -30.66479806614716
    a6 = 2.506628277459239
    b1 = -54.47609879822406
    b2 = 161.5858368580409
    b3 = -155.6989798598866
    b4 = 66.80131188771972
    b5 = -13.28068155288572
    c1 = -0.007784894002430293
    c2 = -0.3223964580411365
    c3 = -2.400758277161838
    c4 = -2.549732539343734
    c5 = 4.374664141464968
    c6 = 2.938163982698783
    d1 = 0.007784695709041462
    d2 = 0.3224671290700398
    d3 = 2.445134137142996
    d4 = 3.754408661907416
    low = 0.02425
    qtail = tl.sqrt(-2.0 * tl.log(q))
    tail_num = (
        (((c1 * qtail + c2) * qtail + c3) * qtail + c4) * qtail + c5
    ) * qtail + c6
    tail_den = (((d1 * qtail + d2) * qtail + d3) * qtail + d4) * qtail + 1.0
    tail = tail_num / tail_den
    r = q - 0.5
    s = r * r
    center_num = (((((a1 * s + a2) * s + a3) * s + a4) * s + a5) * s + a6) * r
    center_den = ((((b1 * s + b2) * s + b3) * s + b4) * s + b5) * s + 1.0
    center = center_num / center_den
    return tl.where(q < low, tail, tl.where(q > 1.0 - low, -tail, center))


@triton.jit
def _top_p_pivot_fast(
    X,
    P,
    V: tl.constexpr,
    RS: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_ITERS: tl.constexpr,
    Flags=None,
    GUARD: tl.constexpr = False,
):
    row = tl.program_id(0)
    ix = tl.arange(0, BLOCK)

    # Accumulate row reductions lane-wise while streaming tiles, then reduce
    # once per statistic. This avoids a shared-memory reduction/barrier for
    # every vocabulary tile.
    vec_max = tl.full((BLOCK,), -float("inf"), tl.float32)
    vec_min = tl.full((BLOCK,), float("inf"), tl.float32)
    vec_sum_x = tl.zeros((BLOCK,), tl.float32)
    vec_sum_x2 = tl.zeros((BLOCK,), tl.float32)
    vec_n = tl.zeros((BLOCK,), tl.int32)
    vec_nan = tl.full((BLOCK,), False, tl.int1)
    for tile in range(tl.cdiv(V, BLOCK)):
        j = tile * BLOCK + ix
        x = tl.load(X + row * RS + j, j < V, other=-float("inf"))
        finite = (j < V) & (x > -float("inf"))
        vec_nan = vec_nan | ((j < V) & (x != x))
        vec_max = tl.maximum(vec_max, tl.where(finite, x, -float("inf")))
        vec_min = tl.minimum(vec_min, tl.where(finite, x, float("inf")))
        vec_sum_x += tl.where(finite, x, 0.0)
        vec_sum_x2 += tl.where(finite, x * x, 0.0)
        vec_n += finite.to(tl.int32)
    mx = tl.max(vec_max, 0)
    mn = tl.min(vec_min, 0)
    sum_x = tl.sum(vec_sum_x, 0)
    sum_x2 = tl.sum(vec_sum_x2, 0)
    n = tl.sum(vec_n, 0)
    has_nan = tl.max(vec_nan.to(tl.int32), 0) != 0

    # One additional streaming pass computes the stable normalizer and max ties.
    vec_total = tl.zeros((BLOCK,), tl.float32)
    vec_max_count = tl.zeros((BLOCK,), tl.int32)
    for tile in range(tl.cdiv(V, BLOCK)):
        j = tile * BLOCK + ix
        x = tl.load(X + row * RS + j, j < V, other=-float("inf"))
        finite = (j < V) & (x > -float("inf"))
        vec_total += tl.where(finite, tl.exp(x - mx), 0.0)
        vec_max_count += (finite & (x == mx)).to(tl.int32)
    total = tl.sum(vec_total, 0)
    max_count = tl.sum(vec_max_count, 0)

    p = tl.load(P + row)
    ambiguous = (
        (p <= 0.0)
        | (p >= 1.0)
        | (p != p)
        | (mn == mx)
        | (n == 0)
        | has_nan
        | (mx == float("inf"))
        | (sum_x2 == float("inf"))
    )
    pivot = -float("inf")
    remove_equal = 0
    if (p < 1.0) & (n > 0) & (mx < float("inf")) & (has_nan == 0):
        target = (1.0 - p) * total
        mu = sum_x / n
        variance = tl.maximum(sum_x2 / n - mu * mu, 0.0)
        sigma = tl.sqrt(variance)

        # Exact shortcut for a constant row; rank selection handles ties.
        if mn == mx:
            pivot = mx
            ratio = (1.0 - p) * n
            nearest = tl.floor(ratio + 0.5)
            ratio = tl.where(tl.abs(ratio - nearest) < 1.0e-4, nearest, ratio)
            remove_equal = tl.minimum(
                n - 1, tl.maximum(0, tl.floor(ratio).to(tl.int32))
            )
        elif p <= 0.0:
            pivot = mx
            remove_equal = tl.maximum(0, max_count - 1)
        else:
            q = 1.0 - p
            # Under exponential tilting a normal N(mu,sigma^2) becomes
            # N(mu+sigma^2,sigma^2). This is only the first trial pivot.
            guess = mu + variance + sigma * _normal_inv_cdf(q)
            guess = tl.minimum(tl.maximum(guess, mn), mx)
            # A sharp row makes the normal approximation overshoot to mx.
            # Starting exactly at mx yields an empty ``x > mid`` bucket and
            # prematurely selects mx; use the bracket midpoint in that case.
            guess = tl.where((guess >= mx) | (guess <= mn), 0.5 * (mn + mx), guess)
            lo = mn
            hi = mx
            flo = 0.0
            fhi = total
            mid = guess
            found = 0
            it = 0
            while (found == 0) & (it < MAX_ITERS):
                vec_below = tl.zeros((BLOCK,), tl.float32)
                vec_nxt = tl.full((BLOCK,), float("inf"), tl.float32)
                vec_nxt_count = tl.zeros((BLOCK,), tl.int32)
                for tile in range(tl.cdiv(V, BLOCK)):
                    j = tile * BLOCK + ix
                    x = tl.load(X + row * RS + j, j < V, other=-float("inf"))
                    valid = (j < V) & (x > -float("inf"))
                    vec_below += tl.where(valid & (x <= mid), tl.exp(x - mx), 0.0)
                    candidate = tl.where(valid & (x > mid), x, float("inf"))
                    is_new = candidate < vec_nxt
                    is_same = candidate == vec_nxt
                    vec_nxt_count = tl.where(
                        is_new, 1, vec_nxt_count + is_same.to(tl.int32)
                    )
                    vec_nxt = tl.minimum(vec_nxt, candidate)
                below = tl.sum(vec_below, 0)
                nxt = tl.min(vec_nxt, 0)
                nxt_count = tl.sum(tl.where(vec_nxt == nxt, vec_nxt_count, 0), 0)
                weight = tl.where(nxt < float("inf"), tl.exp(nxt - mx), 0.0)
                next_mass = below + nxt_count * weight
                crosses = (
                    (below <= target) & (next_mass >= target) & (nxt < float("inf"))
                )
                if crosses:
                    ratio = (target - below) / weight
                    nearest = tl.floor(ratio + 0.5)
                    # A different reduction order can move the FP32 CDF across
                    # a token boundary. Keep such rows on the original rule.
                    uncertainty = 4.0e-6 * target / weight + 1.0e-3
                    ambiguous = ambiguous | (tl.abs(ratio - nearest) <= uncertainty)
                    ratio = tl.where(tl.abs(ratio - nearest) < 1.0e-4, nearest, ratio)
                    remove_equal = tl.minimum(
                        nxt_count, tl.maximum(0, tl.floor(ratio).to(tl.int32))
                    )
                    pivot = nxt
                    found = 1
                else:
                    if below > target:
                        hi = mid
                        fhi = below
                    else:
                        lo = nxt
                        flo = next_mass
                    width = hi - lo
                    denom = fhi - flo
                    frac = tl.where(denom > 0.0, (target - flo) / denom, 0.5)
                    secant = lo + width * frac
                    lower_guard = lo + width * 0.1
                    upper_guard = hi - width * 0.1
                    mid = tl.where(
                        (secant > lower_guard) & (secant < upper_guard),
                        secant,
                        lo + width * 0.5,
                    )
                    if nxt == float("inf"):
                        # No greater finite value exists; the maximum is the
                        # only valid bucket and one instance must survive.
                        pivot = mx
                        remove_equal = tl.maximum(0, max_count - 1)
                        found = 1
                it += 1

            # A capped secant search has not proved that its last bucket is
            # the quantile. Finish by binary search over ordered FP32 bit keys.
            # At most 32 decisions cover the entire finite FP32 domain.
            if found == 0:
                min_bits = mn.to(tl.uint32, bitcast=True)
                max_bits = mx.to(tl.uint32, bitcast=True)
                low_key = tl.where(
                    (min_bits & 0x80000000) != 0, ~min_bits, min_bits ^ 0x80000000
                ).to(tl.uint32)
                high_key = tl.where(
                    (max_bits & 0x80000000) != 0, ~max_bits, max_bits ^ 0x80000000
                ).to(tl.uint32)
                for _ in range(32):
                    if low_key < high_key:
                        key = low_key + ((high_key - low_key) >> 1)
                        bits = tl.where(
                            (key & 0x80000000) != 0, key ^ 0x80000000, ~key
                        ).to(tl.uint32)
                        trial = bits.to(tl.float32, bitcast=True)
                        vec_mass = tl.zeros((BLOCK,), tl.float32)
                        for tile in range(tl.cdiv(V, BLOCK)):
                            j = tile * BLOCK + ix
                            x = tl.load(X + row * RS + j, j < V, other=-float("inf"))
                            valid = (j < V) & (x > -float("inf"))
                            vec_mass += tl.where(
                                valid & (x <= trial), tl.exp(x - mx), 0.0
                            )
                        mass = tl.sum(vec_mass, 0)
                        if mass > target:
                            high_key = key
                        else:
                            low_key = key + 1
                pivot_bits = tl.where(
                    (low_key & 0x80000000) != 0, low_key ^ 0x80000000, ~low_key
                ).to(tl.uint32)
                pivot = pivot_bits.to(tl.float32, bitcast=True)
                vec_below_final = tl.zeros((BLOCK,), tl.float32)
                vec_eq_final = tl.zeros((BLOCK,), tl.int32)
                for tile in range(tl.cdiv(V, BLOCK)):
                    j = tile * BLOCK + ix
                    x = tl.load(X + row * RS + j, j < V, other=-float("inf"))
                    valid = (j < V) & (x > -float("inf"))
                    vec_below_final += tl.where(
                        valid & (x < pivot), tl.exp(x - mx), 0.0
                    )
                    vec_eq_final += (valid & (x == pivot)).to(tl.int32)
                below_final = tl.sum(vec_below_final, 0)
                eq_final = tl.sum(vec_eq_final, 0)
                weight_final = tl.where(pivot < float("inf"), tl.exp(pivot - mx), 0.0)
                ratio_final = tl.where(
                    weight_final > 0.0, (target - below_final) / weight_final, 0.0
                )
                ratio_final = tl.maximum(ratio_final, 0.0)
                nearest_final = tl.floor(ratio_final + 0.5)
                uncertainty_final = (
                    4.0e-6 * target / tl.maximum(weight_final, 1.0e-38) + 1.0e-3
                )
                ambiguous = ambiguous | (
                    tl.abs(ratio_final - nearest_final) <= uncertainty_final
                )
                ratio_final = tl.where(
                    tl.abs(ratio_final - nearest_final) < 1.0e-4,
                    nearest_final,
                    ratio_final,
                )
                remove_equal = tl.minimum(
                    eq_final, tl.maximum(0, tl.floor(ratio_final).to(tl.int32))
                )
                remove_equal = tl.where(
                    pivot == mx, tl.minimum(remove_equal, max_count - 1), remove_equal
                )

    if GUARD:
        tl.store(Flags + row, ambiguous.to(tl.int32))
    seen = 0
    for tile in range(tl.cdiv(V, BLOCK)):
        j = tile * BLOCK + ix
        x = tl.load(X + row * RS + j, j < V, other=-float("inf"))
        finite = (j < V) & (x > -float("inf"))
        eq = finite & (x == pivot)
        rank = seen + tl.cumsum(eq.to(tl.int32), 0)
        seen += tl.sum(eq.to(tl.int32), 0)
        keep = finite & ((x > pivot) | (eq & (rank > remove_equal)))
        # Preserve an all-masked input unchanged.
        keep = tl.where(n == 0, j < V, keep)
        tl.store(X + row * RS + j, tl.where(keep, x, -float("inf")), j < V)


def _reference_top_p_(logits, p):
    values, indices = logits.sort(dim=-1, descending=False)
    probabilities = values.softmax(dim=-1)
    cdf = torch.cumsum(probabilities, dim=-1, out=probabilities)
    remove = cdf <= 1 - p.unsqueeze(dim=1)
    remove[:, -1] = False
    values.masked_fill_(remove, -float("inf"))
    return logits.scatter_(dim=-1, index=indices, src=values)


def apply_top_p_pivot_(
    logits: torch.Tensor, p: torch.Tensor, *, block=8192, max_iters=40
):
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or logits.stride(1) != 1
        or not logits.is_cuda
        or logits.shape[1] == 0
    ):
        raise ValueError("Expected FP32 [B,V] with unit vocabulary stride")
    if (
        p.ndim != 1
        or p.shape[0] != logits.shape[0]
        or p.dtype != torch.float32
        or p.device != logits.device
        or p.stride(0) != 1
    ):
        raise ValueError("Expected contiguous FP32 p shaped [B] on the logits device")
    if logits.shape[0] == 0:
        return logits
    if torch.cuda.is_current_stream_capturing():
        return _reference_top_p_(logits, p)
    original = logits.clone()
    flags = torch.empty(logits.shape[0], device=logits.device, dtype=torch.int32)
    _top_p_pivot_fast[(logits.shape[0],)](
        logits,
        p,
        logits.shape[1],
        logits.stride(0),
        BLOCK=block,
        MAX_ITERS=max_iters,
        Flags=flags,
        GUARD=True,
        num_warps=16,
        enable_fp_fusion=False,
    )
    # This is outside model-forward graph replay. The small index readback is
    # deliberate: only uncertain rows incur the original vocabulary sort.
    rows = torch.nonzero(flags, as_tuple=False).flatten()
    if rows.numel():
        repaired = _reference_top_p_(
            original.index_select(0, rows), p.index_select(0, rows)
        )
        logits.index_copy_(0, rows, repaired)
    return logits
