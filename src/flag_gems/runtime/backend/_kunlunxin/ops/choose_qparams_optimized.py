"""choose_qparams_optimized as one fused Triton kernel.

Semantics (recovered from the ATen CPU reference by numerical probing):
  xmin = min(x[:numel]), xmax = max(x[:numel])
  stepsize  = (xmax - xmin) / n_bins              (fp32)
  min_bins  = int(n_bins * (1 - ratio))           (host double arithmetic)
  threshold = min_bins * stepsize
  lo, hi = xmin, xmax ; cur_loss = err(lo, hi)
  while lo + threshold < hi:
      l1 = err(lo + stepsize, hi)
      l2 = err(lo, hi - stepsize)
      if cur_loss < l1 and cur_loss < l2: break
      if l1 < l2: lo += stepsize ; cur_loss = l1
      else:       hi -= stepsize ; cur_loss = l2
  return (hi, lo)                       # (max_out, min_out), shape [1] each

err(lo, hi) = sum_k (x_k - (q_k * scale + lo_h))^2 with
  lo_h  = half(lo)
  scale = half((hi - lo_h) / qmax)   (1.0 when hi == lo_h)
  q_k   = clamp(round_half_even((x_k - lo_h) / scale), 0, qmax)

Everything (min/max, every candidate error sum and the greedy walk) runs on the
device in a single launch: the input tile lives in registers and the loop only
exchanges rank-0 scalars.

Cost structure on this backend (measured): ~0.3 us per dependent reduction plus
~0.3 us per candidate error evaluation, while a block reduction cannot be
lowered inside a `while`/dynamic `if` (only inside a runtime `for`).  The walk
therefore uses early-exit-guarded single-trip loops: after one fused 3-result
reduction covers the initial loss plus both first-step children, NUNROLL steps
are emitted as separate runtime `for` loops whose bound is tl.where(cont, 1, 0)
- a zero-trip `for` is legal and free, unlike an `if` around a reduce - and a
final tail loop (bound tl.where(cont, nsteps - NUNROLL, 0)) covers the rare deep
walk.  Executed work then scales with the length of the actual walk (0-3 steps
for the benchmarked inputs) instead of the worst-case n_bins - min_bins.
"""

import torch
import triton
import triton.language as tl

# 1.5 * 2^23: adding and subtracting this rounds an fp32 value to the nearest
# integer with ties-to-even, matching libdevice nearbyint / std::nearbyint.
# Passed as a runtime scalar so the compiler cannot fold the pair away.
_RND_MAGIC = 12582912.0

# steps emitted behind individual early-exit guards (measured walk lengths on
# randn inputs at the benchmark settings are 0-3)
_NUNROLL = 6


@triton.jit
def _half_round(v):
    """Emulate an at::Half round-trip, as the reference does for lo/scale."""
    return v.to(tl.float16).to(tl.float32)


@triton.jit
def _qsq(x, mask, lo, hi, qmax, magic):
    """Per-element squared quantization error of the candidate range (lo, hi)."""
    lo_h = _half_round(lo)
    data_range = hi - lo_h
    scale = tl.where(data_range == 0.0, 1.0, _half_round(data_range / qmax))
    inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
    y = (x - lo_h) * inv
    q = (y + magic) - magic  # round half to even
    q = tl.minimum(tl.maximum(q, 0.0), qmax)
    d = x - (q * scale + lo_h)
    return tl.where(mask, d * d, 0.0)


@triton.jit
def _minmax(a0, a1, b0, b1):
    return tl.minimum(a0, b0), tl.maximum(a1, b1)


@triton.jit
def _add3(a0, a1, a2, b0, b1, b2):
    return a0 + b0, a1 + b1, a2 + b2


@triton.jit
def _add2(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@triton.jit
def _choose_qparams_optimized_kernel(
    x_ptr,
    out_ptr,
    numel,
    n_bins,
    min_bins,
    qmax,
    magic,
    nsteps,
    NUNROLL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    xmin, xmax = tl.reduce(
        (
            tl.where(mask, x, float("inf")),
            tl.where(mask, x, float("-inf")),
        ),
        0,
        _minmax,
    )

    stepsize = (xmax - xmin) / n_bins
    threshold = min_bins * stepsize

    # reference step 0: loss of the full range and of both one-step children
    lo = xmin
    hi = xmax
    lo1 = lo + stepsize
    hi1 = hi - stepsize
    cur_loss, loss_lo, loss_hi = tl.reduce(
        (
            _qsq(x, mask, lo, hi, qmax, magic),
            _qsq(x, mask, lo1, hi, qmax, magic),
            _qsq(x, mask, lo, hi1, qmax, magic),
        ),
        0,
        _add3,
    )
    cont = (lo + threshold) < hi
    go = cont & ((cur_loss >= loss_lo) | (cur_loss >= loss_hi))
    move_lo = go & (loss_lo < loss_hi)
    move_hi = go & (loss_lo >= loss_hi)
    lo = tl.where(move_lo, lo1, lo)
    hi = tl.where(move_hi, hi1, hi)
    cur_loss = tl.where(move_lo, loss_lo, tl.where(move_hi, loss_hi, cur_loss))
    cont = go & ((lo + threshold) < hi)

    # early-exit-guarded steps: the body only runs while the walk is still going
    for _d in tl.static_range(0, NUNROLL):
        one = tl.where(cont, 1, 0)
        for _k in range(0, one):
            lo1 = lo + stepsize
            hi1 = hi - stepsize
            loss_lo, loss_hi = tl.reduce(
                (
                    _qsq(x, mask, lo1, hi, qmax, magic),
                    _qsq(x, mask, lo, hi1, qmax, magic),
                ),
                0,
                _add2,
            )
            go = cont & ((cur_loss >= loss_lo) | (cur_loss >= loss_hi))
            move_lo = go & (loss_lo < loss_hi)
            move_hi = go & (loss_lo >= loss_hi)
            lo = tl.where(move_lo, lo1, lo)
            hi = tl.where(move_hi, hi1, hi)
            cur_loss = tl.where(
                move_lo, loss_lo, tl.where(move_hi, loss_hi, cur_loss)
            )
            cont = go & ((lo + threshold) < hi)

    # remaining worst-case depth (rarely executed, e.g. crafted inputs)
    rest = tl.maximum(nsteps - NUNROLL, 0)
    tail = tl.where(cont, rest, 0)
    for _k in range(0, tail):
        lo1 = lo + stepsize
        hi1 = hi - stepsize
        loss_lo, loss_hi = tl.reduce(
            (
                _qsq(x, mask, lo1, hi, qmax, magic),
                _qsq(x, mask, lo, hi1, qmax, magic),
            ),
            0,
            _add2,
        )
        go = cont & ((cur_loss >= loss_lo) | (cur_loss >= loss_hi))
        move_lo = go & (loss_lo < loss_hi)
        move_hi = go & (loss_lo >= loss_hi)
        lo = tl.where(move_lo, lo1, lo)
        hi = tl.where(move_hi, hi1, hi)
        cur_loss = tl.where(move_lo, loss_lo, tl.where(move_hi, loss_hi, cur_loss))
        cont = go & ((lo + threshold) < hi)

    tl.store(out_ptr + 0, hi)
    tl.store(out_ptr + 1, lo)


def run(input, numel, n_bins, ratio, bit_width):
    if input.dtype != torch.float32:
        raise AssertionError(
            "choose_qparams_optimized only supports float32 input, "
            f"got {input.dtype}"
        )

    n = int(numel)
    nb = int(n_bins)
    min_bins = int(nb * (1.0 - float(ratio)))
    qmax = float((1 << int(bit_width)) - 1)

    # worst-case walk depth (plus slack for fp ties); the unrolled steps are
    # early-exit guarded, so the typical cost is far below this bound
    nsteps = nb - min_bins + 2
    if nsteps < 1:
        nsteps = 1

    x = input.view(-1)
    out = torch.empty(2, dtype=torch.float32, device=input.device)

    block_n = triton.next_power_of_2(n if n > 0 else 1)
    if block_n >= 4096:
        num_warps = 8
    elif block_n >= 1024:
        num_warps = 4
    elif block_n >= 256:
        num_warps = 2
    else:
        num_warps = 1
    _choose_qparams_optimized_kernel[(1,)](
        x,
        out,
        n,
        float(nb),
        float(min_bins),
        qmax,
        _RND_MAGIC,
        nsteps,
        NUNROLL=_NUNROLL,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out[0:1], out[1:2]

# --- backend entry point ---------------------------------------------------
def choose_qparams_optimized(input, numel, n_bins, ratio, bit_width):
    return run(input, numel, n_bins, ratio, bit_width)
