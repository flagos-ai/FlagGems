import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_nearbyint = tl_extra_shim.nearbyint

_MAX_BLOCK = 4096
# Candidate depth evaluated per launch. The greedy search is measured to settle
# within 0-3 steps on the tested distributions, so one launch normally covers
# the whole search; deeper searches simply take another round.
_DEPTH = 3


@triton.jit
def _range_loss(x_ptr, lo, hi, qmax, n_elements, n_tiles, BLOCK: tl.constexpr):
    """Sum of squared quantization errors of x over the range [lo, hi].

    ``lo`` and the derived scale are rounded through half precision first,
    matching the reference numerics.
    """
    lo_h = lo.to(tl.float16).to(tl.float32)
    data_range = hi - lo_h
    scale = tl.where(
        data_range == 0.0, 1.0, (data_range / qmax).to(tl.float16).to(tl.float32)
    )
    inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in range(0, n_tiles):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q = _nearbyint((x - lo_h) * inv)
        q = tl.minimum(tl.maximum(q, 0.0), qmax)
        d = x - (q * scale + lo_h)
        acc += tl.where(mask, d * d, 0.0)
    return tl.sum(acc)


@libentry()
@triton.jit
def _qloss_grid_kernel(
    x_ptr,
    buf_ptr,
    n_elements,
    n_tiles,
    n_bins_f,
    qmax,
    a0,
    b0,
    DP1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per candidate range of a DP1 x DP1 neighbourhood.

    Program ``(i, j)`` evaluates the range narrowed by ``a0 + i`` bins from the
    low side and ``b0 + j`` bins from the high side, and writes the triple
    (loss, high, low) for it. Evaluating a whole neighbourhood per launch lets
    the host finish the sequential search from a single readback instead of one
    readback per step.
    """
    pid = tl.program_id(0)
    i = pid // DP1
    j = pid % DP1

    vmin = tl.full((BLOCK,), float("inf"), tl.float32)
    vmax = tl.full((BLOCK,), float("-inf"), tl.float32)
    for t in range(0, n_tiles):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        lo_t = tl.load(x_ptr + offs, mask=mask, other=float("inf")).to(tl.float32)
        hi_t = tl.load(x_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        vmin = tl.minimum(vmin, lo_t)
        vmax = tl.maximum(vmax, hi_t)
    xmin = tl.min(vmin)
    xmax = tl.max(vmax)

    step = (xmax - xmin) / n_bins_f
    cmin = xmin + (a0 + i).to(tl.float32) * step
    cmax = xmax - (b0 + j).to(tl.float32) * step
    loss = _range_loss(x_ptr, cmin, cmax, qmax, n_elements, n_tiles, BLOCK)

    tl.store(buf_ptr + 2 + 3 * pid + 0, loss)
    tl.store(buf_ptr + 2 + 3 * pid + 1, cmax)
    tl.store(buf_ptr + 2 + 3 * pid + 2, cmin)
    if pid == 0:
        tl.store(buf_ptr + 0, xmin)
        tl.store(buf_ptr + 1, xmax)


def choose_qparams_optimized(input, numel, n_bins, ratio, bit_width):
    logger.debug("GEMS_KUNLUNXIN CHOOSE_QPARAMS_OPTIMIZED")

    assert input.dtype == torch.float32, (
        "choose_qparams_optimized only supports float32 input, " f"got {input.dtype}"
    )

    x = input.contiguous().view(-1)[:numel]
    n = x.numel()
    qmax = float((1 << bit_width) - 1)
    min_bins = int(n_bins * (1.0 - ratio))
    max_steps = max(n_bins - min_bins, 0)

    block = min(_MAX_BLOCK, triton.next_power_of_2(max(n, 1)))
    n_tiles = triton.cdiv(n, block)
    dp1 = _DEPTH + 1
    buf = torch.empty(2 + 3 * dp1 * dp1, dtype=torch.float32, device=input.device)

    a0 = 0
    b0 = 0
    while True:
        _qloss_grid_kernel[(dp1 * dp1,)](
            x,
            buf,
            n,
            n_tiles,
            float(n_bins),
            qmax,
            a0,
            b0,
            DP1=dp1,
            BLOCK=block,
            num_warps=4,
        )
        host = buf.tolist()
        xmin, xmax = host[0], host[1]
        stepsize = (xmax - xmin) / n_bins
        threshold = min_bins * stepsize

        i = 0
        j = 0
        cur_loss = host[2]
        cur_max = host[3]
        cur_min = host[4]
        exhausted = False
        while cur_min + threshold < cur_max and a0 + i + b0 + j < max_steps:
            if i == _DEPTH or j == _DEPTH:
                exhausted = True
                break
            loss1 = host[2 + 3 * ((i + 1) * dp1 + j)]
            loss2 = host[2 + 3 * (i * dp1 + (j + 1))]
            if cur_loss < loss1 and cur_loss < loss2:
                break
            if loss1 < loss2:
                i += 1
                cur_loss = loss1
            else:
                j += 1
                cur_loss = loss2
            slot = 2 + 3 * (i * dp1 + j)
            cur_max = host[slot + 1]
            cur_min = host[slot + 2]
        if not exhausted:
            break
        a0 += i
        b0 += j

    slot = 2 + 3 * (i * dp1 + j)
    return buf[slot + 1 : slot + 2], buf[slot + 2 : slot + 3]
