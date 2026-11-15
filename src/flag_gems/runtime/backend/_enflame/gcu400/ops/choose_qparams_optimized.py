# Copyright 2026, The FlagOS Contributors.
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
#
"""choose_qparams_optimized - fully fused single-launch Triton kernel.

Reference semantics (aten / flag_gems):
  x   = first `numel` contiguous float32 elements
  lo  = min(x), hi = max(x)
  step = (hi - lo) / n_bins,  qmax = 2**bit_width - 1
  err(lo_c, hi_c):  scale = half((hi_c - half(lo_c)) / qmax),  inv = 1/scale
                    q = clamp(nearbyint((x - half(lo_c)) * inv), 0, qmax)
                    err = sum((x - (q*scale + half(lo_c)))**2)
  The range is shrunk one step at a time from the low or the high side for as
  long as an improvement exists; the final state is returned as (hi, lo)
  1-element float32 tensors.

Everything runs inside one program: the input tile stays in registers, the first
three candidate losses are evaluated in a single pass, and the remaining search
steps are a data-dependent device-side while loop. No host sync, one launch.
`numel` is a constexpr so the tile masks and min/max selects constant-fold, and
all launch-time constants are memoized per request shape.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:  # portable libdevice shim (flag_gems uses the same import elsewhere)
    import triton.language.extra.libdevice as _libdevice
except ImportError:  # pragma: no cover - backend without a libdevice shim
    _libdevice = None

_MAX_RESIDENT = 8192


@triton.jit
def _rint(x):
    """Round half to even (std::nearbyint) built from floor only."""
    f = tl.floor(x)
    d = x - f
    is_odd = 2.0 * tl.floor(f * 0.5) != f
    up = (d > 0.5) | ((d == 0.5) & is_odd)
    return f + tl.where(up, 1.0, 0.0)


@triton.jit
def _rint_hw(x):
    """Same rounding via the backend's nearbyint (one instruction on GCU)."""
    return _libdevice.rint(x)


@triton.jit
def _half(v):
    return v.to(tl.float16).to(tl.float32)


@triton.jit
def _scale_inv(xmin_h, xmax, qmax):
    dr = xmax - xmin_h
    s = _half(dr / qmax)
    scale = tl.where(dr == 0.0, 1.0, s)
    inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
    return scale, inv


@triton.jit
def _err(x, mask, xmin_h, scale, inv, qmax, HW: tl.constexpr):
    z = (x - xmin_h) * inv
    if HW:
        q = _rint_hw(z)
    else:
        q = _rint(z)
    q = tl.minimum(tl.maximum(q, 0.0), qmax)
    d = x - (q * scale + xmin_h)
    return tl.sum(tl.where(mask, d * d, 0.0))


@triton.jit
def _fused_kernel(
    x_ptr,
    out_ptr,
    n_bins,
    min_bins,
    qmax,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
    HW: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < NUMEL
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    xmin = tl.min(tl.where(mask, x, float("inf")))
    xmax = tl.max(tl.where(mask, x, float("-inf")))

    stepsize = (xmax - xmin) / n_bins
    threshold = stepsize * min_bins

    xmin_h = _half(xmin)
    scale0, inv0 = _scale_inv(xmin_h, xmax, qmax)
    e0 = _err(x, mask, xmin_h, scale0, inv0, qmax, HW)

    # First search step: low-side shrink and high-side shrink.
    xmin1 = _half(xmin + stepsize)
    s1, i1 = _scale_inv(xmin1, xmax, qmax)
    e1 = _err(x, mask, xmin1, s1, i1, qmax, HW)

    s2, i2 = _scale_inv(xmin_h, xmax - stepsize, qmax)
    e2 = _err(x, mask, xmin_h, s2, i2, qmax, HW)

    go = (xmin + threshold) < xmax
    do = go & ((e0 >= e1) | (e0 >= e2))
    t1 = e1 < e2
    t2 = e2 <= e1

    cur_min = tl.where(do & t1, xmin + stepsize, xmin)
    cur_max = tl.where(do & t2, xmax - stepsize, xmax)
    cur_loss = tl.where(do, tl.where(t1, e1, e2), e0)

    cond = do & ((cur_min + threshold) < cur_max)

    while cond:
        xm1 = _half(cur_min + stepsize)
        sa, ia = _scale_inv(xm1, cur_max, qmax)
        la = _err(x, mask, xm1, sa, ia, qmax, HW)

        xm2 = _half(cur_min)
        sb, ib = _scale_inv(xm2, cur_max - stepsize, qmax)
        lb = _err(x, mask, xm2, sb, ib, qmax, HW)

        upd = (cur_loss >= la) | (cur_loss >= lb)
        a1 = la < lb
        a2 = lb <= la
        cur_min = tl.where(upd & a1, cur_min + stepsize, cur_min)
        cur_max = tl.where(upd & a2, cur_max - stepsize, cur_max)
        cur_loss = tl.where(upd, tl.where(a1, la, lb), cur_loss)
        cond = upd & ((cur_min + threshold) < cur_max)

    tl.store(out_ptr + 0, cur_max)
    tl.store(out_ptr + 1, cur_min)


# ---------------------------------------------------------------------------
# Fallback for numel > _MAX_RESIDENT: the reference's host-driven search loop.
# ---------------------------------------------------------------------------


@triton.jit
def _minmax_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    vmin = tl.full((BLOCK,), float("inf"), tl.float32)
    vmax = tl.full((BLOCK,), float("-inf"), tl.float32)
    for off in range(0, numel, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < numel
        v = tl.load(x_ptr + idx, mask=m, other=0.0)
        vmin = tl.minimum(vmin, tl.where(m, v, float("inf")))
        vmax = tl.maximum(vmax, tl.where(m, v, float("-inf")))
    tl.store(out_ptr + 0, tl.min(vmin))
    tl.store(out_ptr + 1, tl.max(vmax))


@triton.jit
def _qerror_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    xmin1,
    scale1,
    inv1,
    xmin2,
    scale2,
    inv2,
    qmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    q1 = _rint((x - xmin1) * inv1)
    q1 = tl.minimum(tl.maximum(q1, 0.0), qmax)
    d1 = x - (q1 * scale1 + xmin1)
    err1 = tl.sum(tl.where(mask, d1 * d1, 0.0))

    q2 = _rint((x - xmin2) * inv2)
    q2 = tl.minimum(tl.maximum(q2, 0.0), qmax)
    d2 = x - (q2 * scale2 + xmin2)
    err2 = tl.sum(tl.where(mask, d2 * d2, 0.0))

    tl.store(out_ptr + 2 * pid + 0, err1)
    tl.store(out_ptr + 2 * pid + 1, err2)


@triton.jit
def _reduce2_kernel(p_ptr, out_ptr, n_rows, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    m = r < n_rows
    a = tl.load(p_ptr + 2 * r, mask=m, other=0.0)
    b = tl.load(p_ptr + 2 * r + 1, mask=m, other=0.0)
    tl.store(out_ptr + 0, tl.sum(a))
    tl.store(out_ptr + 1, tl.sum(b))


@triton.jit
def _store2_kernel(a_ptr, b_ptr, va, vb):
    tl.store(a_ptr, va)
    tl.store(b_ptr, vb)


def _half_round(value):
    """float32 -> float16 -> float32, as at::Half does on the reference side."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _make_params(xmin, xmax, qmax):
    xmin_h = _half_round(xmin)
    data_range = xmax - xmin_h
    scale = 1.0 if data_range == 0.0 else _half_round(data_range / qmax)
    inv = 0.0 if scale == 0.0 else 1.0 / scale
    return xmin_h, scale, inv


def _run_fallback(x, numel, n_bins, ratio, bit_width, device):
    mm = torch.empty(2, dtype=torch.float32, device=device)
    _minmax_kernel[(1,)](x, mm, numel, BLOCK=1024)
    xmin, xmax = (float(v) for v in mm.tolist())

    stepsize = (xmax - xmin) / n_bins
    min_bins = int(n_bins * (1.0 - ratio))
    qmax = float((1 << bit_width) - 1)

    EB = 1024
    nblk = max(1, (numel + EB - 1) // EB)
    partial = torch.empty((nblk, 2), dtype=torch.float32, device=device)
    buf = torch.empty(2, dtype=torch.float32, device=device)
    RB = max(1, triton.next_power_of_2(nblk))

    def errors(p1, p2):
        _qerror_kernel[(nblk,)](
            x,
            partial,
            numel,
            p1[0],
            p1[1],
            p1[2],
            p2[0],
            p2[1],
            p2[2],
            qmax,
            BLOCK=EB,
        )
        _reduce2_kernel[(1,)](partial, buf, nblk, BLOCK=RB)
        vals = buf.tolist()
        return float(vals[0]), float(vals[1])

    cur_min, cur_max = xmin, xmax
    params0 = _make_params(cur_min, cur_max, qmax)
    cur_loss, _ = errors(params0, params0)

    threshold = min_bins * stepsize
    while cur_min + threshold < cur_max:
        params1 = _make_params(cur_min + stepsize, cur_max, qmax)
        params2 = _make_params(cur_min, cur_max - stepsize, qmax)
        loss1, loss2 = errors(params1, params2)

        if cur_loss < loss1 and cur_loss < loss2:
            break

        if loss1 < loss2:
            cur_min = cur_min + stepsize
            cur_loss = loss1
        else:
            cur_max = cur_max - stepsize
            cur_loss = loss2

    max_out = torch.empty(1, dtype=torch.float32, device=device)
    min_out = torch.empty(1, dtype=torch.float32, device=device)
    _store2_kernel[(1,)](max_out, min_out, float(cur_max), float(cur_min))
    return max_out, min_out


# Per-call plan cache: all launch-time constants are pure functions of the
# (numel, n_bins, bit_width, ratio) request, so they are computed once and then
# only looked up. Nothing data-dependent is cached.
_PLANS = {}


def _plan(numel, n_bins, bit_width, ratio):
    key = (numel, n_bins, bit_width, ratio)
    p = _PLANS.get(key)
    if p is None:
        block = triton.next_power_of_2(numel)
        if block < 64:
            block = 64
        # Single-program, latency-bound launch.  Measured on gcu:6 (paired
        # do_bench, same process): one warp is fastest for tiles of <=1024
        # elements (fewer cross-warp reduction steps per search step), while a
        # 4096-element tile needs 8 warps to keep each step off the critical
        # path (nw4 1.0x, nw8 ~1.1x faster at BLOCK=4096).
        if block <= 1024:
            nw = 1
        elif block <= 2048:
            nw = 4
        else:
            nw = 8
        # libdevice nearbyint replaces the hand-rolled floor-based rounding
        # only for the smallest tile: measured 17.9->16.5us at BLOCK=256 while
        # it is neutral/slower at 1024/4096 (where the reduction, not the
        # rounding sequence, is on the critical path).
        hw = block <= 256 and _libdevice is not None
        p = (
            block,
            nw,
            float(n_bins),
            int(n_bins * (1.0 - ratio)),
            float((1 << bit_width) - 1),
            hw,
        )
        _PLANS[key] = p
    return p


def run(input, numel, n_bins, ratio, bit_width):
    if input.dim() == 1 and input.is_contiguous():
        x = input
    else:
        x = input.contiguous().reshape(-1)

    numel = int(numel)
    n_bins = int(n_bins)
    bit_width = int(bit_width)
    ratio = float(ratio)

    if numel <= 0 or numel > _MAX_RESIDENT:
        return _run_fallback(x, numel, n_bins, ratio, bit_width, x.device)

    block, nw, nb_f, min_bins, qmax, hw = _plan(numel, n_bins, bit_width, ratio)
    out = torch.empty(2, dtype=torch.float32, device=x.device)
    _fused_kernel[(1,)](
        x,
        out,
        nb_f,
        min_bins,
        qmax,
        NUMEL=numel,
        BLOCK=block,
        HW=hw,
        num_warps=nw,
    )
    return out[0:1], out[1:2]


def choose_qparams_optimized(input, numel, n_bins, ratio, bit_width):
    logger.debug("GEMS CHOOSE_QPARAMS_OPTIMIZED")

    # The aten reference is float32-only; reject other dtypes explicitly.
    assert input.dtype == torch.float32, (
        "choose_qparams_optimized only supports float32 input, " f"got {input.dtype}"
    )
    return run(input, numel, n_bins, ratio, bit_width)


__all__ = ["choose_qparams_optimized", "run"]
