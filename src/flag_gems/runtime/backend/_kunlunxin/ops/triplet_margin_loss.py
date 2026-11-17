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
"""Kunlunxin XPU3 override for ``triplet_margin_loss``.

The generic kernels reduce *inside* a runtime loop: ``_row_parts`` (sum / max /
min over ``[BLOCK_C]``) is called from ``_tm_rows_kernel``'s ``for i in
range(0, RPP)`` and ``_tm_cols_kernel``'s ``for r in range(0, R)``.  On XPU3 a
loop-carried ``tt.reduce`` never legalises:

    failed to legalize operation 'tt.reduce' that was explicitly marked illegal
    Pipeline failed while executing [`ConvertTritonXPUToLLVM`]

Every one of the 685 ``triplet_margin_loss`` test cases hits that error.

This override keeps the *exact* generic mathematics -- ``d_p(x, y) = norm(x - y
+ eps, p, last dim)``, ``swap`` / ``margin`` / ``none|mean|sum`` -- but reshapes
the work into the forms already proven on XPU3 (see the wiki entry
``pattern-xpu3-loop-carried-reduce-vector-accumulate`` and
``multi_margin_loss``):

* **one program per (row, distance-pair), pure 1-D tile** -- no 2-D tensor tiles
  (they silently mis-store rows on this backend) and no runtime-loop-carried
  reduction;
* **exactly one reduction per kernel, outside any loop** -- the three Lp
  distances (a-p, a-n, p-n) are computed by three program instances of a single
  kernel; when a row is wider than one tile the blocks are folded with an
  element-wise vector accumulator and reduced once at the end;
* the reduction kind is selected by a ``MODE`` constexpr, so a compiled kernel
  instance only ever contains ``tl.sum`` *or* ``tl.max`` *or* ``tl.min`` (never
  the three ``tt.reduce`` kinds in one kernel that the generic row-parts helper
  produced);
* every masked load clamps its addresses; the combine/reduce kernel is a single
  program with one vector load and at most one ``tl.sum``.

fp64 inputs are computed in fp32 and the result widened back (the Triton XPU
backend cannot take fp64 pointer arguments), matching the generic behaviour.
"""

import logging
import sys

import torch
import triton
import triton.language as tl

from flag_gems.ops.triplet_margin_loss import _lead_groups, _MAX_LEAD, _mode_of
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_SINGLE_TILE = 8192
_LOOP_BLOCK = 128


@triton.jit(do_not_specialize=["R", "C", "eps", "pp"])
def _dist_kernel(
    APtr,
    PPtr,
    NPtr,
    Out,
    R,
    C,
    eps,
    pp,
    g0s,
    g1s,
    g2s,
    g3s,
    g0a,
    g1a,
    g2a,
    g3a,
    g0p,
    g1p,
    g2p,
    g3p,
    g0n,
    g1n,
    g2n,
    g3n,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (pair, row); ``row`` fits one tile. One reduce total."""
    pair = tl.program_id(0)
    row = tl.program_id(1)

    rem = row
    off_a = 0
    off_p = 0
    off_n = 0
    idx = rem % g0s
    rem = rem // g0s
    off_a += idx * g0a
    off_p += idx * g0p
    off_n += idx * g0n
    idx = rem % g1s
    rem = rem // g1s
    off_a += idx * g1a
    off_p += idx * g1p
    off_n += idx * g1n
    idx = rem % g2s
    rem = rem // g2s
    off_a += idx * g2a
    off_p += idx * g2p
    off_n += idx * g2n
    idx = rem % g3s
    rem = rem // g3s
    off_a += idx * g3a
    off_p += idx * g3p
    off_n += idx * g3n

    cols = tl.arange(0, BLOCK)
    m = cols < C
    safe = tl.minimum(cols, C - 1)
    av = tl.load(APtr + off_a + safe, mask=m, other=0.0).to(tl.float32)
    pv = tl.load(PPtr + off_p + safe, mask=m, other=0.0).to(tl.float32)
    nv = tl.load(NPtr + off_n + safe, mask=m, other=0.0).to(tl.float32)
    is0 = pair == 0
    is1 = pair == 1
    d = tl.where(is0, av - pv, tl.where(is1, av - nv, pv - nv)) + eps
    ad = tl.abs(d)

    if MODE == 2:
        part = tl.sum(tl.where(m, d * d, 0.0), axis=0)
    elif MODE == 1:
        part = tl.sum(tl.where(m, ad, 0.0), axis=0)
    elif MODE == 0:
        part = tl.sum(tl.where(m & (d != 0.0), 1.0, 0.0), axis=0)
    elif MODE == 5:
        part = tl.sum(tl.where(m, tl.exp2(pp * tl.log2(ad)), 0.0), axis=0)
    elif MODE == 3:
        part = tl.max(tl.where(m, ad, 0.0), axis=0)
    else:  # MODE == 4 (-inf norm)
        part = tl.min(tl.where(m, ad, float("inf")), axis=0)

    if MODE == 2:
        dist = tl.sqrt(part)
    elif MODE == 5:
        dist = tl.exp2((1.0 / pp) * tl.log2(part))
    else:
        dist = part
    tl.store(Out + pair * R + row, dist)


@triton.jit(do_not_specialize=["R", "C", "eps", "pp"])
def _dist_loop_kernel(
    APtr,
    PPtr,
    NPtr,
    Out,
    R,
    C,
    eps,
    pp,
    g0s,
    g1s,
    g2s,
    g3s,
    g0a,
    g1a,
    g2a,
    g3a,
    g0p,
    g1p,
    g2p,
    g3p,
    g0n,
    g1n,
    g2n,
    g3n,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Wide rows: fold C blocks into a vector accumulator, reduce once."""
    pair = tl.program_id(0)
    row = tl.program_id(1)

    rem = row
    off_a = 0
    off_p = 0
    off_n = 0
    idx = rem % g0s
    rem = rem // g0s
    off_a += idx * g0a
    off_p += idx * g0p
    off_n += idx * g0n
    idx = rem % g1s
    rem = rem // g1s
    off_a += idx * g1a
    off_p += idx * g1p
    off_n += idx * g1n
    idx = rem % g2s
    rem = rem // g2s
    off_a += idx * g2a
    off_p += idx * g2p
    off_n += idx * g2n
    idx = rem % g3s
    rem = rem // g3s
    off_a += idx * g3a
    off_p += idx * g3p
    off_n += idx * g3n

    cols = tl.arange(0, BLOCK)
    if MODE == 4:
        acc = tl.full((BLOCK,), float("inf"), tl.float32)
    else:
        acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, C, BLOCK):
        cc = start + cols
        m = cc < C
        safe = tl.minimum(cc, C - 1)
        av = tl.load(APtr + off_a + safe, mask=m, other=0.0).to(tl.float32)
        pv = tl.load(PPtr + off_p + safe, mask=m, other=0.0).to(tl.float32)
        nv = tl.load(NPtr + off_n + safe, mask=m, other=0.0).to(tl.float32)
        is0 = pair == 0
        is1 = pair == 1
        d = tl.where(is0, av - pv, tl.where(is1, av - nv, pv - nv)) + eps
        ad = tl.abs(d)
        if MODE == 2:
            acc += tl.where(m, d * d, 0.0)
        elif MODE == 1:
            acc += tl.where(m, ad, 0.0)
        elif MODE == 0:
            acc += tl.where(m & (d != 0.0), 1.0, 0.0)
        elif MODE == 5:
            acc += tl.where(m, tl.exp2(pp * tl.log2(ad)), 0.0)
        elif MODE == 3:
            acc = tl.maximum(acc, tl.where(m, ad, 0.0))
        else:  # MODE == 4
            acc = tl.minimum(acc, tl.where(m, ad, float("inf")))

    if MODE == 3:
        part = tl.max(acc, axis=0)
    elif MODE == 4:
        part = tl.min(acc, axis=0)
    else:
        part = tl.sum(acc, axis=0)

    if MODE == 2:
        dist = tl.sqrt(part)
    elif MODE == 5:
        dist = tl.exp2((1.0 / pp) * tl.log2(part))
    else:
        dist = part
    tl.store(Out + pair * R + row, dist)


@triton.jit(do_not_specialize=["R", "margin", "scale"])
def _combine_kernel(
    Dist,
    Out,
    R,
    margin,
    scale,
    SWAP: tl.constexpr,
    WRITE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program: combine the three distances; at most one tl.sum."""
    offs = tl.arange(0, BLOCK)
    m = offs < R
    safe = tl.minimum(offs, R - 1)
    d_ap = tl.load(Dist + safe, mask=m, other=0.0)
    d_an = tl.load(Dist + R + safe, mask=m, other=0.0)
    d_pn = tl.load(Dist + 2 * R + safe, mask=m, other=0.0)
    if SWAP:
        d_an = tl.minimum(d_an, d_pn)
    loss = tl.maximum(margin + d_ap - d_an, 0.0)
    if WRITE_ROWS:
        tl.store(Out + offs, loss.to(Out.dtype.element_ty), mask=m)
    else:
        total = tl.sum(tl.where(m, loss, 0.0), axis=0)
        tl.store(Out, (total * scale).to(Out.dtype.element_ty))


@triton.jit(do_not_specialize=["R", "margin", "scale"])
def _combine_loop_kernel(
    Dist,
    Out,
    R,
    margin,
    scale,
    SWAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Many rows: vector accumulator, one reduce outside the loop."""
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, R, BLOCK):
        offs = start + cols
        m = offs < R
        safe = tl.minimum(offs, R - 1)
        d_ap = tl.load(Dist + safe, mask=m, other=0.0)
        d_an = tl.load(Dist + R + safe, mask=m, other=0.0)
        d_pn = tl.load(Dist + 2 * R + safe, mask=m, other=0.0)
        if SWAP:
            d_an = tl.minimum(d_an, d_pn)
        loss = tl.maximum(margin + d_ap - d_an, 0.0)
        acc += tl.where(m, loss, 0.0)
    total = tl.sum(acc, axis=0)
    tl.store(Out, (total * scale).to(Out.dtype.element_ty))


@triton.jit(do_not_specialize=["n", "value"])
def _fill_kernel(Out, n, value, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    for start in range(0, n, BLOCK):
        offs = start + cols
        m = offs < n
        tl.store(
            Out + tl.minimum(offs, n - 1),
            tl.full((BLOCK,), value, Out.dtype.element_ty),
            mask=m,
        )


def _fill(out, value):
    n = out.numel()
    if n == 0:
        return
    block = min(_MAX_SINGLE_TILE, triton.next_power_of_2(n))
    with torch_device_fn.device(out.device):
        _fill_kernel[(1,)](out, n, float(value), BLOCK=block)


def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-6,
    swap=False,
    reduction="mean",
):
    logger.debug("GEMS_KUNLUNXIN TRIPLET_MARGIN_LOSS")

    if isinstance(reduction, str):
        red = {"none": 0, "mean": 1, "sum": 2}[reduction.lower()]
    elif reduction is None:
        red = 0
    else:
        red = int(reduction)
    if red not in (0, 1, 2):
        raise ValueError("triplet_margin_loss: invalid reduction")

    shape = torch.broadcast_shapes(
        tuple(anchor.shape), tuple(positive.shape), tuple(negative.shape)
    )
    out_dtype = torch.promote_types(
        torch.promote_types(anchor.dtype, positive.dtype), negative.dtype
    )
    a = anchor if tuple(anchor.shape) == shape else anchor.expand(shape)
    b = positive if tuple(positive.shape) == shape else positive.expand(shape)
    c = negative if tuple(negative.shape) == shape else negative.expand(shape)
    dev = anchor.device
    # Triton XPU cannot take fp64 pointer arguments: compute in fp32, widen back.
    res_dtype = out_dtype
    if out_dtype == torch.float64:
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        c = c.to(torch.float32)
        out_dtype = torch.float32

    ndim = len(shape)
    if ndim == 0:
        C = 1
        R = 1
        out_shape = ()
    else:
        C = shape[-1]
        R = 1
        for s in shape[:-1]:
            R *= s
        out_shape = tuple(shape[:-1])

    pval = float(p)
    mode = _mode_of(pval)
    swap_i = 1 if swap else 0
    scale = (1.0 / R) if red == 1 else 1.0

    def _ret(t):
        return t.to(res_dtype) if res_dtype != out_dtype else t

    if red == 0:
        out = torch.empty(out_shape, dtype=out_dtype, device=dev)
    else:
        out = torch.empty((), dtype=out_dtype, device=dev)
    if R <= 0:
        return _ret(out)

    # Empty feature dimension: every Lp distance is 0, loss = max(margin, 0).
    if C == 0:
        value = max(float(margin), 0.0)
        if red == 0:
            _fill(out.reshape(-1), value)
        elif red == 1:
            _fill(out, value)
        else:
            _fill(out, value * R)
        return _ret(out)

    groups = _lead_groups([a, b, c], shape) if ndim >= 1 else []
    if len(groups) > _MAX_LEAD:
        raise RuntimeError(
            "triplet_margin_loss: unsupported stride pattern "
            f"({len(groups)} leading stride groups)"
        )
    gs = [1] * _MAX_LEAD
    ga = [0] * _MAX_LEAD
    gp = [0] * _MAX_LEAD
    gn = [0] * _MAX_LEAD
    for k, (size, strides) in enumerate(groups):
        gs[k] = int(size)
        ga[k] = int(strides[0])
        gp[k] = int(strides[1])
        gn[k] = int(strides[2])
    gargs = [
        gs[0],
        gs[1],
        gs[2],
        gs[3],
        ga[0],
        ga[1],
        ga[2],
        ga[3],
        gp[0],
        gp[1],
        gp[2],
        gp[3],
        gn[0],
        gn[1],
        gn[2],
        gn[3],
    ]

    dist = torch.empty((3 * R,), dtype=torch.float32, device=dev)
    with torch_device_fn.device(dev):
        if C <= _MAX_SINGLE_TILE:
            _dist_kernel[(3, R)](
                a,
                b,
                c,
                dist,
                R,
                C,
                float(eps),
                pval,
                *gargs,
                MODE=mode,
                BLOCK=triton.next_power_of_2(C),
            )
        else:
            _dist_loop_kernel[(3, R)](
                a,
                b,
                c,
                dist,
                R,
                C,
                float(eps),
                pval,
                *gargs,
                MODE=mode,
                BLOCK=_LOOP_BLOCK,
            )

        if red == 0:
            rows = out.reshape(-1)
            _combine_kernel[(1,)](
                dist,
                rows,
                R,
                float(margin),
                1.0,
                SWAP=swap_i != 0,
                WRITE_ROWS=True,
                BLOCK=triton.next_power_of_2(R)
                if R <= _MAX_SINGLE_TILE
                else _LOOP_BLOCK,
            )
        elif R <= _MAX_SINGLE_TILE:
            _combine_kernel[(1,)](
                dist,
                out,
                R,
                float(margin),
                scale,
                SWAP=swap_i != 0,
                WRITE_ROWS=False,
                BLOCK=triton.next_power_of_2(R),
            )
        else:
            _combine_loop_kernel[(1,)](
                dist,
                out,
                R,
                float(margin),
                scale,
                SWAP=swap_i != 0,
                BLOCK=_LOOP_BLOCK,
            )
    return _ret(out)


# The marker test binds ``flag_gems.triplet_margin_loss`` (the package attribute,
# which the SpecOpRegistrar already patches from ``_kunlunxin.ops``); rebind the
# generic submodule attributes too so any ``from flag_gems.ops.triplet_margin_loss
# import triplet_margin_loss`` path also resolves here (precedent:
# ``_kunlunxin/ops/_batch_norm_impl_index.py``).
for _mod_name in ("flag_gems.ops", "flag_gems.ops.triplet_margin_loss"):
    _mod = sys.modules.get(_mod_name)
    if _mod is not None:
        setattr(_mod, "triplet_margin_loss", triplet_margin_loss)
