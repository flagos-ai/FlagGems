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
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _pth_power(x, keep, POW, PK: tl.constexpr):
    """|x| ** POW for one tile.  ``keep`` is the (integer) load mask.

    PK is a compile-time tag for the exactly representable integer powers so
    the hot cases need neither log/exp nor any compare at all.
    """
    if PK == 2:
        return x * x
    elif PK == 1:
        return tl.abs(x)
    elif PK == 3:
        ax = tl.abs(x)
        return ax * ax * ax
    else:
        # 1e-30 is a floor, not a compare: log() must never see 0.0 because
        # exp(-inf) is unreliable here.  POW * log(1e-30) <= -69 underflows
        # exp() to 0.0 for every POW >= 1; the mask select (integer predicate,
        # unlike the generic implementation's float `ax > 0.0`) zeroes the
        # padding lanes for the remaining POW values.
        ax = tl.abs(x)
        t = tl.exp(POW * tl.log(tl.maximum(ax, 1e-30)))
        return tl.where(keep, t, 0.0)


@triton.jit
def _finalize(s, INV_POW, PK: tl.constexpr):
    if PK == 2:
        return tl.sqrt(s)
    elif PK == 1:
        return s
    else:
        # Clamp instead of comparing (see _pth_power).  The floor must stay a
        # *normal* f32: with a subnormal floor (1e-38) the same expression stops
        # linking entirely -- `undefined symbol: log2/exp2`
        # (harness/run_logs/norm_except_dim/probe_explog3.py: w_max38 FAIL vs
        # w_max30 OK).  Consequence: an all-zero reduction returns
        # (1e-30) ** (1/p) (1e-10 for p == 3) instead of exactly 0.0, which is
        # far below the fp32 atol of 1e-4.
        return tl.exp(INV_POW * tl.log(tl.maximum(s, 1e-30)))


@triton.jit
def _nxd_fused_inner_kernel(
    v_ptr,
    out_ptr,
    inner,
    POW,
    INV_POW,
    PK: tl.constexpr,
    FIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # d == 0 and a single reduction chunk: one program per output element walks
    # its whole contiguous block, so nothing is shared between programs.
    i = tl.program_id(0)
    base = i * inner
    acc = tl.zeros([BLOCK], tl.float32)
    for q0 in range(0, inner, BLOCK):
        q = q0 + tl.arange(0, BLOCK)
        mask = q < inner
        x = tl.load(v_ptr + base + q, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        acc += _pth_power(x, mask, POW, PK)
    s = tl.sum(acc, axis=0)
    if FIN:
        tl.store(out_ptr + i, _finalize(s, INV_POW, PK))
    else:
        tl.store(out_ptr + i, s)


@triton.jit
def _nxd_inner_partial_kernel(
    v_ptr,
    part_ptr,
    D,
    inner,
    chunk,
    POW,
    PK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # d == 0, split reduction.  Program (i, c) owns partial slot [c, i]; no two
    # programs touch the same address, so no atomics are needed.
    i = tl.program_id(0)
    c = tl.program_id(1)
    start = c * chunk
    end = tl.minimum(start + chunk, inner)
    base = i * inner
    acc = tl.zeros([BLOCK], tl.float32)
    for q0 in range(start, end, BLOCK):
        q = q0 + tl.arange(0, BLOCK)
        mask = q < end
        x = tl.load(v_ptr + base + q, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        acc += _pth_power(x, mask, POW, PK)
    tl.store(part_ptr + c * D + i, tl.sum(acc, axis=0))


@triton.jit
def _nxd_fused_outer_kernel(
    v_ptr,
    out_ptr,
    pre,
    D,
    POW,
    INV_POW,
    PK: tl.constexpr,
    BC: tl.constexpr,
):
    # d == ndim - 1 and a single row chunk: one program per column block walks
    # every row itself.
    cb = tl.program_id(0)
    cols = cb * BC + tl.arange(0, BC)
    cmask = cols < D
    acc = tl.zeros([BC], tl.float32)
    for p in range(pre):
        x = tl.load(v_ptr + p * D + cols, mask=cmask, other=0.0).to(tl.float32)
        x = tl.where(cmask, x, 0.0)
        acc += _pth_power(x, cmask, POW, PK)
    tl.store(out_ptr + cols, _finalize(acc, INV_POW, PK), mask=cmask)


@triton.jit
def _nxd_outer_partial_kernel(
    v_ptr,
    part_ptr,
    pre,
    D,
    rchunk,
    POW,
    PK: tl.constexpr,
    BC: tl.constexpr,
):
    # d == ndim - 1, split over rows.  Program (cb, rc) owns partial slots
    # [rc, cb * BC ... ) exclusively.
    cb = tl.program_id(0)
    rc = tl.program_id(1)
    cols = cb * BC + tl.arange(0, BC)
    cmask = cols < D
    rstart = rc * rchunk
    rend = tl.minimum(rstart + rchunk, pre)
    acc = tl.zeros([BC], tl.float32)
    for p in range(rstart, rend):
        x = tl.load(v_ptr + p * D + cols, mask=cmask, other=0.0).to(tl.float32)
        x = tl.where(cmask, x, 0.0)
        acc += _pth_power(x, cmask, POW, PK)
    tl.store(part_ptr + rc * D + cols, acc, mask=cmask)


@triton.jit
def _nxd_combine_kernel(
    part_ptr,
    out_ptr,
    D,
    nparts,
    POW,
    INV_POW,
    PK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Stage 2 of the split paths: sum the private partials of every output
    # element (one program per output block) and finalize.  Replaces the
    # generic implementation's atomic accumulation + separate finalize launch.
    # BLOCK must obey the <= 64 / == 1024 dynamic-loop tile rule; the caller
    # passes 64 (1024 OORs here, unlike in the reduction kernels above).
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < D
    acc = tl.zeros([BLOCK], tl.float32)
    for c in range(nparts):
        s = tl.load(part_ptr + c * D + off, mask=mask, other=0.0)
        acc += tl.where(mask, s, 0.0)
    tl.store(out_ptr + off, _finalize(acc, INV_POW, PK), mask=mask)


def _pow_kind(pval):
    # Exactly representable integer powers get a branch-free body; anything
    # else falls back to the exp/log form (still compare-free).
    if pval == 1.0:
        return 1
    if pval == 2.0:
        return 2
    if pval == 3.0:
        return 3
    return 0


def norm_except_dim(v, pow=2, dim=0):
    logger.debug("GEMS_KUNLUNXIN NORM_EXCEPT_DIM")
    logging.getLogger("flag_gems.ops.norm_except_dim").debug("GEMS NORM_EXCEPT_DIM")

    assert v.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"norm_except_dim: unsupported dtype {v.dtype}"

    v = v.contiguous()
    # aten treats dim == -1 as a sentinel meaning "norm over the whole tensor"
    # (0-dim scalar output), NOT "the last dim". Only the literal -1 is special;
    # other negative dims index from the end as usual.
    full_norm = int(dim) == -1
    if full_norm:
        v = v.reshape(1, -1)
    ndim = v.dim()
    d = 0 if full_norm else int(dim) % ndim
    numel = v.numel()
    D = v.size(d)
    inner = 1
    for k in range(d + 1, ndim):
        inner *= v.size(k)
    pre = numel // (D * inner) if D * inner > 0 else 0

    out_shape = [1] * ndim
    out_shape[d] = D

    pval = float(pow)
    pk = _pow_kind(pval)
    inv_pow = 1.0 / pval if pval != 0.0 else 0.0

    if numel == 0:
        acc = torch.zeros(out_shape, device=v.device, dtype=v.dtype)
        return acc.reshape(()) if full_norm else acc

    out = torch.empty(out_shape, device=v.device, dtype=v.dtype)

    # BLOCK 1024 reduction tile width / BC 64 column-block width / TARGET 4096
    # parallelism target.  BC is *not* the generic implementation's 256: a 1D
    # tile inside a dynamic-trip-count loop only compiles at <= 64 or exactly
    # 1024 on this backend; 256 dies in `TritonXPUUnrollControl` /
    # `out of resource: uni_sram` (reproduced with BC=256 on shape (1, 2),
    # dim=1 -- a geometry the generic file's own `_fused_outer_kernel` never
    # reaches from tests/).
    BLOCK = 1024
    BC = 64
    # Stage-2 tile width; also inside a dynamic loop, so also <= 64.
    BCOMB = 64
    TARGET = 4096

    # p in {1, 2} finalizes with identity / sqrt, which are fine on a scalar,
    # so the single-pass kernels can finish in place.  Any other p needs
    # log/exp on a tile, so those kernels dump raw fp32 sums and
    # _nxd_combine_kernel takes the root.
    fin = pk in (1, 2)

    if d == 0:
        # pre == 1, contiguous inner blocks.
        nchunks = max(1, min((inner + BLOCK - 1) // BLOCK, (TARGET + D - 1) // D))
        if nchunks == 1 and fin:
            _nxd_fused_inner_kernel[(D,)](v, out, inner, pval, inv_pow, pk, True, BLOCK)
        else:
            if nchunks == 1:
                part = torch.empty((1, D), device=v.device, dtype=torch.float32)
                _nxd_fused_inner_kernel[(D,)](
                    v, part, inner, pval, inv_pow, pk, False, BLOCK
                )
                nparts = 1
            else:
                chunk = (inner + nchunks - 1) // nchunks
                part = torch.empty((nchunks, D), device=v.device, dtype=torch.float32)
                _nxd_inner_partial_kernel[(D, nchunks)](
                    v, part, D, inner, chunk, pval, pk, BLOCK
                )
                nparts = nchunks
            _nxd_combine_kernel[((D + BCOMB - 1) // BCOMB,)](
                part, out, D, nparts, pval, inv_pow, pk, BCOMB
            )
    elif d == ndim - 1:
        # Thresholds pre >= 256 and colblocks < 64 gate splitting the row
        # reduction so wide/short tensors still saturate the device; 2048 caps
        # the row-chunk count.
        colblocks = (D + BC - 1) // BC
        if pre >= 256 and colblocks < 64:
            nrowchunks = max(1, min(pre, max(1, 2048 // colblocks)))
        else:
            nrowchunks = 1
        if nrowchunks == 1:
            # Reduces into a [BC] vector -> can always finalize inline.
            _nxd_fused_outer_kernel[(colblocks,)](v, out, pre, D, pval, inv_pow, pk, BC)
        else:
            rchunk = (pre + nrowchunks - 1) // nrowchunks
            part = torch.empty((nrowchunks, D), device=v.device, dtype=torch.float32)
            _nxd_outer_partial_kernel[(colblocks, nrowchunks)](
                v, part, pre, D, rchunk, pval, pk, BC
            )
            _nxd_combine_kernel[((D + BCOMB - 1) // BCOMB,)](
                part, out, D, nrowchunks, pval, inv_pow, pk, BCOMB
            )
    else:
        # Middle kept dim, viewed as (pre, D, inner).  The generic file does
        # this with one nested-dynamic-loop kernel; that structure does not
        # compile here (`TritonXPUUnrollControl` failure / `tt.addptr` shape
        # error on the 4-D shapes), so it is factored into the two single-loop
        # kernels that the other paths already use:
        #   stage 1: row-wise p-th power sums over `inner`, i.e. exactly the
        #            d == 0 geometry on a (pre * D, inner) view -> part[pre, D]
        #   stage 2: column-wise sum of those partials over `pre` + root,
        #            i.e. _nxd_combine_kernel with nparts = pre.
        part = torch.empty((pre, D), device=v.device, dtype=torch.float32)
        _nxd_fused_inner_kernel[(pre * D,)](
            v, part, inner, pval, inv_pow, pk, False, BLOCK
        )
        _nxd_combine_kernel[((D + BCOMB - 1) // BCOMB,)](
            part, out, D, pre, pval, inv_pow, pk, BCOMB
        )

    # dim == -1 collapses to a 0-dim scalar (matches aten's whole-tensor norm).
    return out.reshape(()) if full_norm else out
