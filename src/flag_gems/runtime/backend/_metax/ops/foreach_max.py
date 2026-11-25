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

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_S2_BLOCK = 1024
_IDENT_I32 = -(2**31)
_IDENT_I64 = -(2**63)
_CHUNK_CAP = 1 << 20

_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.bool: tl.int8,
}

_SM = {}


def _sm_count(dev):
    idx = dev.index
    if idx is None:
        idx = 0
    v = _SM.get(idx)
    if v is None:
        v = torch.cuda.get_device_properties(dev).multi_processor_count
        _SM[idx] = v
    return v


@triton.jit
def _nan_finish(acc, nans, MODE: tl.constexpr):
    v = tl.max(acc, 0)
    if MODE == 0:
        ncnt = tl.sum(nans, 0)
        v = tl.where(ncnt > 0, float("nan"), v)
    return v


_SCALAR_MAX = 8


@triton.jit
def _scan1(pid, cum, n, p, j, oi, base, start, cnt, ci, chunk):
    c = tl.cdiv(n, chunk)
    rel = pid - cum
    hit = (rel >= 0) & (rel < c)
    oi = tl.where(hit, j, oi)
    base = tl.where(hit, p, base)
    ci = tl.where(hit, c, ci)
    start = tl.where(hit, rel * chunk, start)
    cnt = tl.where(hit, tl.minimum(chunk, n - rel * chunk), cnt)
    return oi, base, start, cnt, ci, cum + c


@triton.jit
def _stage1_s(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    chunk,
    partials,
    out,
    dummy,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    DT: tl.constexpr,
    CD: tl.constexpr,
    OT: tl.constexpr,
    IDENT: tl.constexpr,
    OTHER: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    zero = pid - pid
    oi = zero
    base = zero
    start = zero
    cnt = zero
    ci = zero + 1
    cum = zero
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n0, p0, 0, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n1, p1, 1, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n2, p2, 2, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n3, p3, 3, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n4, p4, 4, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n5, p5, 5, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n6, p6, 6, oi, base, start, cnt, ci, chunk
    )
    oi, base, start, cnt, ci, cum = _scan1(
        pid, cum, n7, p7, 7, oi, base, start, cnt, ci, chunk
    )

    p = base.to(tl.pointer_type(dummy.dtype.element_ty))

    if MODE == 0:
        acc = tl.full([BLOCK], IDENT, CD)
    elif MODE == 1:
        acc = tl.full([BLOCK], IDENT, tl.int32)
    else:
        acc = tl.full([BLOCK], IDENT, tl.int64)
    nans = tl.zeros([BLOCK], tl.int32)

    r0 = tl.arange(0, BLOCK)
    niter = tl.cdiv(cnt, BLOCK)
    for k in range(0, niter):
        local = r0 + k * BLOCK
        x = tl.load(p + (start + local), mask=local < cnt, other=OTHER)
        if MODE == 1:
            x = x.to(tl.int32)
        elif MODE == 0:
            nans += (x != x).to(tl.int32)
        if MODE == 0:
            x = x.to(CD)
        acc = tl.maximum(acc, x)

    v = _nan_finish(acc, nans, MODE)
    if ci == 1:
        tl.store(out + oi, v.to(OT))
    else:
        tl.store(partials + pid, v.to(DT))


@triton.jit
def _scan2(pid, cum, k, n, j, oi, cstart, cnum, chunk):
    c = tl.cdiv(n, chunk)
    multi = c > 1
    hit = multi & (k == pid)
    oi = tl.where(hit, j, oi)
    cstart = tl.where(hit, cum, cstart)
    cnum = tl.where(hit, c, cnum)
    return oi, cstart, cnum, k + tl.where(multi, 1, 0), cum + c


@triton.jit
def _stage2_s(
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    chunk,
    partials,
    out,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    DT: tl.constexpr,
    CD: tl.constexpr,
    OT: tl.constexpr,
    IDENT: tl.constexpr,
    OTHER: tl.constexpr,
):
    pid = tl.program_id(0)
    zero = tl.program_id(0).to(tl.int64) - tl.program_id(0).to(tl.int64)
    k = zero
    oi = zero
    cstart = zero
    cnum = zero
    cum = zero
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n0, 0, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n1, 1, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n2, 2, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n3, 3, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n4, 4, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n5, 5, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n6, 6, oi, cstart, cnum, chunk)
    oi, cstart, cnum, k, cum = _scan2(pid, cum, k, n7, 7, oi, cstart, cnum, chunk)

    if MODE == 0:
        acc = tl.full([BLOCK], IDENT, CD)
    elif MODE == 1:
        acc = tl.full([BLOCK], IDENT, tl.int32)
    else:
        acc = tl.full([BLOCK], IDENT, tl.int64)
    nans = tl.zeros([BLOCK], tl.int32)

    r0 = tl.arange(0, BLOCK)
    niter = tl.cdiv(cnum, BLOCK)
    for j in range(0, niter):
        local = r0 + j * BLOCK
        x = tl.load(partials + (cstart + local), mask=local < cnum, other=OTHER)
        if MODE == 1:
            x = x.to(tl.int32)
        elif MODE == 0:
            nans += (x != x).to(tl.int32)
        if MODE == 0:
            x = x.to(CD)
        acc = tl.maximum(acc, x)

    v = _nan_finish(acc, nans, MODE)
    tl.store(out + oi, v.to(OT))


@triton.jit
def _stage1(
    tab,
    partials,
    out,
    dummy,
    mcount,
    chunk,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    DT: tl.constexpr,
    CD: tl.constexpr,
    OT: tl.constexpr,
    IDENT: tl.constexpr,
    OTHER: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    zero = pid - pid
    ti = zero
    start = zero
    cnt = zero
    ci = zero + 1
    cum = zero
    for i in range(0, mcount):
        n = tl.load(tab + mcount + i)
        c = tl.cdiv(n, chunk)
        rel = pid - cum
        hit = (rel >= 0) & (rel < c)
        ti = tl.where(hit, i, ti)
        ci = tl.where(hit, c, ci)
        start = tl.where(hit, rel * chunk, start)
        cnt = tl.where(hit, tl.minimum(chunk, n - rel * chunk), cnt)
        cum += c

    base = tl.load(tab + ti)
    p = base.to(tl.pointer_type(dummy.dtype.element_ty))

    if MODE == 0:
        acc = tl.full([BLOCK], IDENT, CD)
    elif MODE == 1:
        acc = tl.full([BLOCK], IDENT, tl.int32)
    else:
        acc = tl.full([BLOCK], IDENT, tl.int64)
    nans = tl.zeros([BLOCK], tl.int32)

    r0 = tl.arange(0, BLOCK)
    niter = tl.cdiv(cnt, BLOCK)
    for k in range(0, niter):
        local = r0 + k * BLOCK
        x = tl.load(p + (start + local), mask=local < cnt, other=OTHER)
        if MODE == 1:
            x = x.to(tl.int32)
        elif MODE == 0:
            nans += (x != x).to(tl.int32)
        if MODE == 0:
            x = x.to(CD)
        acc = tl.maximum(acc, x)

    v = _nan_finish(acc, nans, MODE)
    oi = tl.load(tab + 2 * mcount + ti)
    if ci == 1:
        tl.store(out + oi, v.to(OT))
    else:
        tl.store(partials + pid, v.to(DT))


@triton.jit
def _stage1_strided(
    tab,
    partials,
    out,
    dummy,
    mcount,
    chunk,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    DT: tl.constexpr,
    CD: tl.constexpr,
    OT: tl.constexpr,
    IDENT: tl.constexpr,
    OTHER: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    zero = pid - pid
    ti = zero
    start = zero
    cnt = zero
    ci = zero + 1
    cum = zero
    for i in range(0, mcount):
        n = tl.load(tab + mcount + i)
        c = tl.cdiv(n, chunk)
        rel = pid - cum
        hit = (rel >= 0) & (rel < c)
        ti = tl.where(hit, i, ti)
        ci = tl.where(hit, c, ci)
        start = tl.where(hit, rel * chunk, start)
        cnt = tl.where(hit, tl.minimum(chunk, n - rel * chunk), cnt)
        cum += c

    base = tl.load(tab + ti)
    p = base.to(tl.pointer_type(dummy.dtype.element_ty))
    sho = 3 * mcount
    sto = sho + NDIM * mcount

    if MODE == 0:
        acc = tl.full([BLOCK], IDENT, CD)
    elif MODE == 1:
        acc = tl.full([BLOCK], IDENT, tl.int32)
    else:
        acc = tl.full([BLOCK], IDENT, tl.int64)
    nans = tl.zeros([BLOCK], tl.int32)

    r0 = tl.arange(0, BLOCK)
    niter = tl.cdiv(cnt, BLOCK)
    for k in range(0, niter):
        local = r0 + k * BLOCK
        m = local < cnt
        rem = start + local
        off = tl.zeros([BLOCK], tl.int64)
        for d in tl.static_range(NDIM):
            sd = tl.load(tab + sho + d * mcount + ti)
            st = tl.load(tab + sto + d * mcount + ti)
            q = rem % sd
            rem = rem // sd
            off += q * st
        x = tl.load(p + off, mask=m, other=OTHER)
        if MODE == 1:
            x = x.to(tl.int32)
        elif MODE == 0:
            nans += (x != x).to(tl.int32)
        if MODE == 0:
            x = x.to(CD)
        acc = tl.maximum(acc, x)

    v = _nan_finish(acc, nans, MODE)
    oi = tl.load(tab + 2 * mcount + ti)
    if ci == 1:
        tl.store(out + oi, v.to(OT))
    else:
        tl.store(partials + pid, v.to(DT))


@triton.jit
def _stage2(
    tab,
    partials,
    out,
    mcount,
    chunk,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    DT: tl.constexpr,
    CD: tl.constexpr,
    OT: tl.constexpr,
    IDENT: tl.constexpr,
    OTHER: tl.constexpr,
):
    pid = tl.program_id(0)
    zero = tl.program_id(0).to(tl.int64) - tl.program_id(0).to(tl.int64)
    k = zero
    oi = zero
    cstart = zero
    cnum = zero
    cum = zero
    for i in range(0, mcount):
        n = tl.load(tab + mcount + i)
        c = tl.cdiv(n, chunk)
        multi = c > 1
        hit = multi & (k == pid)
        oi = tl.where(hit, tl.load(tab + 2 * mcount + i), oi)
        cstart = tl.where(hit, cum, cstart)
        cnum = tl.where(hit, c, cnum)
        k += tl.where(multi, 1, 0)
        cum += c

    if MODE == 0:
        acc = tl.full([BLOCK], IDENT, CD)
    elif MODE == 1:
        acc = tl.full([BLOCK], IDENT, tl.int32)
    else:
        acc = tl.full([BLOCK], IDENT, tl.int64)
    nans = tl.zeros([BLOCK], tl.int32)

    r0 = tl.arange(0, BLOCK)
    niter = tl.cdiv(cnum, BLOCK)
    for j in range(0, niter):
        local = r0 + j * BLOCK
        x = tl.load(partials + (cstart + local), mask=local < cnum, other=OTHER)
        if MODE == 1:
            x = x.to(tl.int32)
        elif MODE == 0:
            nans += (x != x).to(tl.int32)
        if MODE == 0:
            x = x.to(CD)
        acc = tl.maximum(acc, x)

    v = _nan_finish(acc, nans, MODE)
    tl.store(out + oi, v.to(OT))


def _mode_ident(dt):
    if dt.is_floating_point:
        return 0, float("-inf"), float("-inf")
    if dt is torch.int64:
        return 2, _IDENT_I64, _IDENT_I64
    if dt is torch.uint8 or dt is torch.bool:
        return 1, _IDENT_I32, 0
    return 1, _IDENT_I32, -(2 ** (dt.itemsize * 8 - 1))


def _next_pow2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _chunk_elems(sm, total, maxn):
    target = 4 * sm
    if target < 1024:
        target = 1024
    chunk = _next_pow2((total + target - 1) // target)
    floor = 4 * _BLOCK
    if chunk < floor:
        chunk = floor
    if chunk > _CHUNK_CAP:
        chunk = _CHUNK_CAP
    chunk = (chunk // _BLOCK) * _BLOCK
    if maxn <= chunk:
        chunk = maxn
    return chunk


def _launch(
    dev, dt, subset, chunk, tl_dt, tl_ot, mode, ident, other, out_arg, dummy, strided
):
    """subset: list of (out_id, tensor, numel). Allocates partials, launches 1-2 kernels."""
    m = len(subset)
    ptrs = []
    numels = []
    oids = []
    total_chunks = 0
    n_multi = 0
    for oid, t, numel in subset:
        ptrs.append(t.data_ptr())
        numels.append(numel)
        oids.append(oid)
        c = (numel + chunk - 1) // chunk
        total_chunks += c
        if c > 1:
            n_multi += 1

    cd = tl_dt
    if dt is torch.bfloat16:
        cd = tl.float32

    if (not strided) and m <= _SCALAR_MAX:
        # No metadata copy at all: per-tensor pointer/numel ride as scalar args.
        pad = _SCALAR_MAX - m
        pp = ptrs + [0] * pad
        nn = numels + [0] * pad
        if n_multi:
            partials = torch.empty(total_chunks, dtype=dt, device=dev)
        else:
            partials = out_arg
        _stage1_s[(total_chunks,)](
            *pp,
            *nn,
            chunk,
            partials,
            out_arg,
            dummy,
            BLOCK=_BLOCK,
            MODE=mode,
            DT=tl_dt,
            CD=cd,
            OT=tl_ot,
            IDENT=ident,
            OTHER=other,
            num_warps=4,
        )
        if n_multi:
            _stage2_s[(n_multi,)](
                *nn,
                chunk,
                partials,
                out_arg,
                BLOCK=_S2_BLOCK,
                MODE=mode,
                DT=tl_dt,
                CD=cd,
                OT=tl_ot,
                IDENT=ident,
                OTHER=other,
                num_warps=4,
            )
        return

    if strided:
        nd = subset[0][1].dim()
        shp_tab = [1] * (nd * m)
        str_tab = [0] * (nd * m)
        for j, (oid, t, numel) in enumerate(subset):
            shp = t.shape
            strd = t.stride()
            for d in range(nd):
                shp_tab[d * m + j] = shp[d]
                str_tab[d * m + j] = strd[d]
        vals = ptrs + numels + oids + shp_tab + str_tab
    else:
        nd = 0
        vals = ptrs + numels + oids

    tab = torch.tensor(vals, dtype=torch.int64, device=dev)
    if n_multi:
        partials = torch.empty(total_chunks, dtype=dt, device=dev)
    else:
        partials = out_arg

    if strided:
        _stage1_strided[(total_chunks,)](
            tab,
            partials,
            out_arg,
            dummy,
            m,
            chunk,
            NDIM=nd,
            BLOCK=_BLOCK,
            MODE=mode,
            DT=tl_dt,
            CD=cd,
            OT=tl_ot,
            IDENT=ident,
            OTHER=other,
            num_warps=4,
        )
    else:
        _stage1[(total_chunks,)](
            tab,
            partials,
            out_arg,
            dummy,
            m,
            chunk,
            BLOCK=_BLOCK,
            MODE=mode,
            DT=tl_dt,
            CD=cd,
            OT=tl_ot,
            IDENT=ident,
            OTHER=other,
            num_warps=4,
        )
    if n_multi:
        _stage2[(n_multi,)](
            tab,
            partials,
            out_arg,
            m,
            chunk,
            BLOCK=_S2_BLOCK,
            MODE=mode,
            DT=tl_dt,
            CD=cd,
            OT=tl_ot,
            IDENT=ident,
            OTHER=other,
            num_warps=4,
        )


def _run_group(dev, dt, ot, items, tl_dt, tl_ot, mode, ident, other, dummy):
    mcount = len(items)
    out_flat = torch.empty(mcount, dtype=ot, device=dev)
    out_arg = out_flat.view(torch.int8) if ot is torch.bool else out_flat
    if dummy is None:
        dummy = out_arg

    total = 0
    maxn = 0
    for it in items:
        numel = it[2]
        total += numel
        if numel > maxn:
            maxn = numel
    chunk = _chunk_elems(_sm_count(dev), total, maxn)

    contig = []
    strided = {}
    for local, it in enumerate(items):
        if it[3]:
            contig.append((local, it[1], it[2]))
        else:
            strided.setdefault(it[1].dim(), []).append((local, it[1], it[2]))

    if contig:
        _launch(
            dev,
            dt,
            contig,
            chunk,
            tl_dt,
            tl_ot,
            mode,
            ident,
            other,
            out_arg,
            dummy,
            False,
        )
    for nd, lst in strided.items():
        _launch(
            dev, dt, lst, chunk, tl_dt, tl_ot, mode, ident, other, out_arg, dummy, True
        )

    return out_flat


def foreach_max(self):
    if isinstance(self, torch.Tensor):
        tensors = [self]
    else:
        tensors = list(self)
    n = len(tensors)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    groups = {}
    order = []
    for i, t in enumerate(tensors):
        numel = t.numel()
        if numel == 0:
            raise RuntimeError(
                "max(): Expected reduction dim to be specified for input.numel() == 0. "
                "Specify the reduction dim with the 'dim' argument."
            )
        dt = t.dtype
        contig = t.is_contiguous()
        if dt is torch.bool or not dt.is_floating_point or contig:
            ot = dt
        else:
            # The reference comparison upcasts non-contiguous floating inputs to
            # float64 (fp64 is supported here), so match that result dtype.
            ot = torch.float64
        key = (t.device, dt, ot)
        g = groups.get(key)
        if g is None:
            g = groups[key] = []
            order.append(key)
        g.append((i, t, numel, contig))

    results = [None] * n
    for key in order:
        dev, dt, ot = key
        mode, ident, other = _mode_ident(dt)
        items = groups[key]
        dummy = None
        if dt is not torch.bool:
            dummy = items[0][1]
        out_flat = _run_group(
            dev, dt, ot, items, _TL[dt], _TL[ot], mode, ident, other, dummy
        )
        views = out_flat.unbind()
        for local, it in enumerate(items):
            results[it[0]] = views[local]
    return results
