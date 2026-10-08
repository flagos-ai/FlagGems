import torch
import triton
import triton.language as tl

BLOCK = 1024
RED_BLOCK = 2048
CHUNK = 16384


@triton.jit
def _round_even(v):
    # round-half-to-even (libdevice.rint semantics)
    r = tl.floor(v)
    f = v - r
    odd = (r - 2.0 * tl.floor(r * 0.5)) == 1.0
    up = (f > 0.5) | ((f == 0.5) & odd)
    return tl.where(up, r + 1.0, r)


@triton.jit
def _reduce_minmax(x_ptr, cmin_ptr, cmax_ptr, R, BLOCK: tl.constexpr):
    # grid = (ceil(R / BLOCK), C); atomic per-channel min/max over the R dim.
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.program_id(1)
    m = j < R
    x = tl.load(x_ptr + c * R + j, mask=m, other=0.0).to(tl.float32)
    xmin = tl.where(m, x, float("inf"))
    xmax = tl.where(m, x, float("-inf"))
    lmin = tl.min(xmin, axis=0)
    lmax = tl.max(xmax, axis=0)
    tl.atomic_min(cmin_ptr + c, lmin)
    tl.atomic_max(cmax_ptr + c, lmax)


@triton.jit
def _pt_partial_minmax(
    x_ptr, min_ptr, max_ptr, n_elements, CHUNK: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * CHUNK
    vmin = tl.full([BLOCK], float("inf"), tl.float32)
    vmax = tl.full([BLOCK], float("-inf"), tl.float32)
    for i in range(0, CHUNK, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        m = offs < n_elements
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        vmin = tl.minimum(vmin, tl.where(m, x, float("inf")))
        vmax = tl.maximum(vmax, tl.where(m, x, float("-inf")))
    tl.store(min_ptr + pid, tl.min(vmin, axis=0))
    tl.store(max_ptr + pid, tl.max(vmax, axis=0))


@triton.jit
def _pt_finalize_qparams(
    min_ptr,
    max_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    n_part,
    avg_c,
    qmin,
    qmax,
    OBS: tl.constexpr,
    SYM: tl.constexpr,
    FQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    acc_min = tl.full([BLOCK], float("inf"), tl.float32)
    acc_max = tl.full([BLOCK], float("-inf"), tl.float32)
    for i in range(0, n_part, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        m = offs < n_part
        acc_min = tl.minimum(
            acc_min, tl.load(min_ptr + offs, mask=m, other=float("inf"))
        )
        acc_max = tl.maximum(
            acc_max, tl.load(max_ptr + offs, mask=m, other=float("-inf"))
        )
    rmin = tl.load(rmin_ptr).to(tl.float32)
    rmax = tl.load(rmax_ptr).to(tl.float32)
    if OBS:
        cmin = tl.min(acc_min, axis=0)
        cmax = tl.max(acc_max, axis=0)
        a64 = avg_c.to(tl.float64)
        rmin = (rmin.to(tl.float64) * (1.0 - a64) + cmin.to(tl.float64) * a64).to(
            tl.float32
        )
        rmax = (rmax.to(tl.float64) * (1.0 - a64) + cmax.to(tl.float64) * a64).to(
            tl.float32
        )
        tl.store(rmin_ptr, rmin)
        tl.store(rmax_ptr, rmax)
    if FQ:
        qmnf = qmin.to(tl.float32)
        qmxf = qmax.to(tl.float32)
        mn = tl.minimum(rmin, 0.0)
        mx = tl.maximum(rmax, 0.0)
        sc = (mx - mn) / (qmxf - qmnf)
        sc = tl.where(sc == 0.0, 0.1, sc)
        z = _round_even(qmnf - mn / sc)
        z = tl.minimum(tl.maximum(z, qmnf), qmxf)
        if SYM:
            both = (mn < 0.0) & (mx > 0.0)
            sc_sym = tl.maximum((-mn) / (-qmnf), mx / qmxf)
            sc_sym = tl.where(sc_sym == 0.0, 0.1, sc_sym)
            z_sym = _round_even((qmnf + qmxf) * 0.5)
            sc = tl.where(both, sc_sym, sc)
            z = tl.where(both, z_sym, z)
        tl.store(scale_ptr, sc)
        tl.store(zp_ptr, z.to(tl.int32))


@triton.jit
def _qparams_pc(
    cmin_ptr,
    cmax_ptr,
    rmin_ptr,
    rmax_ptr,
    rmin_out_ptr,
    rmax_out_ptr,
    scale_ptr,
    zp_ptr,
    C,
    avg_c,
    qmin,
    qmax,
    OBS: tl.constexpr,
    SYM: tl.constexpr,
    FQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < C
    rmin = tl.load(rmin_ptr + off, mask=m, other=0.0).to(tl.float32)
    rmax = tl.load(rmax_ptr + off, mask=m, other=0.0).to(tl.float32)
    if OBS:
        cmin = tl.load(cmin_ptr + off, mask=m, other=0.0).to(tl.float32)
        cmax = tl.load(cmax_ptr + off, mask=m, other=0.0).to(tl.float32)
        a64 = avg_c.to(tl.float64)
        rmin = (rmin.to(tl.float64) * (1.0 - a64) + cmin.to(tl.float64) * a64).to(
            tl.float32
        )
        rmax = (rmax.to(tl.float64) * (1.0 - a64) + cmax.to(tl.float64) * a64).to(
            tl.float32
        )
        tl.store(rmin_out_ptr + off, rmin, mask=m)
        tl.store(rmax_out_ptr + off, rmax, mask=m)
    if FQ:
        qmnf = qmin.to(tl.float32)
        qmxf = qmax.to(tl.float32)
        mn = tl.minimum(rmin, 0.0)
        mx = tl.maximum(rmax, 0.0)
        # per-channel reference computes scale in fp64 then rounds to fp32
        mn64 = mn.to(tl.float64)
        mx64 = mx.to(tl.float64)
        qmn64 = qmnf.to(tl.float64)
        qmx64 = qmxf.to(tl.float64)
        sc = ((mx64 - mn64) / (qmx64 - qmn64)).to(tl.float32)
        sc = tl.where(sc == 0.0, 0.1, sc)
        z = _round_even(qmnf - mn / sc)
        z = tl.minimum(tl.maximum(z, qmnf), qmxf)
        if SYM:
            both = (mn < 0.0) & (mx > 0.0)
            sc_sym = tl.maximum((-mn64) / (-qmn64), mx64 / qmx64).to(tl.float32)
            sc_sym = tl.where(sc_sym == 0.0, 0.1, sc_sym)
            z_sym = _round_even((qmnf + qmxf) * 0.5)
            sc = tl.where(both, sc_sym, sc)
            z = tl.where(both, z_sym, z)
        tl.store(scale_ptr + off, sc, mask=m)
        tl.store(zp_ptr + off, z.to(tl.int32), mask=m)


@triton.jit
def _fq_pt(
    x_ptr, out_ptr, mask_ptr, scale_ptr, zp_ptr, N, qmin, qmax, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    q = _round_even(x / s) + z
    valid = (q >= qmnf) & (q <= qmxf)
    qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
    out = (qc - z) * s
    tl.store(out_ptr + off, out, mask=m)
    tl.store(mask_ptr + off, valid, mask=m)


@triton.jit
def _fq_pc(
    x_ptr, out_ptr, mask_ptr, scale_ptr, zp_ptr, N, R, qmin, qmax, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    c = off // R
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + c, mask=m, other=1.0).to(tl.float32)
    z = tl.load(zp_ptr + c, mask=m, other=0).to(tl.float32)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    q = _round_even(x * (1.0 / s)) + z
    valid = (q >= qmnf) & (q <= qmxf)
    qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
    out = (qc - z) * s
    tl.store(out_ptr + off, out, mask=m)
    tl.store(mask_ptr + off, valid, mask=m)


@triton.jit
def _fused_small_pc(
    x_ptr,
    out_ptr,
    mask_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    N,
    C,
    R,
    avg_c,
    qmin,
    qmax,
    OBS: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single-block fused path for small per-channel tensors: per-channel min/max,
    # observer update, qparams, and fake-quant in ONE launch. No cmin/cmax
    # scratch: each channel's qparams are computed right after its reduction.
    offs_r = tl.arange(0, BLOCK_R)
    offs = tl.arange(0, BLOCK)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    qmn64 = qmnf.to(tl.float64)
    qmx64 = qmxf.to(tl.float64)
    a64 = avg_c.to(tl.float64)
    for cc in range(0, C):
        lmin = tl.full([BLOCK_R], float("inf"), tl.float32)
        lmax = tl.full([BLOCK_R], float("-inf"), tl.float32)
        for i in range(0, R, BLOCK_R):
            o = cc * R + i + offs_r
            m = (i + offs_r) < R
            x = tl.load(x_ptr + o, mask=m, other=0.0)
            lmin = tl.minimum(lmin, tl.where(m, x, float("inf")))
            lmax = tl.maximum(lmax, tl.where(m, x, float("-inf")))
        cmin_s = tl.min(lmin, axis=0)
        cmax_s = tl.max(lmax, axis=0)
        rmin = tl.load(rmin_ptr + cc).to(tl.float32)
        rmax = tl.load(rmax_ptr + cc).to(tl.float32)
        if OBS:
            rmin = (rmin.to(tl.float64) * (1.0 - a64) + cmin_s.to(tl.float64) * a64).to(
                tl.float32
            )
            rmax = (rmax.to(tl.float64) * (1.0 - a64) + cmax_s.to(tl.float64) * a64).to(
                tl.float32
            )
            tl.store(rmin_ptr + cc, rmin)
            tl.store(rmax_ptr + cc, rmax)
        mn = tl.minimum(rmin, 0.0)
        mx = tl.maximum(rmax, 0.0)
        sc = ((mx.to(tl.float64) - mn.to(tl.float64)) / (qmx64 - qmn64)).to(tl.float32)
        sc = tl.where(sc == 0.0, 0.1, sc)
        z = _round_even(qmnf - mn / sc)
        z = tl.minimum(tl.maximum(z, qmnf), qmxf)
        if SYM:
            both = (mn < 0.0) & (mx > 0.0)
            sc_sym = tl.maximum(
                (-mn.to(tl.float64)) / (-qmn64), mx.to(tl.float64) / qmx64
            ).to(tl.float32)
            sc_sym = tl.where(sc_sym == 0.0, 0.1, sc_sym)
            z_sym = _round_even((qmnf + qmxf) * 0.5)
            sc = tl.where(both, sc_sym, sc)
            z = tl.where(both, z_sym, z)
        tl.store(scale_ptr + cc, sc)
        tl.store(zp_ptr + cc, z.to(tl.int32))
    tl.debug_barrier()
    # Fake-quant phase (scale/zp now visible block-wide)
    for base in range(0, N, BLOCK):
        o = base + offs
        m = o < N
        chan = o // R
        x = tl.load(x_ptr + o, mask=m, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + chan, mask=m, other=1.0).to(tl.float32)
        zv = tl.load(zp_ptr + chan, mask=m, other=0).to(tl.float32)
        q = _round_even(x * (1.0 / s)) + zv
        valid = (q >= qmnf) & (q <= qmxf)
        qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
        outv = (qc - zv) * s
        tl.store(out_ptr + o, outv, mask=m)
        tl.store(mask_ptr + o, valid, mask=m)


def _as_bool(v):
    if isinstance(v, torch.Tensor):
        return bool(v.item())
    return bool(v)


def run(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    x = self
    if x.dtype is not torch.float32:
        name = {
            torch.float16: "Half",
            torch.bfloat16: "BFloat16",
            torch.float64: "Double",
        }.get(x.dtype, str(x.dtype))
        raise RuntimeError(f"expected scalar type Float but found {name}")
    obs = _as_bool(observer_on)
    fq = _as_bool(fake_quant_on)
    sym = bool(symmetric_quant)
    pc = bool(per_row_fake_quant)
    qmin = int(quant_min)
    qmax = int(quant_max)
    ac = float(averaging_const)

    N = x.numel()
    xf = x.contiguous().view(-1)
    dev = x.device

    C = x.shape[int(ch_axis)] if pc else 1
    R = N // C

    if not fq:
        if obs:
            if pc:
                cmin = torch.full((C,), float("inf"), dtype=torch.float32, device=dev)
                cmax = torch.full((C,), float("-inf"), dtype=torch.float32, device=dev)
                _reduce_minmax[(triton.cdiv(R, RED_BLOCK), C)](
                    xf, cmin, cmax, R, BLOCK=RED_BLOCK
                )
                _qparams_pc[(triton.cdiv(C, 128),)](
                    cmin,
                    cmax,
                    running_min,
                    running_max,
                    running_min,
                    running_max,
                    scale,
                    zero_point,
                    C,
                    ac,
                    qmin,
                    qmax,
                    OBS=True,
                    SYM=sym,
                    FQ=False,
                    BLOCK=128,
                )
            else:
                n_part = triton.cdiv(N, CHUNK)
                mn = torch.empty(n_part, dtype=torch.float32, device=dev)
                mx = torch.empty(n_part, dtype=torch.float32, device=dev)
                _pt_partial_minmax[(n_part,)](
                    xf, mn, mx, N, CHUNK=CHUNK, BLOCK=RED_BLOCK
                )
                _pt_finalize_qparams[(1,)](
                    mn,
                    mx,
                    running_min,
                    running_max,
                    scale,
                    zero_point,
                    n_part,
                    ac,
                    qmin,
                    qmax,
                    OBS=True,
                    SYM=sym,
                    FQ=False,
                    BLOCK=RED_BLOCK,
                )
        mask = torch.ones(x.shape, dtype=torch.bool, device=dev)
        return (x, mask)

    # fake-quant on
    out = torch.empty_like(xf, dtype=torch.float32)
    mask = torch.empty(N, dtype=torch.bool, device=dev)

    # Small per-channel fused path: single kernel does minmax + qparams + fq.
    if obs and pc and fq and N <= 131072 and C <= 1024:
        blk_r = min(1024, triton.next_power_of_2(max(R, 64)))
        _fused_small_pc[(1,)](
            xf,
            out,
            mask,
            running_min,
            running_max,
            scale,
            zero_point,
            N,
            C,
            R,
            ac,
            qmin,
            qmax,
            OBS=True,
            SYM=sym,
            BLOCK_R=blk_r,
            BLOCK=1024,
        )
        return (out.view(x.shape), mask.view(x.shape))

    if obs:
        if pc:
            cmin = torch.full((C,), float("inf"), dtype=torch.float32, device=dev)
            cmax = torch.full((C,), float("-inf"), dtype=torch.float32, device=dev)
            _reduce_minmax[(triton.cdiv(R, RED_BLOCK), C)](
                xf, cmin, cmax, R, BLOCK=RED_BLOCK
            )
            _qparams_pc[(triton.cdiv(C, 128),)](
                cmin,
                cmax,
                running_min,
                running_max,
                running_min,
                running_max,
                scale,
                zero_point,
                C,
                ac,
                qmin,
                qmax,
                OBS=True,
                SYM=sym,
                FQ=True,
                BLOCK=128,
            )
        else:
            n_part = triton.cdiv(N, CHUNK)
            mn = torch.empty(n_part, dtype=torch.float32, device=dev)
            mx = torch.empty(n_part, dtype=torch.float32, device=dev)
            _pt_partial_minmax[(n_part,)](xf, mn, mx, N, CHUNK=CHUNK, BLOCK=RED_BLOCK)
            _pt_finalize_qparams[(1,)](
                mn,
                mx,
                running_min,
                running_max,
                scale,
                zero_point,
                n_part,
                ac,
                qmin,
                qmax,
                OBS=True,
                SYM=sym,
                FQ=True,
                BLOCK=RED_BLOCK,
            )
    else:
        if pc:
            _qparams_pc[(triton.cdiv(C, 128),)](
                running_min,
                running_max,
                running_min,
                running_max,
                running_min,
                running_max,
                scale,
                zero_point,
                C,
                ac,
                qmin,
                qmax,
                OBS=False,
                SYM=sym,
                FQ=True,
                BLOCK=128,
            )
        else:
            _pt_finalize_qparams[(1,)](
                xf,
                xf,
                running_min,
                running_max,
                scale,
                zero_point,
                0,
                ac,
                qmin,
                qmax,
                OBS=False,
                SYM=sym,
                FQ=True,
                BLOCK=RED_BLOCK,
            )

    grid = (triton.cdiv(N, BLOCK),)
    if pc:
        _fq_pc[grid](xf, out, mask, scale, zero_point, N, R, qmin, qmax, BLOCK=BLOCK)
    else:
        _fq_pt[grid](xf, out, mask, scale, zero_point, N, qmin, qmax, BLOCK=BLOCK)

    return (out.view(x.shape), mask.view(x.shape))


# Alias for FlagGems import convention
fused_moving_avg_obs_fq_helper = run
