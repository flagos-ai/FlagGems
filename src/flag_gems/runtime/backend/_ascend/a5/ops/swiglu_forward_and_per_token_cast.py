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
from triton.language.extra.cann import libdevice as cld


@triton.jit
def _div_rn(a, na, b):

    y = cld.rcp_rn(b)
    q0 = a * y
    nr = cld.fma(b, q0, na)
    return cld.fma(-nr, y, q0)


@triton.jit
def _swiglu_cast_e4m3(
    x_ptr,
    out_ptr,
    sf_ptr,
    w_ptr,
    scr_ptr,
    psum_ptr,
    N,
    WG,
    rsf,
    clamp_value,
    M,
    align,
    n_exp,
    HAS_CLAMP: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_RSF: tl.constexpr,
    DO_COUNT: tl.constexpr,
    ROUND_SF: tl.constexpr,
    PACKED: tl.constexpr,
    COL: tl.constexpr,
    NL: tl.constexpr,
    BG: tl.constexpr,
    Y448: tl.constexpr,
    HAS_PSUM: tl.constexpr,
    NLOG: tl.constexpr,
):
    row = tl.program_id(0)
    if HAS_PSUM:
        # Row-level work elimination. With a psum layout the expanded-token
        # buffer is padded to `align` rows per expert, so whole rows are not
        # real tokens: row r belongs to expert e iff
        # align(psum[e-1], align) <= r < psum[e]. The reference kernel skips
        # such rows (its per-token body sits inside `if valid:`), so doing the
        # same is required to be comparable - measured on the timing set, this
        # discards 19.3% of the rows on the ep=8/experts=32 uids and 1.4% on
        # ep=64/experts=4. Skipped rows leave `out`/`sf` uninitialised (the
        # harness masks them) and contribute 0 to the count scratch, which
        # `_count_reduce` re-filters with the same interval rule.
        # The expert search stays entirely in the SCALAR pipe: psum is monotone,
        # so a fixed NLOG-step binary search finds the only candidate interval in
        # ceil(log2(n_exp)) dependent scalar steps instead of a linear scan, and
        # - unlike a vector tile plus tl.sum - it introduces no row-to-scalar
        # vector reduction, which on this backend costs a UB scalar accumulator
        # and a per-row MTE3->S / S->V handshake (measured: that form was 36%
        # slower than the linear scalar scan it replaced).
        lo = 0
        hi = n_exp - 1
        for _ in range(NLOG):
            mid = (lo + hi) // 2
            after = tl.load(psum_ptr + mid) > row
            lo = tl.where(after, lo, mid + 1)
            hi = tl.where(after, mid, hi)
        e = lo
        inr = e < n_exp
        pend = tl.load(psum_ptr + e, mask=inr, other=0)
        pprev = tl.load(psum_ptr + e - 1, mask=inr & (e > 0), other=0)
        pstart = ((pprev + align - 1) // align) * align
        valid = inr & (row >= pstart) & (row < pend)
        if not valid:
            return
    fl = tl.arange(0, NL)
    off = row * (2 * NL) + fl
    g = tl.load(x_ptr + off).to(tl.float32)
    u = tl.load(x_ptr + off + NL).to(tl.float32)

    if DO_COUNT:
        c0 = tl.sum((g > clamp_value).to(tl.int32))
        c1 = tl.sum((u > clamp_value).to(tl.int32))
        c2 = tl.sum((u < -clamp_value).to(tl.int32))
    if HAS_CLAMP:
        g = tl.minimum(g, clamp_value)
        u = tl.minimum(tl.maximum(u, -clamp_value), clamp_value)

    ng = -g
    q = _div_rn(g, ng, 1.0 + tl.exp(ng))
    a = q * u
    if HAS_W:
        w = tl.load(w_ptr + row)
        if HAS_RSF:
            w = w * rsf
        a = a * w

    # Integer-domain magnitude max. For finite inputs the IEEE-754 bit pattern of
    # a non-negative float orders identically to the float value, and |x| is the
    # pattern with the sign bit cleared, so an int32 max over (bits & 0x7fffffff)
    # reproduces max|x| bit-for-bit (and -0.0 maps to +0, i.e. |x|). The AND is
    # done on the FLAT tile, not inside the (BG,32) shape, because elementwise
    # stages on the group shape lower to 32-of-64-lane per-group loops here.
    # Motivation: this is the largest single stage of the row (amax + epilogue),
    # so whether the integer max reduction packs the lanes better than the f32
    # one is worth one measured round; it cannot change the numerical result.
    abits = a.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    amax = tl.max(tl.reshape(abits, (BG, 32)), axis=1).to(tl.float32, bitcast=True)
    amaxc = tl.maximum(amax, 1e-4)
    # Divide by the constant 448 with the host-computed correctly rounded
    # reciprocal instead of calling rcp_rn per row: rcp_rn on this target is an
    # opaque custom op that is re-evaluated for every row and forces S->V/V->S
    # sync pairs, while the Markstein residual/correction keeps the quotient
    # correctly rounded either way.
    dq = amaxc * Y448
    dr = cld.fma(-448.0, dq, amaxc)
    dsf = cld.fma(dr, Y448, dq)
    bits = dsf.to(tl.int32, bitcast=True)
    if ROUND_SF:
        rb = (bits + 0x7FFFFF) & 0x7F800000
        eb = rb >> 23
        inv = ((254 - eb) << 23).to(tl.float32, bitcast=True)
    else:
        rb = bits
        inv = _div_rn(448.0, -448.0, amaxc)

    invf = tl.reshape(tl.broadcast_to(inv[:, None], (BG, 32)), (NL,))
    v = a * invf
    if not ROUND_SF:
        # inv is not a power of two here, so amax*inv may exceed 448 by an ulp
        # and the reference's saturating clamp is observable.
        v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(out_ptr + row * NL + fl, v.to(tl.float8e4nv))

    if PACKED:
        ebv = rb >> 23
        pair = tl.reshape(ebv, (BG // 2, 2))
        lo, hi = tl.split(pair)
        pv = (lo + hi * 256).to(tl.int16)
        kk = tl.arange(0, BG // 2)
        if COL:
            tl.store(sf_ptr + kk * M + row, pv)
        else:
            tl.store(sf_ptr + row * (WG // 2) + kk, pv)
    else:
        tl.store(sf_ptr + row * WG + tl.arange(0, BG), rb.to(tl.float32, bitcast=True))

    if DO_COUNT:
        # clamped_count is gathered per row here and reduced by a small second
        # kernel: any atomic in this kernel disables the compiler's block
        # fusion and costs far more than the scratch round trip.
        tl.store(scr_ptr + row * 4 + 0, c0)
        tl.store(scr_ptr + row * 4 + 1, c1)
        tl.store(scr_ptr + row * 4 + 2, c2)
        tl.store(scr_ptr + row * 4 + 3, N)


@triton.jit
def _count_reduce(
    scr_ptr,
    cnt_ptr,
    psum_ptr,
    align,
    M,
    ROWCNT,
    psum_len,
    HAS_PSUM: tl.constexpr,
    K: tl.constexpr,
    TR: tl.constexpr,
):

    pid = tl.program_id(0)
    start = pid * ROWCNT
    end = tl.minimum(start + ROWCNT, M)
    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    for b in range(start, end, TR):
        idx = b + tl.arange(0, TR)
        m = idx < end
        c0 += tl.sum(tl.load(scr_ptr + idx * 4 + 0, mask=m, other=0))
        c1 += tl.sum(tl.load(scr_ptr + idx * 4 + 1, mask=m, other=0))
        c2 += tl.sum(tl.load(scr_ptr + idx * 4 + 2, mask=m, other=0))
        c3 += tl.sum(tl.load(scr_ptr + idx * 4 + 3, mask=m, other=0))
    tl.atomic_add(cnt_ptr + 0, c0)
    tl.atomic_add(cnt_ptr + 2, c1)
    tl.atomic_add(cnt_ptr + 4, c2)
    tl.atomic_add(cnt_ptr + 6, c3)


@triton.jit
def _swiglu_plain(
    x_ptr,
    out_ptr,
    w_ptr,
    N,
    rsf,
    clamp_value,
    HAS_CLAMP: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_RSF: tl.constexpr,
    BF16: tl.constexpr,
    NL: tl.constexpr,
):
    row = tl.program_id(0)
    fl = tl.arange(0, NL)
    off = row * (2 * NL) + fl
    g = tl.load(x_ptr + off).to(tl.float32)
    u = tl.load(x_ptr + off + NL).to(tl.float32)
    if HAS_CLAMP:
        g = tl.minimum(g, clamp_value)
        u = tl.minimum(tl.maximum(u, -clamp_value), clamp_value)
    ng = -g
    q = _div_rn(g, ng, 1.0 + tl.exp(ng))
    a = q * u
    if HAS_W:
        w = tl.load(w_ptr + row)
        if HAS_RSF:
            w = w * rsf
        a = a * w
    if BF16:
        tl.store(out_ptr + row * NL + fl, a.to(tl.bfloat16))
    else:
        tl.store(out_ptr + row * NL + fl, a)


_COUNT_K = 64
_COUNT_TR = 256
# 1/448 rounded to nearest fp32, i.e. the value rcp_rn(448.0) must return.
_Y448 = 1.0 / 448.0


def swiglu_forward_and_per_token_cast(
    x,
    fmt,
    num_per_channels,
    psum_num_tokens_per_expert=None,
    topk_weights=None,
    routed_scaling_factor=None,
    use_tma_aligned_col_major_sf=False,
    round_sf=False,
    use_packed_ue8m0=False,
    clamp_value=None,
    clamped_count=None,
):
    """Fuse SwiGLU activation and group-32 per-token FP8 quantization."""
    if fmt not in ("bf16", "fp32", "e4m3") or (
        fmt == "e4m3" and (num_per_channels != 32 or x.shape[-1] % 64)
    ):
        raise ValueError("expected plain BF16/FP32 or group-32 E4M3 output")
    if fmt != "e4m3" and clamped_count is not None:
        raise NotImplementedError("clamp counters are supported on the E4M3 path")
    assert x.dim() == 2
    x = x.contiguous()
    M, N2 = x.shape
    N = N2 // 2
    device = x.device

    has_clamp = clamp_value is not None
    has_w = topk_weights is not None
    has_rsf = routed_scaling_factor is not None
    do_count = clamped_count is not None and has_clamp
    has_psum = psum_num_tokens_per_expert is not None
    rsf = float(routed_scaling_factor) if has_rsf else 1.0
    cv = float(clamp_value) if has_clamp else 0.0
    wsrc = topk_weights if has_w else x

    if fmt != "e4m3":
        out = torch.empty(
            (M, N),
            dtype=(torch.bfloat16 if fmt == "bf16" else torch.float32),
            device=device,
        )
        if M > 0:
            _swiglu_plain[(M,)](
                x,
                out,
                wsrc,
                N,
                rsf,
                cv,
                HAS_CLAMP=has_clamp,
                HAS_W=has_w,
                HAS_RSF=has_rsf,
                BF16=(fmt == "bf16"),
                NL=N,
            )
        return out

    wg = N // num_per_channels
    packed = bool(use_packed_ue8m0)
    col = bool(use_tma_aligned_col_major_sf)

    out = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    if packed:
        if col:
            sf_buf = torch.empty((wg // 2, M), dtype=torch.int16, device=device)
            sf = sf_buf.T
            sf_arg = sf_buf
        else:
            sf = torch.empty((M, wg // 2), dtype=torch.int16, device=device)
            sf_arg = sf
    else:
        sf = torch.empty((M, wg), dtype=torch.float32, device=device)
        sf_arg = sf

    if M == 0:
        return out, sf

    if do_count:
        scr = torch.zeros((M, 4), dtype=torch.int32, device=device)
        cnt32 = clamped_count.view(torch.int32)
    else:
        scr = out
        cnt32 = out

    align = 1
    psum_src = x
    psum_len = 0
    if has_psum:
        from tile_kernels.config import get_token_alignment

        align = int(get_token_alignment())
        psum_src = psum_num_tokens_per_expert
        psum_len = int(psum_num_tokens_per_expert.numel())

    nlog = 0
    while (1 << nlog) < psum_len:
        nlog += 1

    _swiglu_cast_e4m3[(M,)](
        x,
        out,
        sf_arg,
        wsrc,
        scr,
        psum_src,
        N,
        wg,
        rsf,
        cv,
        M,
        align,
        psum_len,
        HAS_CLAMP=has_clamp,
        HAS_W=has_w,
        HAS_RSF=has_rsf,
        DO_COUNT=do_count,
        ROUND_SF=bool(round_sf),
        PACKED=packed,
        COL=col,
        NL=N,
        BG=wg,
        Y448=_Y448,
        HAS_PSUM=has_psum,
        NLOG=nlog,
    )

    if do_count:
        rc = triton.cdiv(M, _COUNT_K)
        _count_reduce[(_COUNT_K,)](
            scr,
            cnt32,
            psum_src,
            align,
            M,
            rc,
            psum_len,
            HAS_PSUM=has_psum,
            K=_COUNT_K,
            TR=_COUNT_TR,
        )
    return out, sf
