import math

import torch
import triton
import triton.language as tl

# Dense-score formulation of sparse attention with a sink term.
#
# The reference semantics are
#
#   s[b,m,h,t] = scale * <q[b,m,h,:], kv[b, idx[b,m,t], :]>
#   p = softmax(cat([s, sink], -1))
#   out[b,m,h,:] = sum_t p[t] * kv[b, idx[b,m,t], :]
#
# Instead of gathering the top-k keys we build a padded copy of the whole KV
# cache (JP = kv_len rounded up to a multiple of the tile), score every key
# with a dot, exponentiate, and weight each key by its multiplicity among the
# top-k indices.  Padding columns and keys that were never selected have
# multiplicity 0, so they do not contribute to either the numerator or the
# denominator.  The denominator is computed from the same exponentiated scores
# plus exp(sink - G), so the result is mathematically identical to the
# gather-based softmax while every kernel is a plain affine/dot kernel.

# Largest number of heads handled by one dense score/output tile.  Wider head
# counts are processed in chunks of this size (the per-chunk computation is
# identical, so the result is unchanged).  HP=128 does not compile for the
# output kernel on this backend (fp32 accumulator SRAM footprint), so 64 is
# the safe upper bound.
_MAX_HEAD_TILE = 64


@triton.jit
def _pack_kv_kernel(SRC, DST, n_elems, SRC_STRIDE, DST_STRIDE, BLK: tl.constexpr):
    i_b = tl.program_id(0)
    i_c = tl.program_id(1)
    offs = i_c * BLK + tl.arange(0, BLK)
    v = tl.load(SRC + i_b * SRC_STRIDE + offs, mask=offs < n_elems, other=0.0)
    tl.store(DST + i_b * DST_STRIDE + offs, v)


@triton.jit
def _pack_q_kernel(SRC, DST, n_elems, SRC_STRIDE, DST_STRIDE, BLK: tl.constexpr):
    i_bm = tl.program_id(0)
    i_c = tl.program_id(1)
    offs = i_c * BLK + tl.arange(0, BLK)
    v = tl.load(SRC + i_bm * SRC_STRIDE + offs, mask=offs < n_elems, other=0.0)
    tl.store(DST + i_bm * DST_STRIDE + offs, v)


@triton.jit
def _zero_pad_kernel(
    DST, npad, row_stride, pad_off, BLK: tl.constexpr, DT: tl.constexpr
):
    i_r = tl.program_id(0)
    i_c = tl.program_id(1)
    o = i_c * BLK + tl.arange(0, BLK)
    tl.store(DST + i_r * row_stride + pad_off + o, tl.zeros((BLK,), DT), mask=o < npad)


@triton.jit
def _scores_kernel(
    KP, QP, ST, JP, D: tl.constexpr, HP: tl.constexpr, BM: tl.constexpr, M: tl.constexpr
):
    i_bm = tl.program_id(0)
    i_m = tl.program_id(1)
    i_b = i_bm // M
    om = i_m * BM + tl.arange(0, BM)
    od = tl.arange(0, D)
    oh = tl.arange(0, HP)
    a = tl.load(KP + i_b * (JP * D) + om[:, None] * D + od[None, :])
    q = tl.load(QP + i_bm * (HP * D) + oh[:, None] * D + od[None, :])
    acc = tl.dot(a, tl.trans(q), out_dtype=tl.float32)
    tl.store(ST + i_bm * (HP * JP) + oh[:, None] * JP + om[None, :], tl.trans(acc))


@triton.jit
def _max_partial_kernel(ST, PM, NELEM, BLK: tl.constexpr):
    i = tl.program_id(0)
    o = i * BLK + tl.arange(0, BLK)
    v = tl.load(ST + o, mask=o < NELEM, other=float("-inf"))
    tl.store(PM + i, tl.max(v))


@triton.jit
def _max_final_kernel(PM, G, NPART, scale, NP: tl.constexpr):
    o = tl.arange(0, NP)
    v = tl.load(PM + o, mask=o < NPART, other=float("-inf"))
    tl.store(G, tl.max(v) * scale)


@triton.jit
def _exp_kernel(ST, E, G, scale, BLK: tl.constexpr, DT: tl.constexpr):
    i = tl.program_id(0)
    o = i * BLK + tl.arange(0, BLK)
    e = tl.exp(tl.load(ST + o) * scale - tl.load(G))
    tl.store(E + o, e.to(DT))


@triton.jit
def _fill_kernel(DST, BLK: tl.constexpr, DT: tl.constexpr):
    i = tl.program_id(0)
    o = i * BLK + tl.arange(0, BLK)
    tl.store(DST + o, tl.zeros((BLK,), DT))


@triton.jit
def _scatter_onehot_kernel(
    IDX, AT, T, JP2, JP, TP: tl.constexpr, BT: tl.constexpr, DT: tl.constexpr
):
    i_bm = tl.program_id(0)
    i_t = tl.program_id(1)
    toff = i_t * BT + tl.arange(0, BT)
    ids = tl.load(IDX + i_bm * T + tl.minimum(toff, T - 1))
    ok = (toff < T) & (ids >= 0)
    col = tl.where(ok, ids, JP)  # dummy column: masked stores are unreliable
    tl.store(AT + i_bm * (TP * JP2) + toff * JP2 + col, tl.full((BT,), 1, DT))


@triton.jit
def _mult_kernel(
    AT,
    MULT,
    JP2,
    TP: tl.constexpr,
    BJ: tl.constexpr,
    BT: tl.constexpr,
    DT: tl.constexpr,
):
    i_bm = tl.program_id(0)
    i_j = tl.program_id(1)
    jj = i_j * BJ + tl.arange(0, BJ)
    o16 = tl.arange(0, 16)
    acc = tl.zeros((16, BJ), dtype=tl.float32)
    for k0 in range(0, TP, BT):
        toff = k0 + tl.arange(0, BT)
        a = tl.load(AT + i_bm * (TP * JP2) + toff[:, None] * JP2 + jj[None, :]).to(DT)
        on = tl.full((16, BT), 1.0, DT)
        acc = tl.dot(on, a, acc, out_dtype=tl.float32)
    tl.store(MULT + i_bm * (16 * JP2) + o16[:, None] * JP2 + jj[None, :], acc)


@triton.jit
def _out_kernel(
    E,
    MULT,
    KP,
    OP,
    SINK,
    G,
    JP,
    JP2,
    D: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    BJ: tl.constexpr,
    M: tl.constexpr,
    HPAD: tl.constexpr,
    DT: tl.constexpr,
):
    i_bm = tl.program_id(0)
    i_b = i_bm // M
    hs = tl.arange(0, HP)
    od = tl.arange(0, D)
    acc = tl.zeros((HP, D), dtype=tl.float32)
    zs = tl.zeros((HP, 16), dtype=tl.float32)
    for it in range(0, JP // BJ):
        k0 = it * BJ
        jj = k0 + tl.arange(0, BJ)
        e = tl.load(E + i_bm * (HP * JP) + hs[:, None] * JP + jj[None, :])
        mult = tl.load(MULT + i_bm * (16 * JP2) + jj)  # all 16 rows are equal
        w = (e.to(tl.float32) * mult[None, :]).to(DT)
        b = tl.load(KP + i_b * (JP * D) + jj[:, None] * D + od[None, :])
        on = tl.full((BJ, 16), 1.0, DT)
        acc = tl.dot(w, b, acc, out_dtype=tl.float32)
        zs = tl.dot(w, on, zs, out_dtype=tl.float32)
    if HPAD:
        sink = tl.load(SINK + hs, mask=hs < H, other=0.0)
    else:
        sink = tl.load(SINK + hs)
    z = tl.sum(zs, axis=1) * 0.0625 + tl.exp(sink - tl.load(G))
    inv = 1.0 / z
    tl.store(
        OP + i_bm * (HP * D) + hs[:, None] * D + od[None, :],
        (acc * inv[:, None]).to(DT),
    )


@triton.jit
def _unpad_kernel(SRC, DST, N, SRC_STRIDE, BLK: tl.constexpr, DT: tl.constexpr):
    i = tl.program_id(0)
    i_c = tl.program_id(1)
    o = i_c * BLK + tl.arange(0, BLK)
    v = tl.load(SRC + i * SRC_STRIDE + o)
    tl.store(DST + i * N + o, v)


def _pow2_ge(n):
    p = 4
    while p < n:
        p *= 2
    return p


def _largest_divisor_block(n, cap=16384):
    blk = 1
    for cand in (16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2):
        if cand <= cap and n % cand == 0:
            blk = cand
            break
    return blk


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    kv_len = kv.shape[1]
    bm = b * m

    # The packing/scatter kernels consume flat contiguous offsets; normalise
    # non-contiguous inputs once (no-op for the common contiguous case).
    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()
    if not topk_idxs.is_contiguous():
        topk_idxs = topk_idxs.contiguous()
    if not attn_sink.is_contiguous():
        attn_sink = attn_sink.contiguous()

    if h > _MAX_HEAD_TILE:
        # The dense score/output tiles are sized by HP; process very wide
        # head counts in tiles (pure data movement around the same kernels).
        parts = []
        for h0 in range(0, h, _MAX_HEAD_TILE):
            h1 = min(h0 + _MAX_HEAD_TILE, h)
            parts.append(
                sparse_attn_triton(
                    q[:, :, h0:h1, :].contiguous(),
                    kv,
                    attn_sink[h0:h1],
                    topk_idxs,
                    softmax_scale,
                )
            )
        return torch.cat(parts, dim=2)

    if q.dtype == torch.bfloat16:
        DT = tl.bfloat16
    elif q.dtype == torch.float16:
        DT = tl.float16
    else:
        DT = tl.float32

    JP = max(64, int(math.ceil(kv_len / 64.0)) * 64)
    JP2 = JP + 64
    TP = max(64, triton.next_power_of_2(int(topk)))
    HP = min(max(4, _pow2_ge(h)), _MAX_HEAD_TILE)
    HPAD = HP != h
    dev = q.device

    KP = torch.empty((b, JP, d), dtype=q.dtype, device=dev)
    BLK = 16384
    _pack_kv_kernel[(b, triton.cdiv(JP * d, BLK))](
        kv, KP, kv_len * d, kv_len * d, JP * d, BLK=BLK
    )
    kpad = (JP - kv_len) * d
    if kpad > 0:
        ZB = 4096
        _zero_pad_kernel[(b, triton.cdiv(kpad, ZB))](
            KP, kpad, JP * d, kv_len * d, BLK=ZB, DT=DT
        )

    qflat = q.reshape(bm, h, d)
    QP = torch.empty((bm, HP, d), dtype=q.dtype, device=dev)
    QB = min(16384, triton.next_power_of_2(HP * d))
    _pack_q_kernel[(bm, triton.cdiv(HP * d, QB))](
        qflat, QP, h * d, h * d, HP * d, BLK=QB
    )
    qpad = (HP - h) * d
    if qpad > 0:
        ZB = 4096
        _zero_pad_kernel[(bm, triton.cdiv(qpad, ZB))](
            QP, qpad, HP * d, h * d, BLK=ZB, DT=DT
        )

    nelem = bm * HP * JP
    NPAD = triton.cdiv(nelem, BLK) * BLK
    STb = torch.empty((NPAD,), dtype=torch.float32, device=dev)
    ST = STb[:nelem].view(bm, HP, JP)
    SBM = 64
    _scores_kernel[(bm, JP // SBM)](KP, QP, ST, JP, D=d, HP=HP, BM=SBM, M=m)

    MB = 8192
    NPART = triton.cdiv(nelem, MB)
    NP = triton.next_power_of_2(NPART)
    PM = torch.empty((NP,), dtype=torch.float32, device=dev)
    G = torch.empty((1,), dtype=torch.float32, device=dev)
    _max_partial_kernel[(NPART,)](STb, PM, nelem, BLK=MB)
    _max_final_kernel[(1,)](PM, G, NPART, softmax_scale, NP=NP)

    Eb = torch.empty((NPAD,), dtype=q.dtype, device=dev)
    E = Eb[:nelem].view(bm, HP, JP)
    EBLK = 8192
    _exp_kernel[(NPAD // EBLK,)](STb, Eb, G, softmax_scale, BLK=EBLK, DT=DT)

    idx = topk_idxs.reshape(bm, topk)
    nAT = bm * TP * JP2
    FBLK = 32768
    NATP = triton.cdiv(nAT, FBLK) * FBLK
    ATb = torch.empty((NATP,), dtype=torch.int8, device=dev)
    AT = ATb[:nAT].view(bm, TP, JP2)
    _fill_kernel[(NATP // FBLK,)](ATb, BLK=FBLK, DT=tl.int8)
    SBT = min(512, TP)
    _scatter_onehot_kernel[(bm, TP // SBT)](
        idx, AT, topk, JP2, JP, TP=TP, BT=SBT, DT=tl.int8
    )

    MULT = torch.empty((bm, 16, JP2), dtype=torch.float32, device=dev)
    BJ = 64
    # BT must not exceed the per-token row count actually allocated in AT
    # (TP).  For topk <= 64, TP == 64, so a fixed BT=128 reads 64 rows past
    # the end of AT and folds uninitialised memory into MULT.  TP >= 128 (all
    # benchmark shapes) keeps BT = 128, so their behaviour is unchanged.
    BT = min(128, TP)
    _mult_kernel[(bm, JP2 // BJ)](AT, MULT, JP2, TP=TP, BJ=BJ, BT=BT, DT=DT)

    OP = (
        torch.empty((bm, HP, d), dtype=q.dtype, device=dev)
        if HPAD
        else torch.empty((b, m, h, d), dtype=q.dtype, device=dev)
    )
    _out_kernel[(bm,)](
        E,
        MULT,
        KP,
        OP,
        attn_sink,
        G,
        JP,
        JP2,
        D=d,
        H=h,
        HP=HP,
        BJ=BJ,
        M=m,
        HPAD=HPAD,
        DT=DT,
    )
    if not HPAD:
        return OP
    out = torch.empty((b, m, h, d), dtype=q.dtype, device=dev)
    N = h * d
    UB = _largest_divisor_block(N)
    _unpad_kernel[(bm, N // UB)](OP, out, N, HP * d, BLK=UB, DT=DT)
    return out
