import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Naive sequential rank-1 ormqr (used for small K / small M)
# ---------------------------------------------------------------------------
@triton.jit
def _ormqr_kernel(
    input_ptr,
    tau_ptr,
    other_ptr,
    out_ptr,
    DIM,
    F,
    K,
    stride_im,
    stride_ik,  # input strides: reflector axis, k
    stride_ib,  # input batch stride
    stride_tb,  # tau batch stride
    stride_c0,
    stride_c1,  # C (other/out) strides: tile axis0, tile axis1
    stride_cb,  # C batch stride
    LEFT: tl.constexpr,
    ASCEND: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offs_f = pid0 * BLOCK_C + tl.arange(0, BLOCK_C)
    fmask = offs_f < F

    base_in = input_ptr + pid1 * stride_ib
    base_tau = tau_ptr + pid1 * stride_tb
    base_out = out_ptr + pid1 * stride_cb
    base_oth = other_ptr + pid1 * stride_cb

    # ---- prelude: out <- other (per program strip) ----
    for r in range(0, DIM, BLOCK_R):
        off_r = r + tl.arange(0, BLOCK_R)
        rmask = off_r < DIM
        if LEFT:
            tile = tl.load(
                base_oth + off_r[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                mask=rmask[:, None] & fmask[None, :],
                other=0.0,
            )
            tl.store(
                base_out + off_r[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                tile,
                mask=rmask[:, None] & fmask[None, :],
            )
        else:
            tile = tl.load(
                base_oth + offs_f[:, None] * stride_c0 + off_r[None, :] * stride_c1,
                mask=fmask[:, None] & rmask[None, :],
                other=0.0,
            )
            tl.store(
                base_out + offs_f[:, None] * stride_c0 + off_r[None, :] * stride_c1,
                tile,
                mask=fmask[:, None] & rmask[None, :],
            )

    acc_dtype = tl.float64 if IS_F64 else tl.float32

    # NOTE: Triton's `range` does not support negative steps; use a positive
    # loop and compute the reflector index according to ASCEND.
    for j in range(0, K):
        if ASCEND:
            i = j
        else:
            i = K - 1 - j

        tau_i = tl.load(base_tau + i)
        if LEFT:
            # tile axis0 = reflector positions, axis1 = free positions
            w = tl.zeros([BLOCK_C], dtype=acc_dtype)
            r = i
            while r < DIM:
                off_r = r + tl.arange(0, BLOCK_R)
                rmask = off_r < DIM
                v = tl.load(
                    base_in + off_r * stride_im + i * stride_ik, mask=rmask, other=0.0
                )
                v = tl.where(off_r == i, 1.0, v)
                ctile = tl.load(
                    base_out + off_r[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                    mask=rmask[:, None] & fmask[None, :],
                    other=0.0,
                )
                w += tl.sum(v[:, None] * ctile, axis=0)
                r += BLOCK_R
            r = i
            while r < DIM:
                off_r = r + tl.arange(0, BLOCK_R)
                rmask = off_r < DIM
                v = tl.load(
                    base_in + off_r * stride_im + i * stride_ik, mask=rmask, other=0.0
                )
                v = tl.where(off_r == i, 1.0, v)
                ctile = tl.load(
                    base_out + off_r[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                    mask=rmask[:, None] & fmask[None, :],
                    other=0.0,
                )
                ctile = ctile - tau_i * (v[:, None] * w[None, :])
                tl.store(
                    base_out + off_r[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                    ctile,
                    mask=rmask[:, None] & fmask[None, :],
                )
                r += BLOCK_R
        else:
            # tile axis0 = free positions, axis1 = reflector positions
            w = tl.zeros([BLOCK_C], dtype=acc_dtype)
            r = i
            while r < DIM:
                off_r = r + tl.arange(0, BLOCK_R)
                rmask = off_r < DIM
                v = tl.load(
                    base_in + off_r * stride_im + i * stride_ik, mask=rmask, other=0.0
                )
                v = tl.where(off_r == i, 1.0, v)
                ctile = tl.load(
                    base_out + offs_f[:, None] * stride_c0 + off_r[None, :] * stride_c1,
                    mask=fmask[:, None] & rmask[None, :],
                    other=0.0,
                )
                w += tl.sum(ctile * v[None, :], axis=1)
                r += BLOCK_R
            r = i
            while r < DIM:
                off_r = r + tl.arange(0, BLOCK_R)
                rmask = off_r < DIM
                v = tl.load(
                    base_in + off_r * stride_im + i * stride_ik, mask=rmask, other=0.0
                )
                v = tl.where(off_r == i, 1.0, v)
                ctile = tl.load(
                    base_out + offs_f[:, None] * stride_c0 + off_r[None, :] * stride_c1,
                    mask=fmask[:, None] & rmask[None, :],
                    other=0.0,
                )
                ctile = ctile - tau_i * (w[:, None] * v[None, :])
                tl.store(
                    base_out + offs_f[:, None] * stride_c0 + off_r[None, :] * stride_c1,
                    ctile,
                    mask=fmask[:, None] & rmask[None, :],
                )
                r += BLOCK_R


# ---------------------------------------------------------------------------
# WY-blocked ormqr (used for large K / large M)
# ---------------------------------------------------------------------------
@triton.jit
def _wy_t_kernel(
    input_ptr,
    tau_ptr,
    t_ptr,
    M,
    K,
    stride_im,
    stride_ik,
    stride_ib,
    stride_tb,
    stride_tbb,  # batch stride of T buffer
    BS: tl.constexpr,
    BR: tl.constexpr,
    IS_F64: tl.constexpr,
):
    blk = tl.program_id(0)
    pid_b = tl.program_id(1)
    j0 = blk * BS
    bse = tl.minimum(K - j0, BS)

    off_b = tl.arange(0, BS)
    off_r = tl.arange(0, BR)
    acc = tl.float64 if IS_F64 else tl.float32

    base_in = input_ptr + pid_b * stride_ib
    base_tau = tau_ptr + pid_b * stride_tb

    # S = V^T V over rows [j0, M)
    S = tl.zeros((BS, BS), dtype=acc)
    r = j0
    while r < M:
        roff = r + off_r
        rmask = roff < M
        v = tl.load(
            base_in + roff[:, None] * stride_im + (j0 + off_b)[None, :] * stride_ik,
            mask=rmask[:, None],
            other=0.0,
        )
        v = tl.where(roff[:, None] < (j0 + off_b)[None, :], 0.0, v)
        v = tl.where(roff[:, None] == (j0 + off_b)[None, :], 1.0, v)
        v = tl.where(off_b[None, :] < bse, v, 0.0)
        S += tl.dot(tl.trans(v), v)
        r += BR

    # dlarft-style recurrence: T[j,j] = tau_j; T[i,j] = -tau_j * (T @ S[:,j])[i]
    T = tl.zeros((BS, BS), dtype=acc)
    for j in range(0, BS):
        tau_j = tl.load(base_tau + j0 + j, mask=j < bse, other=0.0)
        s_j = tl.sum(tl.where(off_b[None, :] == j, S, 0.0), axis=1)
        tcol = -tau_j * tl.sum(T * s_j[None, :], axis=1)
        tcol = tl.where(off_b < j, tcol, 0.0)
        tcol = tl.where(off_b == j, tau_j, tcol)
        tcol = tl.where(off_b < bse, tcol, 0.0)
        T = tl.where(off_b[None, :] == j, tcol[:, None], T)

    tl.store(
        t_ptr
        + pid_b * stride_tbb
        + blk * BS * BS
        + off_b[:, None] * BS
        + off_b[None, :],
        T,
    )


@triton.jit
def _wy_apply_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    t_ptr,
    y_ptr,
    DIM,
    F,
    K,
    NBLK,
    NPROG,
    stride_im,
    stride_ik,
    stride_ib,
    stride_c0,
    stride_c1,
    stride_cb,
    stride_tbb,
    LEFT: tl.constexpr,
    ASCEND: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    IS_F64: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_f = pid_c * BC + tl.arange(0, BC)
    fmask = offs_f < F
    off_r = tl.arange(0, BR)
    off_b = tl.arange(0, BS)

    base_in = input_ptr + pid_b * stride_ib
    base_out = out_ptr + pid_b * stride_cb
    base_oth = other_ptr + pid_b * stride_cb
    base_t = t_ptr + pid_b * stride_tbb
    base_y = y_ptr + (pid_b * NPROG + pid_c) * BS * BC

    # prelude: out <- other
    for r in range(0, DIM, BR):
        roff = r + off_r
        rmask = roff < DIM
        if LEFT:
            tile = tl.load(
                base_oth + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                mask=rmask[:, None] & fmask[None, :],
                other=0.0,
            )
            tl.store(
                base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                tile,
                mask=rmask[:, None] & fmask[None, :],
            )
        else:
            tile = tl.load(
                base_oth + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                mask=fmask[:, None] & rmask[None, :],
                other=0.0,
            )
            tl.store(
                base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                tile,
                mask=fmask[:, None] & rmask[None, :],
            )

    if ASCEND:
        blk = 0
        while blk < NBLK:
            _wy_block(
                base_in,
                base_out,
                base_t,
                base_y,
                DIM,
                K,
                stride_im,
                stride_ik,
                stride_c0,
                stride_c1,
                offs_f,
                fmask,
                off_r,
                off_b,
                blk,
                LEFT,
                TRANSPOSE,
                BS,
                BR,
                BC,
                IS_F64,
                USE_DOT,
            )
            blk += 1
    else:
        blk = NBLK - 1
        while blk >= 0:
            _wy_block(
                base_in,
                base_out,
                base_t,
                base_y,
                DIM,
                K,
                stride_im,
                stride_ik,
                stride_c0,
                stride_c1,
                offs_f,
                fmask,
                off_r,
                off_b,
                blk,
                LEFT,
                TRANSPOSE,
                BS,
                BR,
                BC,
                IS_F64,
                USE_DOT,
            )
            blk -= 1


@triton.jit
def _wy_block(
    base_in,
    base_out,
    base_t,
    base_y,
    DIM,
    K,
    stride_im,
    stride_ik,
    stride_c0,
    stride_c1,
    offs_f,
    fmask,
    off_r,
    off_b,
    blk,
    LEFT: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    IS_F64: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    acc = tl.float64 if IS_F64 else tl.float32
    j0 = blk * BS
    bse = tl.minimum(K - j0, BS)

    if IS_F64 and not USE_DOT:
        # ---------------- fp64 elementwise path (avoids slow fp64 tl.dot) ----
        if LEFT:
            W = tl.zeros((BS, BC), dtype=tl.float64)
        else:
            W = tl.zeros((BC, BS), dtype=tl.float64)
        r = j0
        while r < DIM:
            roff = r + off_r
            rmask = roff < DIM
            if LEFT:
                ctile = tl.load(
                    base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                    mask=rmask[:, None] & fmask[None, :],
                    other=0.0,
                )
            else:
                ctile = tl.load(
                    base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                    mask=fmask[:, None] & rmask[None, :],
                    other=0.0,
                )
            for b in range(0, BS):
                v_b = tl.load(
                    base_in + roff * stride_im + (j0 + b) * stride_ik,
                    mask=rmask,
                    other=0.0,
                )
                v_b = tl.where(roff < j0 + b, 0.0, v_b)
                v_b = tl.where(roff == j0 + b, 1.0, v_b)
                v_b = tl.where(b < bse, v_b, 0.0)
                if LEFT:
                    wb = tl.sum(v_b[:, None] * ctile, axis=0)
                    W += tl.where(off_b[:, None] == b, wb[None, :], 0.0)
                else:
                    wb = tl.sum(ctile * v_b[None, :], axis=1)
                    W += tl.where(off_b[None, :] == b, wb[:, None], 0.0)
            r += BR
        # Y = T W  (or T^T W), stored to per-program scratch
        if LEFT:
            for i in range(0, BS):
                if TRANSPOSE:
                    t_i = tl.load(
                        base_t + blk * BS * BS + off_b * BS + i,
                        mask=off_b < bse,
                        other=0.0,
                    )
                else:
                    t_i = tl.load(
                        base_t + blk * BS * BS + i * BS + off_b,
                        mask=off_b < bse,
                        other=0.0,
                    )
                y_i = tl.sum(t_i[:, None] * W, axis=0)
                tl.store(base_y + i * BC + offs_f, y_i, mask=fmask)
        else:
            for i in range(0, BS):
                if TRANSPOSE:
                    t_i = tl.load(
                        base_t + blk * BS * BS + i * BS + off_b,
                        mask=off_b < bse,
                        other=0.0,
                    )
                else:
                    t_i = tl.load(
                        base_t + blk * BS * BS + off_b * BS + i,
                        mask=off_b < bse,
                        other=0.0,
                    )
                y_i = tl.sum(W * t_i[None, :], axis=1)
                tl.store(base_y + i * BC + offs_f, y_i, mask=fmask)
        r = j0
        while r < DIM:
            roff = r + off_r
            rmask = roff < DIM
            if LEFT:
                ctile = tl.load(
                    base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                    mask=rmask[:, None] & fmask[None, :],
                    other=0.0,
                )
            else:
                ctile = tl.load(
                    base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                    mask=fmask[:, None] & rmask[None, :],
                    other=0.0,
                )
            for b in range(0, BS):
                v_b = tl.load(
                    base_in + roff * stride_im + (j0 + b) * stride_ik,
                    mask=rmask,
                    other=0.0,
                )
                v_b = tl.where(roff < j0 + b, 0.0, v_b)
                v_b = tl.where(roff == j0 + b, 1.0, v_b)
                v_b = tl.where(b < bse, v_b, 0.0)
                y_b = tl.load(base_y + b * BC + offs_f, mask=fmask, other=0.0)
                if LEFT:
                    ctile = ctile - v_b[:, None] * y_b[None, :]
                else:
                    ctile = ctile - y_b[:, None] * v_b[None, :]
            if LEFT:
                tl.store(
                    base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                    ctile,
                    mask=rmask[:, None] & fmask[None, :],
                )
            else:
                tl.store(
                    base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                    ctile,
                    mask=fmask[:, None] & rmask[None, :],
                )
            r += BR
        return

    # ---------------- fp32 tl.dot path ----
    vptr = base_in + (j0 + off_b)[None, :] * stride_ik

    if LEFT:
        W = tl.zeros((BS, BC), dtype=acc)
        r = j0
        while r < DIM:
            roff = r + off_r
            rmask = roff < DIM
            v = tl.load(
                vptr + roff[:, None] * stride_im, mask=rmask[:, None], other=0.0
            )
            v = tl.where(roff[:, None] < (j0 + off_b)[None, :], 0.0, v)
            v = tl.where(roff[:, None] == (j0 + off_b)[None, :], 1.0, v)
            v = tl.where(off_b[None, :] < bse, v, 0.0)
            ctile = tl.load(
                base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                mask=rmask[:, None] & fmask[None, :],
                other=0.0,
            )
            W += tl.dot(tl.trans(v), ctile)
            r += BR
        Tb = tl.load(base_t + blk * BS * BS + off_b[:, None] * BS + off_b[None, :])
        if TRANSPOSE:
            Y = tl.dot(tl.trans(Tb), W)
        else:
            Y = tl.dot(Tb, W)
        r = j0
        while r < DIM:
            roff = r + off_r
            rmask = roff < DIM
            v = tl.load(
                vptr + roff[:, None] * stride_im, mask=rmask[:, None], other=0.0
            )
            v = tl.where(roff[:, None] < (j0 + off_b)[None, :], 0.0, v)
            v = tl.where(roff[:, None] == (j0 + off_b)[None, :], 1.0, v)
            v = tl.where(off_b[None, :] < bse, v, 0.0)
            ctile = tl.load(
                base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                mask=rmask[:, None] & fmask[None, :],
                other=0.0,
            )
            ctile = ctile - tl.dot(v, Y)
            tl.store(
                base_out + roff[:, None] * stride_c0 + offs_f[None, :] * stride_c1,
                ctile,
                mask=rmask[:, None] & fmask[None, :],
            )
            r += BR
    else:
        W = tl.zeros((BC, BS), dtype=acc)
        r = j0
        while r < DIM:
            roff = r + off_r
            rmask = roff < DIM
            v = tl.load(
                vptr + roff[:, None] * stride_im, mask=rmask[:, None], other=0.0
            )
            v = tl.where(roff[:, None] < (j0 + off_b)[None, :], 0.0, v)
            v = tl.where(roff[:, None] == (j0 + off_b)[None, :], 1.0, v)
            v = tl.where(off_b[None, :] < bse, v, 0.0)
            ctile = tl.load(
                base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                mask=fmask[:, None] & rmask[None, :],
                other=0.0,
            )
            W += tl.dot(ctile, v)
            r += BR
        Tb = tl.load(base_t + blk * BS * BS + off_b[:, None] * BS + off_b[None, :])
        if TRANSPOSE:
            Y = tl.dot(W, tl.trans(Tb))
        else:
            Y = tl.dot(W, Tb)
        r = j0
        while r < DIM:
            roff = r + off_r
            rmask = roff < DIM
            v = tl.load(
                vptr + roff[:, None] * stride_im, mask=rmask[:, None], other=0.0
            )
            v = tl.where(roff[:, None] < (j0 + off_b)[None, :], 0.0, v)
            v = tl.where(roff[:, None] == (j0 + off_b)[None, :], 1.0, v)
            v = tl.where(off_b[None, :] < bse, v, 0.0)
            ctile = tl.load(
                base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                mask=fmask[:, None] & rmask[None, :],
                other=0.0,
            )
            ctile = ctile - tl.dot(Y, tl.trans(v))
            tl.store(
                base_out + offs_f[:, None] * stride_c0 + roff[None, :] * stride_c1,
                ctile,
                mask=fmask[:, None] & rmask[None, :],
            )
            r += BR


def _launch_naive(
    input,
    tau,
    other,
    out,
    left,
    transpose,
    F,
    M,
    K,
    B,
    stride_ib,
    stride_tb,
    stride_cb,
    stride_c0,
    stride_c1,
    IS_F64,
):
    if F <= 64:
        BLOCK_C = 64
    else:
        BLOCK_C = 128
    BLOCK_R = 32
    ASCEND = left == transpose
    grid = (triton.cdiv(F, BLOCK_C), B)
    _ormqr_kernel[grid](
        input,
        tau,
        other,
        out,
        M,
        F,
        K,
        input.stride(-2),
        input.stride(-1),
        stride_ib,
        stride_tb,
        stride_c0,
        stride_c1,
        stride_cb,
        LEFT=left,
        ASCEND=ASCEND,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        IS_F64=IS_F64,
        num_warps=2,
    )


def _launch_wy(
    input,
    tau,
    other,
    out,
    left,
    transpose,
    F,
    M,
    K,
    B,
    stride_ib,
    stride_tb,
    stride_cb,
    stride_c0,
    stride_c1,
    IS_F64,
    BS=32,
    BR=32,
    BC=128,
    nw=4,
    use_dot=False,
):
    NBLK = triton.cdiv(K, BS)
    t_buf = torch.empty((B, NBLK, BS, BS), dtype=input.dtype, device=input.device)
    stride_tbb = NBLK * BS * BS
    nprog = triton.cdiv(F, BC)
    y_buf = torch.empty((B * nprog, BS, BC), dtype=input.dtype, device=input.device)
    ASCEND = left == transpose
    grid_t = (NBLK, B)
    _wy_t_kernel[grid_t](
        input,
        tau,
        t_buf,
        M,
        K,
        input.stride(-2),
        input.stride(-1),
        stride_ib,
        stride_tb,
        stride_tbb,
        BS=BS,
        BR=BR,
        IS_F64=IS_F64,
        num_warps=2,
    )
    grid_a = (nprog, B)
    _wy_apply_kernel[grid_a](
        input,
        other,
        out,
        t_buf,
        y_buf,
        M,
        F,
        K,
        NBLK,
        nprog,
        input.stride(-2),
        input.stride(-1),
        stride_ib,
        stride_c0,
        stride_c1,
        stride_cb,
        stride_tbb,
        LEFT=left,
        ASCEND=ASCEND,
        TRANSPOSE=transpose,
        BS=BS,
        BR=BR,
        BC=BC,
        IS_F64=IS_F64,
        USE_DOT=use_dot,
        num_warps=nw,
    )


def run(input, tau, other, left=True, transpose=False):
    M, K = input.shape[-2], input.shape[-1]
    if left:
        N = other.shape[-1]
    else:
        N = other.shape[-2]
    F = N

    out = torch.empty_like(other)

    if input.dim() == 2:
        B = 1
        stride_ib = 0
        stride_tb = 0
        stride_cb = 0
    else:
        B = input.shape[:-2].numel()
        stride_ib = input.stride(-3)
        stride_tb = tau.stride(-2)
        stride_cb = other.stride(-3)

    stride_c0 = other.stride(-2)
    stride_c1 = other.stride(-1)
    IS_F64 = input.dtype == torch.float64

    if IS_F64 and K >= 1024:
        if N <= 4096:
            # fp64 tl.dot WY path (exact fp64 tl.dot on this backend); best for N<=4096
            _launch_wy(
                input,
                tau,
                other,
                out,
                left,
                transpose,
                F,
                M,
                K,
                B,
                stride_ib,
                stride_tb,
                stride_cb,
                stride_c0,
                stride_c1,
                IS_F64,
                BS=32,
                BR=32,
                BC=128,
                nw=8,
                use_dot=True,
            )
        else:
            # fp64 elementwise WY path (tl.dot fp64 is slow at huge N); best for N>4096
            _launch_wy(
                input,
                tau,
                other,
                out,
                left,
                transpose,
                F,
                M,
                K,
                B,
                stride_ib,
                stride_tb,
                stride_cb,
                stride_c0,
                stride_c1,
                IS_F64,
                BS=64,
                BR=64,
                BC=64,
                nw=4,
            )
    elif (not IS_F64) and K >= 512:
        # fp32 WY dot path, large-K tile config
        _launch_wy(
            input,
            tau,
            other,
            out,
            left,
            transpose,
            F,
            M,
            K,
            B,
            stride_ib,
            stride_tb,
            stride_cb,
            stride_c0,
            stride_c1,
            IS_F64,
            BS=64,
            BR=64,
            BC=128,
            nw=8,
        )
    elif (not IS_F64) and K >= 48:
        # fp32 WY dot path, small/medium tile config
        _launch_wy(
            input,
            tau,
            other,
            out,
            left,
            transpose,
            F,
            M,
            K,
            B,
            stride_ib,
            stride_tb,
            stride_cb,
            stride_c0,
            stride_c1,
            IS_F64,
            BS=32,
            BR=32,
            BC=128,
            nw=4,
        )
    else:
        _launch_naive(
            input,
            tau,
            other,
            out,
            left,
            transpose,
            F,
            M,
            K,
            B,
            stride_ib,
            stride_tb,
            stride_cb,
            stride_c0,
            stride_c1,
            IS_F64,
        )
    return out


# Alias for FlagGems import convention
ormqr = run
