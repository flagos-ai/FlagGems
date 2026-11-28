"""column_stack with an explicit ``out`` buffer, implemented with Triton kernels.

Reproduces ``aten::column_stack.out``:
  * empty list -> RuntimeError("column_stack expects a non-empty TensorList")
  * rank-0 -> (1, 1), rank-1 (n,) -> (n, 1), rank >= 2 unchanged
  * all normalized inputs must agree on dim 0 and on dims >= 2
  * result dtype P = promotion of the normalized input dtypes
  * unsafe out dtype -> TypeError, matching ATen's ``cat`` message
  * out is resized in place and the same object is returned
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_T = 16
_MAX_N = 8

_ATEN_NAME = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

# kernel dtype code, passed as an int constexpr (safer than passing tl.dtype)
_DCODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int8: 4,
    torch.uint8: 5,
    torch.int16: 6,
    torch.int32: 7,
    torch.int64: 8,
}


def _cdiv(a, b):
    return -(-a // b)


def _p2le(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _p2ge(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _runs(sizes, strides):
    """Collapse (sizes, strides) into innermost-first contiguous runs."""
    items = [(int(s), int(st)) for s, st in zip(sizes, strides) if int(s) != 1]
    runs = []
    size = None
    stride = 0
    for s, st in reversed(items):
        if size is None:
            size, stride = s, st
        elif st == size * stride:
            size *= s
        else:
            runs.append((size, stride))
            size, stride = s, st
    if size is not None:
        runs.append((size, stride))
    return runs


def _pad_runs(runs, n=4):
    d = [1] * n
    s = [1] * n
    for i, (sz, st) in enumerate(runs[:n]):
        d[i] = sz
        s[i] = st
    return d, s


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@triton.jit
def _off(c, d1, s1, d2, s2, d3, s3, d4, s4, NL: tl.constexpr):
    if NL == 1:
        return c * s1
    elif NL == 2:
        return (c % d1) * s1 + (c // d1) * s2
    elif NL == 3:
        return (c % d1) * s1 + ((c // d1) % d2) * s2 + (c // (d1 * d2)) * s3
    else:
        return (
            (c % d1) * s1
            + ((c // d1) % d2) * s2
            + ((c // (d1 * d2)) % d3) * s3
            + (c // (d1 * d2 * d3)) * s4
        )


@triton.jit
def _cast_code(v, CODE: tl.constexpr):
    if CODE == 0:
        return v.to(tl.float16)
    elif CODE == 1:
        return v.to(tl.bfloat16)
    elif CODE == 2:
        return v.to(tl.float32)
    elif CODE == 3:
        return v.to(tl.float64)
    elif CODE == 4:
        return v.to(tl.int8)
    elif CODE == 5:
        return v.to(tl.uint8)
    elif CODE == 6:
        return v.to(tl.int16)
    elif CODE == 7:
        return v.to(tl.int32)
    elif CODE == 8:
        return v.to(tl.int64)
    else:
        return v


@triton.jit
def _copy_kernel(
    in_ptr,
    out_ptr,
    R,
    B,
    in_rs,
    out_rs,
    col_off,
    d1,
    s1,
    d2,
    s2,
    d3,
    s3,
    d4,
    s4,
    e1,
    u1,
    e2,
    u2,
    e3,
    u3,
    e4,
    u4,
    NL_IN: tl.constexpr,
    NL_OUT: tl.constexpr,
    NEED_CAST: tl.constexpr,
    PCODE: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    m = (rows < R)[:, None] & (cols < B)[None, :]
    ioff = _off(cols, d1, s1, d2, s2, d3, s3, d4, s4, NL_IN)
    v = tl.load(in_ptr + rows[:, None] * in_rs + ioff[None, :], mask=m, other=0)
    if NEED_CAST:
        v = _cast_code(v, PCODE).to(out_ptr.dtype.element_ty)
    ooff = _off(col_off + cols, e1, u1, e2, u2, e3, u3, e4, u4, NL_OUT)
    tl.store(out_ptr + rows[:, None] * out_rs + ooff[None, :], v, mask=m)


@triton.jit
def _eq_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    out_ptr,
    R,
    B,
    in_rs,
    out_rs,
    colstep,
    d1,
    s1,
    d2,
    s2,
    d3,
    s3,
    d4,
    s4,
    e1,
    u1,
    e2,
    u2,
    e3,
    u3,
    e4,
    u4,
    NL_IN: tl.constexpr,
    NL_OUT: tl.constexpr,
    ALIGNED: tl.constexpr,
    NEED_CAST: tl.constexpr,
    PCODE: tl.constexpr,
    NIN: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    which = tl.program_id(2)
    src = p0
    if NIN > 1:
        if which == 1:
            src = p1
    if NIN > 2:
        if which == 2:
            src = p2
    if NIN > 3:
        if which == 3:
            src = p3
    if NIN > 4:
        if which == 4:
            src = p4
    if NIN > 5:
        if which == 5:
            src = p5
    if NIN > 6:
        if which == 6:
            src = p6
    if NIN > 7:
        if which == 7:
            src = p7
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    if ALIGNED:
        # R % BR == 0, B % BC == 0, colstep % BC == 0, unit inner strides and a
        # 16-byte aligned base: every access is a full BC-wide contiguous run,
        # so no mask is needed and the offsets are provably aligned.
        ioff = tl.max_contiguous(tl.multiple_of(cols, BC), BC)
        v = tl.load(src + rows[:, None] * in_rs + ioff[None, :])
        if NEED_CAST:
            v = _cast_code(v, PCODE).to(out_ptr.dtype.element_ty)
        ooff = which * colstep + cols
        ooff = tl.max_contiguous(tl.multiple_of(ooff, BC), BC)
        tl.store(out_ptr + rows[:, None] * out_rs + ooff[None, :], v)
    else:
        m = (rows < R)[:, None] & (cols < B)[None, :]
        ioff = _off(cols, d1, s1, d2, s2, d3, s3, d4, s4, NL_IN)
        v = tl.load(src + rows[:, None] * in_rs + ioff[None, :], mask=m, other=0)
        if NEED_CAST:
            v = _cast_code(v, PCODE).to(out_ptr.dtype.element_ty)
        ooff = _off(which * colstep + cols, e1, u1, e2, u2, e3, u3, e4, u4, NL_OUT)
        tl.store(out_ptr + rows[:, None] * out_rs + ooff[None, :], v, mask=m)


@triton.jit
def _put(acc, p, rows, lm, j, NEED_CAST: tl.constexpr, PCODE: tl.constexpr):
    t = tl.load(p + rows, mask=lm, other=0)
    if NEED_CAST:
        t = _cast_code(t, PCODE).to(acc.dtype)
    idx = tl.arange(0, acc.shape[1])
    return tl.where(idx[None, :] == j, t[:, None], acc)


@triton.jit
def _col_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    out_ptr,
    R,
    n_rem,
    base,
    out_rs,
    NEED_CAST: tl.constexpr,
    PCODE: tl.constexpr,
    BR: tl.constexpr,
    BW: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    acc = tl.zeros((BR, BW), dtype=out_ptr.dtype.element_ty)
    acc = _put(acc, p0, rows, rm & (0 < n_rem), 0, NEED_CAST, PCODE)
    acc = _put(acc, p1, rows, rm & (1 < n_rem), 1, NEED_CAST, PCODE)
    acc = _put(acc, p2, rows, rm & (2 < n_rem), 2, NEED_CAST, PCODE)
    acc = _put(acc, p3, rows, rm & (3 < n_rem), 3, NEED_CAST, PCODE)
    acc = _put(acc, p4, rows, rm & (4 < n_rem), 4, NEED_CAST, PCODE)
    acc = _put(acc, p5, rows, rm & (5 < n_rem), 5, NEED_CAST, PCODE)
    acc = _put(acc, p6, rows, rm & (6 < n_rem), 6, NEED_CAST, PCODE)
    acc = _put(acc, p7, rows, rm & (7 < n_rem), 7, NEED_CAST, PCODE)
    acc = _put(acc, p8, rows, rm & (8 < n_rem), 8, NEED_CAST, PCODE)
    acc = _put(acc, p9, rows, rm & (9 < n_rem), 9, NEED_CAST, PCODE)
    acc = _put(acc, p10, rows, rm & (10 < n_rem), 10, NEED_CAST, PCODE)
    acc = _put(acc, p11, rows, rm & (11 < n_rem), 11, NEED_CAST, PCODE)
    acc = _put(acc, p12, rows, rm & (12 < n_rem), 12, NEED_CAST, PCODE)
    acc = _put(acc, p13, rows, rm & (13 < n_rem), 13, NEED_CAST, PCODE)
    acc = _put(acc, p14, rows, rm & (14 < n_rem), 14, NEED_CAST, PCODE)
    acc = _put(acc, p15, rows, rm & (15 < n_rem), 15, NEED_CAST, PCODE)
    idx = tl.arange(0, BW)
    tl.store(
        out_ptr + rows[:, None] * out_rs + (base + idx)[None, :],
        acc,
        mask=rm[:, None] & (idx < n_rem)[None, :],
    )


# --------------------------------------------------------------------------
# host entry point
# --------------------------------------------------------------------------

_CAN_CAST = getattr(torch, "canCast", None)
if _CAN_CAST is None:  # pragma: no cover
    _CAN_CAST = torch.can_cast


def _results_dtype(norm):
    p = norm[0].dtype
    for t in norm[1:]:
        p = torch.promote_types(p, t.dtype)
    return p


def column_stack_out(tensors, out=None):
    logger.debug("GEMS_ILUVATAR COLUMN_STACK_OUT")
    nt = len(tensors)
    if nt == 0:
        raise RuntimeError("column_stack expects a non-empty TensorList")

    shapes = []
    norm = []
    for t in tensors:
        d = t.dim()
        if d == 0:
            shapes.append((1, 1))
            norm.append(t.reshape(1, 1))
        elif d == 1:
            shapes.append((int(t.shape[0]), 1))
            norm.append(t.unsqueeze(1))
        else:
            shapes.append(tuple(int(x) for x in t.shape))
            norm.append(t)

    R = shapes[0][0]
    tail = shapes[0][2:]
    nd = len(shapes[0])
    ncols = 0
    offs = []
    for i in range(nt):
        sh = shapes[i]
        if len(sh) != nd or sh[0] != R or sh[2:] != tail:
            raise RuntimeError(
                "Sizes of tensors must match except in dimension 1. "
                "Expected size %d but got size %d for tensor number %d in the list."
                % (shapes[0][1], sh[1] if len(sh) > 1 else 1, i)
            )
        offs.append(ncols)
        ncols += sh[1]

    P = _results_dtype(norm)

    if out is None:
        out = torch.empty((R, ncols) + tail, dtype=P, device=tensors[0].device)
    else:
        if not _CAN_CAST(P, out.dtype):
            raise TypeError(
                "torch.cat(): input types can't be cast to the desired output type %s"
                % _ATEN_NAME.get(out.dtype, str(out.dtype))
            )
        tgt = (R, ncols) + tail
        if tuple(out.shape) != tgt:
            out.resize_(tgt)

    if out.numel() == 0:
        return out

    K = 1
    for s in tail:
        K *= s

    o_runs = _runs(out.shape[1:], out.stride()[1:])
    if len(o_runs) > 4:
        raise RuntimeError("column_stack: unsupported output layout")
    NL_OUT = len(o_runs) or 1
    od, ou = _pad_runs(o_runs)
    out_rs = int(out.stride(0))
    out_t = out if out.dtype is not torch.bool else out.view(torch.uint8)
    out_kdt = out_t.dtype
    PCODE = _DCODE.get(P, 9)

    # ---- path 1: every input contributes exactly one column ---------------
    if K == 1 and NL_OUT == 1:
        fused = True
        dt0 = norm[0].dtype
        for i in range(nt):
            t = norm[i]
            if shapes[i][1] != 1 or t.dtype != dt0:
                fused = False
                break
            if int(t.shape[0]) != 1 and int(t.stride(0)) != 1:
                fused = False
                break
        if fused:
            kptrs = [
                (t if t.dtype is not torch.bool else t.view(torch.uint8)) for t in norm
            ]
            need_cast = kptrs[0].dtype != out_kdt
            for g in range(_cdiv(ncols, _MAX_T)):
                base = g * _MAX_T
                n_rem = min(_MAX_T, ncols - base)
                bw = min(_MAX_T, _p2ge(n_rem))
                br = _p2le(_cdiv(R, 64))
                if br < 1:
                    br = 1
                lim = max(1, 4096 // bw)
                if br > lim:
                    br = lim
                args = []
                for j in range(_MAX_T):
                    k = base + j
                    args.append(kptrs[k] if k < ncols else kptrs[0])
                e = br * bw
                nw = 1 if e <= 128 else (2 if e <= 512 else 4)
                _col_kernel[(_cdiv(R, br),)](
                    *args,
                    out_t,
                    R,
                    n_rem,
                    base,
                    out_rs,
                    NEED_CAST=need_cast,
                    PCODE=PCODE,
                    BR=br,
                    BW=bw,
                    num_warps=nw,
                )
            return out

    # ---- path 2: identical inputs, one launch for all of them ------------
    if nt <= _MAX_N and NL_OUT == 1:
        same = True
        sh0 = shapes[0]
        st0 = norm[0].stride()
        dt0 = norm[0].dtype
        for i in range(1, nt):
            if shapes[i] != sh0 or norm[i].stride() != st0 or norm[i].dtype != dt0:
                same = False
                break
        if same:
            t = norm[0]
            B = sh0[1] * K
            i_runs = _runs(t.shape[1:], t.stride()[1:])
            NL_IN = len(i_runs) or 1
            if NL_IN <= 4:
                idd, isd = _pad_runs(i_runs)
                kp = [
                    (x if x.dtype is not torch.bool else x.view(torch.uint8))
                    for x in norm
                ]
                while len(kp) < _MAX_N:
                    kp.append(kp[0])
                need_cast = kp[0].dtype != out_kdt
                bc = min(_p2ge(B), 1024)
                ncb = _cdiv(B, bc)
                lim = max(1, 8192 // bc)
                want = _cdiv(1024, ncb * nt)
                br = min(lim, max(1, _p2le(_cdiv(R, max(1, want)))))
                if br < 1:
                    br = 1
                e = br * bc
                nw = 1 if e <= 256 else (2 if e <= 1024 else (4 if e <= 4096 else 8))
                aligned = (
                    R % br == 0
                    and B % bc == 0
                    and NL_IN == 1
                    and isd[0] == 1
                    and ou[0] == 1
                )
                _eq_kernel[(_cdiv(R, br), ncb, nt)](
                    kp[0],
                    kp[1],
                    kp[2],
                    kp[3],
                    kp[4],
                    kp[5],
                    kp[6],
                    kp[7],
                    out_t,
                    R,
                    B,
                    int(t.stride(0)),
                    out_rs,
                    B,
                    idd[0],
                    isd[0],
                    idd[1],
                    isd[1],
                    idd[2],
                    isd[2],
                    idd[3],
                    isd[3],
                    od[0],
                    ou[0],
                    od[1],
                    ou[1],
                    od[2],
                    ou[2],
                    od[3],
                    ou[3],
                    NL_IN=NL_IN,
                    NL_OUT=NL_OUT,
                    ALIGNED=aligned,
                    NEED_CAST=need_cast,
                    PCODE=PCODE,
                    NIN=nt,
                    BR=br,
                    BC=bc,
                    num_warps=nw,
                )
                return out

    # ---- path 3: one launch per input ------------------------------------
    for i in range(nt):
        t = norm[i]
        c = shapes[i][1]
        B = c * K
        if B == 0:
            continue
        in_rs = int(t.stride(0))
        col_off = offs[i] * K
        i_runs = _runs(t.shape[1:], t.stride()[1:])
        if len(i_runs) > 4:
            raise RuntimeError("column_stack: unsupported input layout")
        NL_IN = len(i_runs) or 1
        idd, isd = _pad_runs(i_runs)

        it = t if t.dtype is not torch.bool else t.view(torch.uint8)
        need_cast = it.dtype != out_kdt

        bc = min(_p2ge(B), 1024)
        ncb = _cdiv(B, bc)
        lim = max(1, 8192 // bc)
        want = _cdiv(1024, ncb)
        br = min(lim, max(1, _p2le(_cdiv(R, max(1, want)))))
        if br < 1:
            br = 1
        e = br * bc
        nw = 1 if e <= 256 else (2 if e <= 1024 else (4 if e <= 4096 else 8))

        _copy_kernel[(_cdiv(R, br), ncb)](
            it,
            out_t,
            R,
            B,
            in_rs,
            out_rs,
            col_off,
            idd[0],
            isd[0],
            idd[1],
            isd[1],
            idd[2],
            isd[2],
            idd[3],
            isd[3],
            od[0],
            ou[0],
            od[1],
            ou[1],
            od[2],
            ou[2],
            od[3],
            ou[3],
            NL_IN=NL_IN,
            NL_OUT=NL_OUT,
            NEED_CAST=need_cast,
            PCODE=PCODE,
            BR=br,
            BC=bc,
            num_warps=nw,
        )
    return out
