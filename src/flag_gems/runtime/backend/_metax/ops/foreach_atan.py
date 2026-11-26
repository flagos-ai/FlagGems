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

"""Triton implementation of aten::_foreach_atan (Gems/FlagGems entry point)."""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# atan intrinsic resolution (libdevice across triton/flagtree layouts)
# ---------------------------------------------------------------------------
_ATAN = None
_ATAN2 = None
for _mod in (
    "triton.language.extra.libdevice",
    "triton.language.extra.cuda.libdevice",
):
    try:
        import importlib

        _m = importlib.import_module(_mod)
    except Exception:
        continue
    if _ATAN is None:
        _ATAN = getattr(_m, "atan", None)
    if _ATAN2 is None:
        _ATAN2 = getattr(_m, "atan2", None)
    if _ATAN is not None and _ATAN2 is not None:
        break
if _ATAN is None:
    _ATAN = getattr(getattr(tl, "math", None), "atan", None)
if _ATAN2 is None:
    _ATAN2 = getattr(getattr(tl, "math", None), "atan2", None)
if _ATAN is None:

    @triton.jit
    def _ATAN(x):  # type: ignore[misc]
        return _ATAN2(x, 1.0)


BLOCK = 1024  # 32/64-bit dtypes: 4 elements per lane -> 128-bit access
BLOCK16 = 2048  # 16-bit dtypes: 8 elements per lane -> 128-bit access
NUM_WARPS = 4
_FUSED_MAX = 16
_FUSED_MAX_BYTES = 64 * 1024 * 1024
_RANK_MAX = 8

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _atan_cplx(xr, xi, FP64: tl.constexpr):
    """Complex atan: real = 0.5*atan2(2x, 1-|z|^2), imag = 0.25*ln(|1-iz|^2/|1+iz|^2)."""
    if not FP64:
        xr = xr.to(tl.float32)
        xi = xi.to(tl.float32)
    r2 = xr * xr
    a1 = (1.0 - xi) * (1.0 - xi) + r2
    a2 = (1.0 + xi) * (1.0 + xi) + r2
    yr = 0.5 * _ATAN2(2.0 * xr, 1.0 - r2 - xi * xi)
    yi = 0.25 * (tl.log(a2) - tl.log(a1))
    return yr, yi


@triton.jit(do_not_specialize=["n"])
def _atan_flat_kernel(
    x_ptr,
    y_ptr,
    n,
    FP64: tl.constexpr,
    COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    beg = pid * BLOCK
    offs = beg + tl.arange(0, BLOCK)
    if beg + BLOCK <= n:
        # full block: unmasked so the accesses vectorize
        if COMPLEX:
            xr = tl.load(x_ptr + 2 * offs)
            xi = tl.load(x_ptr + 2 * offs + 1)
            yr, yi = _atan_cplx(xr, xi, FP64)
            tl.store(y_ptr + 2 * offs, yr)
            tl.store(y_ptr + 2 * offs + 1, yi)
        elif FP64:
            x = tl.load(x_ptr + offs)
            tl.store(y_ptr + offs, _ATAN(x))
        else:
            x = tl.load(x_ptr + offs)
            tl.store(y_ptr + offs, _ATAN(x.to(tl.float32)))
    else:
        mask = offs < n
        if COMPLEX:
            xr = tl.load(x_ptr + 2 * offs, mask=mask)
            xi = tl.load(x_ptr + 2 * offs + 1, mask=mask)
            yr, yi = _atan_cplx(xr, xi, FP64)
            tl.store(y_ptr + 2 * offs, yr, mask=mask)
            tl.store(y_ptr + 2 * offs + 1, yi, mask=mask)
        elif FP64:
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(y_ptr + offs, _ATAN(x), mask=mask)
        else:
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(y_ptr + offs, _ATAN(x.to(tl.float32)), mask=mask)


@triton.jit
def _atan_body(x_ptr, y_ptr, offs, full, n, FP64: tl.constexpr):
    if full:
        x = tl.load(x_ptr + offs)
        if FP64:
            tl.store(y_ptr + offs, _ATAN(x))
        else:
            tl.store(y_ptr + offs, _ATAN(x.to(tl.float32)))
    else:
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        if FP64:
            tl.store(y_ptr + offs, _ATAN(x), mask=m)
        else:
            tl.store(y_ptr + offs, _ATAN(x.to(tl.float32)), mask=m)


@triton.jit(do_not_specialize=["n"])
def _atan_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    y_ptr,
    n,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 same-shape contiguous members sharing one output buffer.

    Blocks are laid out member-major (t = pid // nb, blk = pid % nb), so each
    program touches exactly one member; unused slots are never selected.
    """
    pid = tl.program_id(0)
    nb = tl.cdiv(n, BLOCK)
    t = pid // nb
    blk = pid - t * nb
    beg = blk * BLOCK
    offs = beg + tl.arange(0, BLOCK)
    full = beg + BLOCK <= n
    yp = y_ptr + t.to(tl.int64) * n
    if t == 0:
        _atan_body(x0, yp, offs, full, n, FP64)
    if t == 1:
        _atan_body(x1, yp, offs, full, n, FP64)
    if t == 2:
        _atan_body(x2, yp, offs, full, n, FP64)
    if t == 3:
        _atan_body(x3, yp, offs, full, n, FP64)
    if t == 4:
        _atan_body(x4, yp, offs, full, n, FP64)
    if t == 5:
        _atan_body(x5, yp, offs, full, n, FP64)
    if t == 6:
        _atan_body(x6, yp, offs, full, n, FP64)
    if t == 7:
        _atan_body(x7, yp, offs, full, n, FP64)
    if t == 8:
        _atan_body(x8, yp, offs, full, n, FP64)
    if t == 9:
        _atan_body(x9, yp, offs, full, n, FP64)
    if t == 10:
        _atan_body(x10, yp, offs, full, n, FP64)
    if t == 11:
        _atan_body(x11, yp, offs, full, n, FP64)
    if t == 12:
        _atan_body(x12, yp, offs, full, n, FP64)
    if t == 13:
        _atan_body(x13, yp, offs, full, n, FP64)
    if t == 14:
        _atan_body(x14, yp, offs, full, n, FP64)
    if t == 15:
        _atan_body(x15, yp, offs, full, n, FP64)


@triton.jit
def _decode_off(
    idx,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    RANK: tl.constexpr,
):
    """Row-major multi-index -> element offset (strides in elements, int64 safe)."""
    off = (idx % S0).to(tl.int64) * T0
    rem = idx // S0
    if RANK > 1:
        off += (rem % S1).to(tl.int64) * T1
        rem = rem // S1
    if RANK > 2:
        off += (rem % S2).to(tl.int64) * T2
        rem = rem // S2
    if RANK > 3:
        off += (rem % S3).to(tl.int64) * T3
        rem = rem // S3
    if RANK > 4:
        off += (rem % S4).to(tl.int64) * T4
        rem = rem // S4
    if RANK > 5:
        off += (rem % S5).to(tl.int64) * T5
        rem = rem // S5
    if RANK > 6:
        off += (rem % S6).to(tl.int64) * T6
        rem = rem // S6
    if RANK > 7:
        off += (rem % S7).to(tl.int64) * T7
    return off


@triton.jit(
    do_not_specialize=[
        "O0",
        "O1",
        "O2",
        "O3",
        "O4",
        "O5",
        "O6",
        "O7",
        "U0",
        "U1",
        "U2",
        "U3",
        "U4",
        "U5",
        "U6",
        "U7",
        "INNER",
    ]
)
def _atan_layout_kernel(
    x_ptr,
    y_ptr,
    O0,
    O1,
    O2,
    O3,
    O4,
    O5,
    O6,
    O7,
    U0,
    U1,
    U2,
    U3,
    U4,
    U5,
    U6,
    U7,
    INNER,
    RANK: tl.constexpr,
    FP64: tl.constexpr,
    COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Input and output share shape+stride; one axis has stride 1 (inner axis)."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = _decode_off(
        pid1,
        O0,
        O1,
        O2,
        O3,
        O4,
        O5,
        O6,
        O7,
        U0,
        U1,
        U2,
        U3,
        U4,
        U5,
        U6,
        U7,
        RANK,
    )
    i = pid0 * BLOCK + tl.arange(0, BLOCK)
    mask = i < INNER
    off = base + i
    if COMPLEX:
        xr = tl.load(x_ptr + 2 * off, mask=mask)
        xi = tl.load(x_ptr + 2 * off + 1, mask=mask)
        yr, yi = _atan_cplx(xr, xi, FP64)
        tl.store(y_ptr + 2 * off, yr, mask=mask)
        tl.store(y_ptr + 2 * off + 1, yi, mask=mask)
    elif FP64:
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(y_ptr + off, _ATAN(x), mask=mask)
    else:
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(y_ptr + off, _ATAN(x.to(tl.float32)), mask=mask)


@triton.jit(
    do_not_specialize=[
        "numel",
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
    ]
)
def _atan_strided_kernel(
    x_ptr,
    y_ptr,
    numel,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    RANK: tl.constexpr,
    FP64: tl.constexpr,
    COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generic path: arbitrary strided input, dense row-major output."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    off = _decode_off(
        idx,
        S0,
        S1,
        S2,
        S3,
        S4,
        S5,
        S6,
        S7,
        T0,
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        RANK,
    )
    if COMPLEX:
        xr = tl.load(x_ptr + 2 * off, mask=mask)
        xi = tl.load(x_ptr + 2 * off + 1, mask=mask)
        yr, yi = _atan_cplx(xr, xi, FP64)
        tl.store(y_ptr + 2 * idx, yr, mask=mask)
        tl.store(y_ptr + 2 * idx + 1, yi, mask=mask)
    elif FP64:
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(y_ptr + idx, _ATAN(x), mask=mask)
    else:
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(y_ptr + idx, _ATAN(x.to(tl.float32)), mask=mask)


@triton.jit(
    do_not_specialize=[
        "xp",
        "yp",
        "numel",
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
    ]
)
def _atan_cplx_ptr_kernel(
    xp,
    yp,
    numel,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    RANK: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex tensors without a stride-1 axis: raw data_ptr args."""
    if FP64:
        x_ptr = xp.to(tl.pointer_type(tl.float64))
        y_ptr = yp.to(tl.pointer_type(tl.float64))
    else:
        x_ptr = xp.to(tl.pointer_type(tl.float32))
        y_ptr = yp.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    off = _decode_off(
        idx,
        S0,
        S1,
        S2,
        S3,
        S4,
        S5,
        S6,
        S7,
        T0,
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        RANK,
    )
    xr = tl.load(x_ptr + 2 * off, mask=mask)
    xi = tl.load(x_ptr + 2 * off + 1, mask=mask)
    yr, yi = _atan_cplx(xr, xi, FP64)
    tl.store(y_ptr + 2 * idx, yr, mask=mask)
    tl.store(y_ptr + 2 * idx + 1, yi, mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
_CUR_DEV = [None]


def _ensure_device(dev):
    if dev is None:
        return
    cur = _CUR_DEV[0]
    if cur is None:
        try:
            cur = torch.cuda.current_device()
        except Exception:
            cur = dev
        _CUR_DEV[0] = cur
    if dev != cur:
        torch.cuda.set_device(dev)
        _CUR_DEV[0] = dev


def _out_dtype(dt):
    if dt.is_floating_point:
        return dt
    return torch.get_default_dtype()


def _reduce_dims(shape, stride):
    dims = [(int(s), int(st)) for s, st in zip(shape, stride) if int(s) != 1]
    if not dims:
        return [(1, 1)]
    changed = True
    while changed and len(dims) > 1:
        changed = False
        out = [dims[0]]
        for s, st in dims[1:]:
            ps, pst = out[-1]
            if pst == st * s:
                out[-1] = (ps * s, st)
                changed = True
            else:
                out.append((s, st))
        dims = out
    return dims


def _launch_layout(x_ptr_f, y_ptr_f, dims, inner, n, fp64, cplx):
    outer = dims[:inner] + dims[inner + 1 :]
    inner_size = dims[inner][0]
    outer_numel = n // inner_size
    OUT_ = [1] * _RANK_MAX
    U = [1] * _RANK_MAX
    for k, (s, st) in enumerate(outer):
        OUT_[k] = s
        U[k] = st
    grid = (triton.cdiv(inner_size, BLOCK), outer_numel)
    _atan_layout_kernel[grid](
        x_ptr_f,
        y_ptr_f,
        *OUT_,
        *U,
        inner_size,
        RANK=len(outer),
        FP64=fp64,
        COMPLEX=cplx,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )


def _launch_strided(x_ptr_f, y_ptr_f, dims, n, fp64, cplx):
    S = [1] * _RANK_MAX
    T = [1] * _RANK_MAX
    for k, (s, st) in enumerate(list(reversed(dims))[:_RANK_MAX]):
        S[k] = s
        T[k] = st
    grid = (triton.cdiv(n, BLOCK),)
    _atan_strided_kernel[grid](
        x_ptr_f,
        y_ptr_f,
        n,
        *S,
        *T,
        RANK=min(len(dims), _RANK_MAX),
        FP64=fp64,
        COMPLEX=cplx,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )


def _atan_complex_one(t):
    fdt = torch.float32 if t.dtype is torch.complex64 else torch.float64
    fp64 = fdt is torch.float64
    n = t.numel()
    if n == 0:
        return torch.empty_like(t)
    _ensure_device(t.device.index)

    order = None
    tc = t
    if t.dim() == 0:
        tc = t.reshape(1)  # metadata-only
    elif t.stride(-1) != 1:
        cand = [i for i in range(t.dim()) if t.stride(i) == 1]
        if cand:
            order = [i for i in range(t.dim()) if i != cand[0]] + [cand[0]]
            tc = t.permute(order)

    if tc.stride(-1) != 1:
        # No stride-1 axis: no dtype view exists, address through raw pointers.
        dims = _reduce_dims(t.shape, t.stride())
        out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        S = [1] * _RANK_MAX
        T = [1] * _RANK_MAX
        for k, (s, st) in enumerate(list(reversed(dims))[:_RANK_MAX]):
            S[k] = s
            T[k] = st
        _atan_cplx_ptr_kernel[(triton.cdiv(n, BLOCK),)](
            t.data_ptr(),
            out.data_ptr(),
            n,
            *S,
            *T,
            RANK=min(len(dims), _RANK_MAX),
            FP64=fp64,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
        return out

    xv = tc.view(fdt)
    dims = _reduce_dims(tc.shape, tc.stride())
    out = torch.empty_like(tc)
    ov = out.view(fdt)
    if len(dims) == 1 and dims[0][1] == 1:
        _atan_flat_kernel[(triton.cdiv(n, BLOCK),)](
            xv,
            ov,
            n,
            FP64=fp64,
            COMPLEX=True,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
    else:
        inner = -1
        for k, (_, st) in enumerate(dims):
            if st == 1:
                inner = k
                break
        if (
            inner >= 0
            and len(dims) - 1 <= _RANK_MAX
            and tuple(out.stride()) == tuple(tc.stride())
        ):
            _launch_layout(xv, ov, dims, inner, n, fp64, True)
        else:
            if not out.is_contiguous():
                out = torch.empty(tc.shape, dtype=tc.dtype, device=tc.device)
                ov = out.view(fdt)
            _launch_strided(xv, ov, dims, n, fp64, True)
    if order is not None:
        inv = [order.index(i) for i in range(len(order))]
        out = out.permute(inv)
    if t.dim() == 0:
        out = out.reshape(())
    return out


def _atan_one(t):
    dt = t.dtype
    if dt.is_floating_point and t.is_contiguous():
        # hot path: dense floating tensor, one vectorized launch
        n = t.numel()
        out = torch.empty_like(t)
        if n:
            _ensure_device(t.device.index)
            blk = BLOCK16 if (dt is torch.float16 or dt is torch.bfloat16) else BLOCK
            _atan_flat_kernel[((n + blk - 1) // blk,)](
                t,
                out,
                n,
                FP64=(dt is torch.float64),
                COMPLEX=False,
                BLOCK=blk,
                num_warps=NUM_WARPS,
            )
        return out
    return _atan_generic(t)


def _atan_generic(t):
    if t.dtype.is_complex:
        return _atan_complex_one(t)
    n = t.numel()
    odt = _out_dtype(t.dtype)
    if n == 0:
        return torch.empty_like(t, dtype=odt)
    _ensure_device(t.device.index)
    fp64 = odt is torch.float64
    x = t.view(torch.uint8) if t.dtype is torch.bool else t

    dims = _reduce_dims(t.shape, t.stride())
    if len(dims) == 1 and dims[0][1] == 1:
        out = torch.empty_like(t, dtype=odt)
        _atan_flat_kernel[(triton.cdiv(n, BLOCK),)](
            x,
            out,
            n,
            FP64=fp64,
            COMPLEX=False,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
        return out

    out = torch.empty_like(t, dtype=odt)
    inner = -1
    for k, (_, st) in enumerate(dims):
        if st == 1:
            inner = k
            break
    if (
        inner >= 0
        and len(dims) - 1 <= _RANK_MAX
        and tuple(out.stride()) == tuple(t.stride())
    ):
        _launch_layout(x, out, dims, inner, n, fp64, False)
        return out
    if not out.is_contiguous():
        out = torch.empty(t.shape, dtype=odt, device=t.device)
    _launch_strided(x, out, dims, n, fp64, False)
    return out


def _try_fused(items):
    """One launch + one shared allocation for a homogeneous contiguous chunk.

    Returns the list of output views, or None when the chunk is not eligible.
    """
    m = len(items)
    if m < 2 or m > _FUSED_MAX:
        return None
    t0 = items[0]
    dt = t0.dtype
    n = t0.numel()
    if n == 0 or dt.is_complex or dt is torch.bool:
        return None
    odt = dt if dt.is_floating_point else torch.get_default_dtype()
    if n * m * odt.itemsize > _FUSED_MAX_BYTES:
        return None
    shape = t0.shape
    dev = t0.device
    for t in items:
        if (
            t.dtype is not dt
            or t.shape != shape
            or t.device != dev
            or not t.is_contiguous()
        ):
            return None
    _ensure_device(dev.index)
    blk = BLOCK16 if (dt is torch.float16 or dt is torch.bfloat16) else BLOCK
    nb = (n + blk - 1) // blk
    buf = torch.empty((m,) + shape, dtype=odt, device=dev)
    outs = list(buf.unbind(0))
    if m == _FUSED_MAX:
        xs = items
    else:
        xs = list(items) + [items[0]] * (_FUSED_MAX - m)
    _atan_fused_kernel[(nb * m,)](
        *xs,
        buf,
        n,
        FP64=(dt is torch.float64),
        BLOCK=blk,
        num_warps=NUM_WARPS,
    )
    return outs


def foreach_atan(self):
    if not self:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if not isinstance(self, (list, tuple)):
        self = list(self)
    m = len(self)
    if m == 1:
        return [_atan_one(self[0])]
    outs = _try_fused(self)
    if outs is not None:
        return outs
    if m <= _FUSED_MAX:
        return [_atan_one(t) for t in self]
    res = []
    for i in range(0, m, _FUSED_MAX):
        chunk = self[i : i + _FUSED_MAX]
        outs = _try_fused(chunk)
        if outs is None:
            res.extend([_atan_one(t) for t in chunk])
        else:
            res.extend(outs)
    return res
