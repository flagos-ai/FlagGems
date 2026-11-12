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

"""Triton implementation of aten::_foreach_log1p (out-of-place, value-returning).

Reference semantics measured on the target (torch 2.10.0+metax3.8.1.0):
  * floating dtypes keep their dtype, fp16/bf16 computed in fp32 opmath
  * integer / bool inputs are promoted to the default float dtype
  * complex64/complex128 are supported
  * outputs are freshly allocated; noncontiguous inputs keep their strides
  * the return value is a tuple
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 1024
# 16-bit dtypes need twice the elements per thread to reach 128-bit accesses.
_BLOCK_HALF = 2048
_NUM_WARPS = 4
_MAX_DIMS = 8
_MAX_SLOTS = 16
# Lists of large tensors are launch-insensitive and measured faster with
# per-tensor launches, so only short/small tensors are fused.
_FUSE_MAX_NUMEL = 1 << 20
_I64_LIMIT = 2**31 - 1


# ----------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------
@triton.jit
def _fast_log1p(x):
    """log1p via one MUFU lg2.approx: log1p(x) = log(1 + x).

    The single 1+x add keeps the absolute error versus true log1p near 2e-7
    (far below fp16/bf16 storage precision), and the zero case is selected
    explicitly so exact zeros keep their sign. lg2.approx yields -inf at
    1+x == 0 and NaN for 1+x < 0, matching log1p's domain behaviour.
    """
    xf = x.to(tl.float32)
    return tl.where(xf == 0.0, xf, libdevice.fast_logf(1.0 + xf))


@triton.jit
def _log1p_flat_f32(in_ptr, out_ptr, numel, BLOCK: tl.constexpr, I64: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    y = _fast_log1p(x)
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log1p_flat_f64(in_ptr, out_ptr, numel, BLOCK: tl.constexpr, I64: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = libdevice.log1p(x)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _slot_flat(ip, op, offs, n, F64: tl.constexpr):
    mask = offs < n
    if F64:
        x = tl.load(ip + offs, mask=mask, other=0.0)
        y = libdevice.log1p(x)
    else:
        x = tl.load(ip + offs, mask=mask, other=0)
        y = _fast_log1p(x)
    tl.store(op + offs, y.to(op.dtype.element_ty), mask=mask)


@triton.jit
def _log1p_multi(
    ip0,
    ip1,
    ip2,
    ip3,
    ip4,
    ip5,
    ip6,
    ip7,
    ip8,
    ip9,
    ip10,
    ip11,
    ip12,
    ip13,
    ip14,
    ip15,
    op0,
    op1,
    op2,
    op3,
    op4,
    op5,
    op6,
    op7,
    op8,
    op9,
    op10,
    op11,
    op12,
    op13,
    op14,
    op15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    be1,
    be2,
    be3,
    be4,
    be5,
    be6,
    be7,
    be8,
    be9,
    be10,
    be11,
    be12,
    be13,
    be14,
    be15,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
):
    """One launch for up to _MAX_SLOTS contiguous tensors of equal dtype.

    Program id p is mapped to the tensor slot owning block p through the
    cumulative block-end list; the per-tensor block index is p - block_start.
    be{k} is the last block (exclusive) of slot k-1.
    """
    pid = tl.program_id(0)
    slot = 0
    if pid >= be1:
        slot = 1
    if pid >= be2:
        slot = 2
    if pid >= be3:
        slot = 3
    if pid >= be4:
        slot = 4
    if pid >= be5:
        slot = 5
    if pid >= be6:
        slot = 6
    if pid >= be7:
        slot = 7
    if pid >= be8:
        slot = 8
    if pid >= be9:
        slot = 9
    if pid >= be10:
        slot = 10
    if pid >= be11:
        slot = 11
    if pid >= be12:
        slot = 12
    if pid >= be13:
        slot = 13
    if pid >= be14:
        slot = 14
    if pid >= be15:
        slot = 15
    ar = tl.arange(0, BLOCK)
    if slot == 0:
        _slot_flat(ip0, op0, pid * BLOCK + ar, n0, F64=F64)
    elif slot == 1:
        _slot_flat(ip1, op1, (pid - be1) * BLOCK + ar, n1, F64=F64)
    elif slot == 2:
        _slot_flat(ip2, op2, (pid - be2) * BLOCK + ar, n2, F64=F64)
    elif slot == 3:
        _slot_flat(ip3, op3, (pid - be3) * BLOCK + ar, n3, F64=F64)
    elif slot == 4:
        _slot_flat(ip4, op4, (pid - be4) * BLOCK + ar, n4, F64=F64)
    elif slot == 5:
        _slot_flat(ip5, op5, (pid - be5) * BLOCK + ar, n5, F64=F64)
    elif slot == 6:
        _slot_flat(ip6, op6, (pid - be6) * BLOCK + ar, n6, F64=F64)
    elif slot == 7:
        _slot_flat(ip7, op7, (pid - be7) * BLOCK + ar, n7, F64=F64)
    elif slot == 8:
        _slot_flat(ip8, op8, (pid - be8) * BLOCK + ar, n8, F64=F64)
    elif slot == 9:
        _slot_flat(ip9, op9, (pid - be9) * BLOCK + ar, n9, F64=F64)
    elif slot == 10:
        _slot_flat(ip10, op10, (pid - be10) * BLOCK + ar, n10, F64=F64)
    elif slot == 11:
        _slot_flat(ip11, op11, (pid - be11) * BLOCK + ar, n11, F64=F64)
    elif slot == 12:
        _slot_flat(ip12, op12, (pid - be12) * BLOCK + ar, n12, F64=F64)
    elif slot == 13:
        _slot_flat(ip13, op13, (pid - be13) * BLOCK + ar, n13, F64=F64)
    elif slot == 14:
        _slot_flat(ip14, op14, (pid - be14) * BLOCK + ar, n14, F64=F64)
    else:
        _slot_flat(ip15, op15, (pid - be15) * BLOCK + ar, n15, F64=F64)


@triton.jit
def _log1p_strided(
    in_ptr,
    out_ptr,
    numel,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    si0,
    si1,
    si2,
    si3,
    si4,
    si5,
    si6,
    si7,
    so0,
    so1,
    so2,
    so3,
    so4,
    so5,
    so6,
    so7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < numel
    idx = offs
    if I64:
        ioff = tl.zeros([BLOCK], tl.int64)
        ooff = tl.zeros([BLOCK], tl.int64)
    else:
        ioff = tl.zeros([BLOCK], tl.int32)
        ooff = tl.zeros([BLOCK], tl.int32)
    if NDIM > 0:
        q = idx // n0
        r = idx - q * n0
        ioff += r * si0
        ooff += r * so0
        idx = q
    if NDIM > 1:
        q = idx // n1
        r = idx - q * n1
        ioff += r * si1
        ooff += r * so1
        idx = q
    if NDIM > 2:
        q = idx // n2
        r = idx - q * n2
        ioff += r * si2
        ooff += r * so2
        idx = q
    if NDIM > 3:
        q = idx // n3
        r = idx - q * n3
        ioff += r * si3
        ooff += r * so3
        idx = q
    if NDIM > 4:
        q = idx // n4
        r = idx - q * n4
        ioff += r * si4
        ooff += r * so4
        idx = q
    if NDIM > 5:
        q = idx // n5
        r = idx - q * n5
        ioff += r * si5
        ooff += r * so5
        idx = q
    if NDIM > 6:
        q = idx // n6
        r = idx - q * n6
        ioff += r * si6
        ooff += r * so6
        idx = q
    if NDIM > 7:
        q = idx // n7
        r = idx - q * n7
        ioff += r * si7
        ooff += r * so7
        idx = q
    x = tl.load(in_ptr + ioff, mask=mask, other=0)
    if F64:
        y = libdevice.log1p(x)
    else:
        y = _fast_log1p(x)
    tl.store(out_ptr + ooff, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log1p_cplx(
    in_re,
    in_im,
    out_re,
    out_im,
    numel,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    si0,
    si1,
    si2,
    si3,
    si4,
    si5,
    si6,
    si7,
    so0,
    so1,
    so2,
    so3,
    so4,
    so5,
    so6,
    so7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < numel
    idx = offs
    if I64:
        ioff = tl.zeros([BLOCK], tl.int64)
        ooff = tl.zeros([BLOCK], tl.int64)
    else:
        ioff = tl.zeros([BLOCK], tl.int32)
        ooff = tl.zeros([BLOCK], tl.int32)
    if NDIM > 0:
        q = idx // n0
        r = idx - q * n0
        ioff += r * si0
        ooff += r * so0
        idx = q
    if NDIM > 1:
        q = idx // n1
        r = idx - q * n1
        ioff += r * si1
        ooff += r * so1
        idx = q
    if NDIM > 2:
        q = idx // n2
        r = idx - q * n2
        ioff += r * si2
        ooff += r * so2
        idx = q
    if NDIM > 3:
        q = idx // n3
        r = idx - q * n3
        ioff += r * si3
        ooff += r * so3
        idx = q
    if NDIM > 4:
        q = idx // n4
        r = idx - q * n4
        ioff += r * si4
        ooff += r * so4
        idx = q
    if NDIM > 5:
        q = idx // n5
        r = idx - q * n5
        ioff += r * si5
        ooff += r * so5
        idx = q
    if NDIM > 6:
        q = idx // n6
        r = idx - q * n6
        ioff += r * si6
        ooff += r * so6
        idx = q
    if NDIM > 7:
        q = idx // n7
        r = idx - q * n7
        ioff += r * si7
        ooff += r * so7
        idx = q
    a = tl.load(in_re + ioff, mask=mask, other=0.0)
    b = tl.load(in_im + ioff, mask=mask, other=0.0)
    if F64:
        a = a.to(tl.float64)
        b = b.to(tl.float64)
    else:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    u = a + 1.0
    m = u * u + b * b
    yr = 0.5 * libdevice.log(m)
    yi = libdevice.atan2(b, u)
    tl.store(out_re + ooff, yr.to(out_re.dtype.element_ty), mask=mask)
    tl.store(out_im + ooff, yi.to(out_im.dtype.element_ty), mask=mask)


# ----------------------------------------------------------------------------
# host helpers
# ----------------------------------------------------------------------------
def _result_dtype(dtype):
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return torch.get_default_dtype()


def _coalesce(t, out):
    """Jointly coalesce (input, output) dims; returns outer->inner triples."""
    dims = []
    for d in range(t.dim()):
        n = t.shape[d]
        if n == 1:
            continue
        dims.append([n, t.stride(d), out.stride(d)])
    i = 0
    while i < len(dims) - 1:
        a = dims[i]
        b = dims[i + 1]
        if a[1] == b[1] * b[0] and a[2] == b[2] * b[0]:
            a[0] *= b[0]
            del dims[i + 1]
        else:
            i += 1
    return dims


def _dim_args(dims):
    """Pad the (outer->inner) dim triples into _MAX_DIMS kernel slots."""
    ns = [1] * _MAX_DIMS
    sis = [0] * _MAX_DIMS
    sos = [0] * _MAX_DIMS
    nd = len(dims)
    for k in range(nd):
        n, si, so = dims[nd - 1 - k]
        ns[k] = n
        sis[k] = si
        sos[k] = so
    return nd, ns, sis, sos


def _launch(t, out, numel):
    if numel == 0:
        return
    i64 = numel > _I64_LIMIT
    dtype = t.dtype
    if dtype.is_complex:
        in_re = t.real
        in_im = t.imag
        out_re = out.real
        out_im = out.imag
        dims = _coalesce(in_re, out_re)
        nd, ns, sis, sos = _dim_args(dims)
        grid = (triton.cdiv(numel, _BLOCK),)
        _log1p_cplx[grid](
            in_re,
            in_im,
            out_re,
            out_im,
            numel,
            ns[0],
            ns[1],
            ns[2],
            ns[3],
            ns[4],
            ns[5],
            ns[6],
            ns[7],
            sis[0],
            sis[1],
            sis[2],
            sis[3],
            sis[4],
            sis[5],
            sis[6],
            sis[7],
            sos[0],
            sos[1],
            sos[2],
            sos[3],
            sos[4],
            sos[5],
            sos[6],
            sos[7],
            NDIM=nd,
            BLOCK=_BLOCK,
            I64=i64,
            F64=(dtype == torch.complex128),
            num_warps=_NUM_WARPS,
        )
        return
    if t.is_contiguous() and out.is_contiguous():
        blk = (
            _BLOCK_HALF
            if (dtype is torch.float16 or dtype is torch.bfloat16)
            else _BLOCK
        )
        grid = (triton.cdiv(numel, blk),)
        kern = _log1p_flat_f64 if dtype == torch.float64 else _log1p_flat_f32
        kern[grid](t, out, numel, blk, i64, num_warps=_NUM_WARPS)
        return
    dims = _coalesce(t, out)
    nd, ns, sis, sos = _dim_args(dims)
    grid = (triton.cdiv(numel, _BLOCK),)
    _log1p_strided[grid](
        t,
        out,
        numel,
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        ns[4],
        ns[5],
        ns[6],
        ns[7],
        sis[0],
        sis[1],
        sis[2],
        sis[3],
        sis[4],
        sis[5],
        sis[6],
        sis[7],
        sos[0],
        sos[1],
        sos[2],
        sos[3],
        sos[4],
        sos[5],
        sos[6],
        sos[7],
        NDIM=nd,
        BLOCK=_BLOCK,
        I64=i64,
        F64=(dtype == torch.float64),
        num_warps=_NUM_WARPS,
    )


def _launch_multi(items, f64):
    """Fuse contiguous same-dtype tensors: one launch per up-to-16 chunk."""
    for c in range(0, len(items), _MAX_SLOTS):
        chunk = items[c : c + _MAX_SLOTS]
        ins = []
        outs = []
        ns = [0] * _MAX_SLOTS
        bes = [0] * _MAX_SLOTS
        total = 0
        for k, (t, o, numel) in enumerate(chunk):
            total += triton.cdiv(numel, _BLOCK)
            ins.append(t)
            outs.append(o)
            ns[k] = numel
            bes[k] = total
        for k in range(len(chunk), _MAX_SLOTS):
            bes[k] = total
            ins.append(chunk[0][0])
            outs.append(chunk[0][1])
        _log1p_multi[(total,)](
            *ins,
            *outs,
            *ns,
            *bes[: _MAX_SLOTS - 1],
            BLOCK=_BLOCK,
            F64=f64,
            num_warps=_NUM_WARPS,
        )


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
_EMPTY_LIKE = torch.empty_like
_CDIV = triton.cdiv


def _alloc(t):
    """Output tensor: same dtype for floats/complex, default float otherwise."""
    dt = t.dtype
    if dt.is_floating_point or dt.is_complex:
        return _EMPTY_LIKE(t), dt
    odt = torch.get_default_dtype()
    return _EMPTY_LIKE(t, dtype=odt), odt


def foreach_log1p(self):
    if isinstance(self, torch.Tensor):
        self = [self]
    n = len(self)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if n == 1:
        t = self[0]
        out, odt = _alloc(t)
        numel = t.numel()
        dt = t.dtype
        if numel != 0:
            if (
                numel <= _I64_LIMIT
                and not dt.is_complex
                and t.is_contiguous()
                and out.is_contiguous()
            ):
                kern = _log1p_flat_f64 if dt is torch.float64 else _log1p_flat_f32
                blk = (
                    _BLOCK_HALF
                    if (dt is torch.float16 or dt is torch.bfloat16)
                    else _BLOCK
                )
                kern[((numel + blk - 1) // blk,)](
                    t, out, numel, blk, numel > _I64_LIMIT, num_warps=_NUM_WARPS
                )
            else:
                _launch(t, out, numel)
        return (out,)
    outs = [None] * n
    groups = {}
    for i in range(n):
        t = self[i]
        out, odt = _alloc(t)
        outs[i] = out
        numel = t.numel()
        if numel == 0:
            continue
        dt = t.dtype
        if (
            numel <= _FUSE_MAX_NUMEL
            and not dt.is_complex
            and t.is_contiguous()
            and out.is_contiguous()
        ):
            groups.setdefault((dt, odt), []).append((t, out, numel))
        else:
            _launch(t, out, numel)
    for (idt, _odt), items in groups.items():
        if len(items) == 1:
            _launch(items[0][0], items[0][1], items[0][2])
        else:
            _launch_multi(items, idt is torch.float64)
    return tuple(outs)
