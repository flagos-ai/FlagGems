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

"""masked_select_backward — Triton implementation.

Reference semantics (verified against aten::masked_select_backward on the target):
    out = empty(broadcast(input.shape, mask.shape)) filled with the flat logical
    sequence grad[0..K) scattered into the positions where the (broadcast) bool
    mask is True, traversed in logical row-major order and zero elsewhere.
    * mask must be bool                         -> RuntimeError
    * grad.dtype must equal input.dtype         -> RuntimeError
    * all tensors must live on one device       -> RuntimeError
    * input/mask must be broadcast-compatible   -> RuntimeError
    * fewer grad elements than mask ones        -> RuntimeError
    * output strides follow "dense strides in the input's stride-ascending axis
      order" when out.shape == input.shape, else standard contiguous strides
      (probed on 17 layouts: contiguous, transposed dense/non-dense, permuted,
      strided slice, channels-last 3d/4d, size-1 dims, broadcast, offset).

Only mask and grad are read; `input` contributes shape/layout/dtype/device.
"""

import torch
import triton
import triton.language as tl

_MAXD = 6
_BLOCK = 4096
_SCAN_BLOCK = 1024
_WARPS = 8
_SMALL_N = 8192

_Z6 = (1, 1, 1, 1, 1, 1)
_Z6S = (0, 0, 0, 0, 0, 0)

_DT_NAMES = {
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

_SHORT_GRAD = "Number of elements of source < number of ones in mask"

# The reference raises when grad.numel() is smaller than the number of mask ones,
# which is only known on the device.  A DTOH copy costs ~19us per call on this
# target, so the kernels publish the count straight into pinned host-mapped
# memory (verified reachable from a Triton store) and the host polls a numpy view
# of that word.  The word is a scratch slot: it is re-armed with -1 before every
# launch and always drained before the next call, so no value is reused.
_PUB = {}
_SPIN_LIMIT = 200000


def _pub_word(device):
    ent = _PUB.get(device.index)
    if ent is None:
        t = torch.empty(1, dtype=torch.int32, pin_memory=True)
        try:
            view = t.numpy()
        except Exception:  # pragma: no cover - numpy always present with torch
            view = t
        ent = (t, view)
        _PUB[device.index] = ent
    return ent


def _desc(shape, stride):
    """Collapse (shape, stride) into (flat_stride, nd, sizes6, strides6).

    ``flat_stride >= 0`` means logical flat index i maps to ``i * flat_stride``
    with no division.  Otherwise dims are innermost-first and walked with
    successive ``%`` / ``//``.
    """
    dims = [(s, st) for s, st in zip(shape, stride) if s != 1]
    merged = []
    for s, st in dims:
        if merged and merged[-1][1] == st * s:
            merged[-1] = (merged[-1][0] * s, st)
        else:
            merged.append((s, st))
    if not merged:
        return 0, 1, _Z6, _Z6S
    if len(merged) == 1:
        return merged[0][1], 1, _Z6, _Z6S
    merged.reverse()
    n = len(merged)
    if n > _MAXD:
        raise RuntimeError(
            "masked_select_backward: collapsed layout rank exceeds %d" % _MAXD
        )
    sizes = tuple(s for s, _ in merged) + (1,) * (_MAXD - n)
    strides = tuple(st for _, st in merged) + (0,) * (_MAXD - n)
    return -1, n, sizes, strides


def _dense_strides(shape, order):
    st = [0] * len(shape)
    acc = 1
    for d in order:
        st[d] = acc
        acc *= shape[d]
    return tuple(st)


def _out_layout(shape, in_stride):
    """Dense strides following the input's stride-ascending axis order."""
    order = sorted(range(len(shape)), key=lambda i: (in_stride[i], -i))
    return _dense_strides(shape, order)


@triton.jit
def _sb_addr(
    i,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    FLAT: tl.constexpr,
    ND: tl.constexpr,
):
    if FLAT >= 0:
        a = i * FLAT
    else:
        rem = i
        a = (rem % S0) * T0
        rem = rem // S0
        if ND > 1:
            a += (rem % S1) * T1
            rem = rem // S1
        if ND > 2:
            a += (rem % S2) * T2
            rem = rem // S2
        if ND > 3:
            a += (rem % S3) * T3
            rem = rem // S3
        if ND > 4:
            a += (rem % S4) * T4
            rem = rem // S4
        if ND > 5:
            a += (rem % S5) * T5
    return a


# ---------------------------------------------------------------- fast kernels
# Used when mask/out are contiguous (logical order == memory order) and the
# dtype is not complex.  Only flat strides are needed, so the argument list
# stays short and the launch is cheap.


@triton.jit
def _sb_count_kernel(mask_ptr, cnt_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    m = tl.load(mask_ptr + offs, mask=valid, other=0)
    tl.store(cnt_ptr + pid, tl.sum((m != 0).to(tl.int32), axis=0))


@triton.jit
def _sb_scan_kernel(cnt_ptr, off_ptr, total_ptr, P, BLOCK: tl.constexpr):
    carry = 0
    for start in range(0, P, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        valid = idx < P
        v = tl.load(cnt_ptr + idx, mask=valid, other=0)
        inc = tl.cumsum(v, axis=0)
        tl.store(off_ptr + idx, inc - v + carry, mask=valid)
        carry += tl.sum(v, axis=0)
    tl.store(total_ptr, carry)


@triton.jit
def _sb_apply_kernel(mask_ptr, g_ptr, out_ptr, off_ptr, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    m = tl.load(mask_ptr + offs, mask=valid, other=0) != 0
    sel = m.to(tl.int32)
    dst = tl.load(off_ptr + pid) + tl.cumsum(sel, axis=0) - 1
    g = tl.load(g_ptr + dst, mask=m & valid & (dst < K), other=0)
    tl.store(out_ptr + offs, g, mask=valid)


@triton.jit
def _sb_small_kernel(mask_ptr, g_ptr, out_ptr, total_ptr, N, K, BLOCK: tl.constexpr):
    carry = 0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < N
        m = tl.load(mask_ptr + offs, mask=valid, other=0) != 0
        sel = m.to(tl.int32)
        dst = carry + tl.cumsum(sel, axis=0) - 1
        g = tl.load(g_ptr + dst, mask=m & valid & (dst < K), other=0)
        tl.store(out_ptr + offs, g, mask=valid)
        carry += tl.sum(sel, axis=0)
    tl.store(total_ptr, carry)


# ------------------------------------------------------------- general kernel
# Handles arbitrary strides/broadcast masks, non-contiguous outputs and complex
# dtypes (through real views).  Single CTA with a tile loop: layout-only cases.


@triton.jit
def _sb_gen_kernel(
    mask_ptr,
    g_ptr,
    gi_ptr,
    out_ptr,
    total_ptr,
    N,
    K,
    MS0,
    MS1,
    MS2,
    MS3,
    MS4,
    MS5,
    MT0,
    MT1,
    MT2,
    MT3,
    MT4,
    MT5,
    MFLAT: tl.constexpr,
    MND: tl.constexpr,
    GS0,
    GS1,
    GS2,
    GS3,
    GS4,
    GS5,
    GT0,
    GT1,
    GT2,
    GT3,
    GT4,
    GT5,
    GFLAT: tl.constexpr,
    GND: tl.constexpr,
    OS0,
    OS1,
    OS2,
    OS3,
    OS4,
    OS5,
    OT0,
    OT1,
    OT2,
    OT3,
    OT4,
    OT5,
    OFLAT: tl.constexpr,
    OND: tl.constexpr,
    REP: tl.constexpr,
    NEGI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    carry = 0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < N
        ma = _sb_addr(
            offs, MS0, MS1, MS2, MS3, MS4, MS5, MT0, MT1, MT2, MT3, MT4, MT5, MFLAT, MND
        )
        m = tl.load(mask_ptr + ma, mask=valid, other=0) != 0
        sel = m.to(tl.int32)
        dst = carry + tl.cumsum(sel, axis=0) - 1
        pred = m & valid & (dst < K)
        ga = _sb_addr(
            dst, GS0, GS1, GS2, GS3, GS4, GS5, GT0, GT1, GT2, GT3, GT4, GT5, GFLAT, GND
        )
        if REP == 1:
            oa = _sb_addr(
                offs,
                OS0,
                OS1,
                OS2,
                OS3,
                OS4,
                OS5,
                OT0,
                OT1,
                OT2,
                OT3,
                OT4,
                OT5,
                OFLAT,
                OND,
            )
            g = tl.load(g_ptr + ga, mask=pred, other=0)
            tl.store(out_ptr + oa, g, mask=valid)
        else:
            oa = _sb_addr(
                offs * 2,
                OS0,
                OS1,
                OS2,
                OS3,
                OS4,
                OS5,
                OT0,
                OT1,
                OT2,
                OT3,
                OT4,
                OT5,
                OFLAT,
                OND,
            )
            gr = tl.load(g_ptr + ga, mask=pred, other=0)
            gi = tl.load(gi_ptr + ga, mask=pred, other=0)
            if NEGI == 1:
                gi = -gi
            tl.store(out_ptr + oa, gr, mask=valid)
            tl.store(out_ptr + oa + 1, gi, mask=valid)
        carry += tl.sum(sel, axis=0)
    tl.store(total_ptr, carry)


def masked_select_backward(grad, input, mask):
    ishape = input.shape
    if mask.shape == ishape:
        shape = ishape
        same_shape = True
    else:
        shape = torch.broadcast_shapes(ishape, mask.shape)
        same_shape = shape == ishape
    if mask.dtype is not torch.bool:
        raise RuntimeError(
            "masked_scatter_ only supports boolean masks, but got mask with dtype "
            + _DT_NAMES.get(mask.dtype, str(mask.dtype))
        )
    if grad.dtype is not input.dtype:
        raise RuntimeError(
            "masked_scatter_: expected self and source to have same dtypes but got "
            + _DT_NAMES.get(input.dtype, str(input.dtype))
            + " and "
            + _DT_NAMES.get(grad.dtype, str(grad.dtype))
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("masked_select_backward: tensors must be on the same device")

    N = shape.numel()
    if N == 0:
        return torch.empty(shape, dtype=input.dtype, device=input.device)

    if same_shape and not input.is_contiguous():
        out = torch.empty_strided(
            shape,
            _out_layout(shape, input.stride()),
            dtype=input.dtype,
            device=input.device,
        )
    else:
        out = torch.empty(shape, dtype=input.dtype, device=input.device)

    mask_b = mask if mask.shape == shape else mask.expand(shape)
    K = grad.numel()
    pub, npv = _pub_word(out.device)
    npv[0] = -1

    if (
        (not input.is_complex())
        and mask_b.is_contiguous()
        and grad.is_contiguous()
        and out.is_contiguous()
    ):
        # fast path: every tensor is traversed in memory order
        if N <= _SMALL_N:
            _sb_small_kernel[(1,)](
                mask_b,
                grad,
                out,
                pub,
                N,
                K,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        else:
            P = (N + _BLOCK - 1) // _BLOCK
            buf = torch.empty(2 * P, dtype=torch.int32, device=out.device)
            counts = buf[:P]
            offsets = buf[P:]
            _sb_count_kernel[(P,)](mask_b, counts, N, BLOCK=_BLOCK, num_warps=_WARPS)
            _sb_scan_kernel[(1,)](
                counts,
                offsets,
                pub,
                P,
                BLOCK=_SCAN_BLOCK,
                num_warps=4,
            )
            _sb_apply_kernel[(P,)](
                mask_b,
                grad,
                out,
                offsets,
                N,
                K,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
    else:
        # general path: strided/broadcast mask, strided output, complex dtypes
        mflat, mnd, msz, mst = _desc(mask_b.shape, mask_b.stride())
        rep = 1
        negi = 0
        if input.is_complex():
            grad_r = grad.real
            grad_i = grad.imag
            out_store = torch.view_as_real(out)
            rep = 2
            if grad.is_conj():
                negi = 1
        else:
            grad_r = grad
            grad_i = grad
            out_store = out
        gflat, gnd, gsz, gst = _desc(grad_r.shape, grad_r.stride())
        oflat, ond, osz, ost = _desc(out_store.shape, out_store.stride())
        _sb_gen_kernel[(1,)](
            mask_b,
            grad_r,
            grad_i,
            out_store,
            pub,
            N,
            K,
            msz[0],
            msz[1],
            msz[2],
            msz[3],
            msz[4],
            msz[5],
            mst[0],
            mst[1],
            mst[2],
            mst[3],
            mst[4],
            mst[5],
            mflat,
            mnd,
            gsz[0],
            gsz[1],
            gsz[2],
            gsz[3],
            gsz[4],
            gsz[5],
            gst[0],
            gst[1],
            gst[2],
            gst[3],
            gst[4],
            gst[5],
            gflat,
            gnd,
            osz[0],
            osz[1],
            osz[2],
            osz[3],
            osz[4],
            osz[5],
            ost[0],
            ost[1],
            ost[2],
            ost[3],
            ost[4],
            ost[5],
            oflat,
            ond,
            rep,
            negi,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )

    # Wait for the published count (host-mapped store, no DTOH copy) and apply
    # the reference's element-count validation.
    v = -1
    spins = 0
    while v < 0:
        v = npv[0]
        spins += 1
        if spins > _SPIN_LIMIT:
            torch.cuda.current_stream().synchronize()
            v = int(pub[0])
            break
    if K < N and v > K:
        raise RuntimeError(_SHORT_GRAD)
    return out
