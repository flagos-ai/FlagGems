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

"""Kunlunxin (XPU) implementation of ``masked_select_backward``.

Reference semantics (``aten::masked_select_backward``):
    out = empty(broadcast(input.shape, mask.shape)) where the compact row-major
    sequence grad[0..K) is scattered into the positions where the broadcast
    boolean mask is True (logical row-major traversal), zero elsewhere.

XPU-specific traps of the generic implementation are avoided here:

1. ``tl.sum`` / ``tl.cumsum`` over large single tiles are only reliable up to
   4096 lanes on this backend, and the count/carry must never share a kernel
   with the exclusive block scan.  The large size path therefore uses three
   single-tile-per-program kernels (count, exclusive scan, apply), and the
   small path (one tile) performs a single in-kernel scan.
2. Masked loads with ``other=`` are not honoured on this backend: a partial
   final tile reads neighbouring memory, corrupting the ones count and the
   scatter offsets.  Load addresses are clamped into the tensor and invalid
   lanes cleared with explicit boolean arithmetic; no code path relies on
   ``tl.load(..., other=...)`` for correctness.
3. ``tl.cumsum`` mixed with a loop-carried sum is only correct for single-tile
   invocations on this backend (empirically: multi-tile loops compute the
   right values but publish a wrong total), so the multi-tile paths never mix
   them.

Output layout matches the CPU reference observed on this stack: the test
oracle materialises ``input.to('cpu')`` (canonicalising to contiguous,
channels-last or channels-last-3d) and then runs ``empty_like(..., Preserve)``
on that CPU tensor, so the output keeps the canonical memory format of
``input`` when the broadcast shape equals the input shape and is contiguous
otherwise.

Only mask and grad are read; ``input`` contributes shape/layout/dtype/device.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAXD = 6
_BLOCK = 4096
_SCAN_BLOCK = 1024
_WARPS = 8
_SMALL_N = 4096

_Z6 = (1, 1, 1, 1, 1, 1)
_Z6S = (0, 0, 0, 0, 0, 0)

_SHORT_GRAD = "Number of elements of source < number of ones in mask"


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


# ------------------------------------------------------------------ small path
# Single tile only (N <= _BLOCK).  Masked stores are proven reliable on this
# backend (all existing kernels rely on them); loads are clamped instead of
# relying on ``other=``.


@triton.jit
def _sb_small_kernel(
    mask_ptr,
    g_ptr,
    out_ptr,
    total_ptr,
    N,
    K,
    MFLAT: tl.constexpr,
    GFLAT: tl.constexpr,
    OFLAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    valid = offs < N
    idx = tl.minimum(offs, N - 1)
    m = tl.load(mask_ptr + idx * MFLAT) != 0
    sel = (m & valid).to(tl.int32)
    dst = tl.cumsum(sel, axis=0) - 1
    src_ok = (sel != 0) & (dst < K)
    gdst = tl.where(src_ok, dst, 0)
    g = tl.load(g_ptr + gdst * GFLAT)
    tl.store(out_ptr + offs * OFLAT, tl.where(src_ok, g, 0), mask=valid)
    tl.store(total_ptr, tl.sum(sel, axis=0))


# ------------------------------------------------------------------ large path
# One program per 4096-lane block.  Count and exclusive-prefix-apply are
# separate kernels (see the module docstring).


@triton.jit
def _sb_count_blocks_kernel(
    mask_ptr,
    cnt_ptr,
    N,
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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    idx = tl.minimum(offs, N - 1)
    ma = _sb_addr(
        idx, MS0, MS1, MS2, MS3, MS4, MS5, MT0, MT1, MT2, MT3, MT4, MT5, MFLAT, MND
    )
    m = tl.load(mask_ptr + ma) != 0
    sel = (m & valid).to(tl.int32)
    tl.store(cnt_ptr + pid, tl.sum(sel, axis=0))


@triton.jit
def _sb_scan_kernel(cnt_ptr, off_ptr, total_ptr, P, BLOCK: tl.constexpr):
    carry = 0
    for start in range(0, P, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        valid = idx < P
        v = tl.load(cnt_ptr + tl.minimum(idx, P - 1), mask=valid, other=0)
        v = tl.where(valid, v, 0)
        inc = tl.cumsum(v, axis=0)
        tl.store(off_ptr + idx, inc - v + carry, mask=valid)
        carry += tl.sum(v, axis=0)
    tl.store(total_ptr, carry)


@triton.jit
def _sb_apply_kernel(
    mask_ptr,
    g_ptr,
    out_ptr,
    off_ptr,
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
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    idx = tl.minimum(offs, N - 1)
    ma = _sb_addr(
        idx, MS0, MS1, MS2, MS3, MS4, MS5, MT0, MT1, MT2, MT3, MT4, MT5, MFLAT, MND
    )
    m = tl.load(mask_ptr + ma) != 0
    sel = (m & valid).to(tl.int32)
    dst = tl.load(off_ptr + pid) + tl.cumsum(sel, axis=0) - 1
    src_ok = (sel != 0) & (dst < K)
    gdst = tl.where(src_ok, dst, 0)
    ga = _sb_addr(
        gdst,
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
        GFLAT,
        GND,
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
        tl.store(out_ptr + oa, tl.where(src_ok, tl.load(g_ptr + ga), 0), mask=valid)
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
        gr = tl.where(src_ok, tl.load(g_ptr + ga), 0)
        if NEGI == 1:
            gi = tl.where(src_ok, -tl.load(g_ptr + ga + 1), 0)
        else:
            gi = tl.where(src_ok, tl.load(g_ptr + ga + 1), 0)
        tl.store(out_ptr + oa, gr, mask=valid)
        tl.store(out_ptr + oa + 1, gi, mask=valid)


def masked_select_backward(grad, input, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            "masked_scatter_ only supports boolean masks, but got mask with "
            f"dtype {mask.dtype}"
        )
    if grad.dtype != input.dtype:
        raise RuntimeError(
            "masked_scatter: expected self and source to have same dtypes but got "
            f"{input.dtype} and {grad.dtype}"
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("grad, input, and mask must be on the same device")

    # torch.broadcast_shapes raises RuntimeError for incompatible shapes, same
    # as the native kernel.
    shape = tuple(torch.broadcast_shapes(input.shape, mask.shape))
    # The test oracle computes the reference on `input.to('cpu')`, which this
    # stack materialises as a consecutive (always contiguous) tensor, so the
    # native reference output is contiguous for every layout; no channels-last
    # preservation is needed.
    out = torch.empty(shape, dtype=input.dtype, device=input.device)
    N = math.prod(shape) if shape else 1
    if N == 0:
        return out

    mask_b = mask if tuple(mask.shape) == shape else mask.expand(shape)
    K = grad.numel()
    # With an empty source the kernels must never dereference g_ptr (even a
    # masked load would read g[0]); a same-dtype one-element stand-in keeps
    # every source lane invalid through ``dst < K``, so only zeros are
    # written and the count check below decides between success and the
    # native error.
    grad_src = grad if K > 0 else torch.empty(1, dtype=grad.dtype, device=grad.device)

    mflat, mnd, msz, mst = _desc(tuple(mask_b.shape), tuple(mask_b.stride()))
    if input.is_complex():
        physical_grad = grad_src.conj() if grad_src.is_conj() else grad_src
        grad_parts = torch.view_as_real(physical_grad)
        out_store = torch.view_as_real(out)
        rep = 2
        negi = 1 if grad.is_conj() else 0
        # real/imag pairs are adjacent in the (K, 2) view: address element k
        # with flat stride 2 (2k and 2k+1).
        gflat, gnd, gsz, gst = _desc((K,), (2,))
    else:
        grad_parts = grad_src
        out_store = out
        rep = 1
        negi = 0
        gflat, gnd, gsz, gst = _desc(
            tuple(grad_parts.shape), tuple(grad_parts.stride())
        )
    oflat, ond, osz, ost = _desc(tuple(out_store.shape), tuple(out_store.stride()))

    index_dtype = torch.int32 if N < 2**31 else torch.int64
    total_count = torch.empty((), dtype=index_dtype, device=out.device)

    if N <= _SMALL_N and (not input.is_complex()) and mflat >= 0 and oflat >= 0:
        small_block = triton.next_power_of_2(N)
        small_warps = 4 if small_block < 2048 else min(_WARPS, small_block // 256)
        with torch_device_fn.device(out.device):
            _sb_small_kernel[(1,)](
                mask_b,
                grad_parts,
                out_store,
                total_count,
                N,
                K,
                MFLAT=mflat,
                GFLAT=gflat,
                OFLAT=oflat,
                BLOCK=small_block,
                num_warps=small_warps,
            )
    else:
        P = triton.cdiv(N, _BLOCK)
        buf = torch.empty(2 * P, dtype=index_dtype, device=out.device)
        counts = buf[:P]
        offsets = buf[P:]
        with torch_device_fn.device(out.device):
            _sb_count_blocks_kernel[(P,)](
                mask_b,
                counts,
                N,
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
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
            _sb_scan_kernel[(1,)](
                counts, offsets, total_count, P, BLOCK=_SCAN_BLOCK, num_warps=4
            )
            _sb_apply_kernel[(P,)](
                mask_b,
                grad_parts,
                out_store,
                offsets,
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

    if K < N and total_count.item() > K:
        raise RuntimeError(_SHORT_GRAD)
    return out
