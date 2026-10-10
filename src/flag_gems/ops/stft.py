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

import logging
import math
import os
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.stft_fft import fft_frames

logger = logging.getLogger(__name__)

_FRAME_BYTES = 32 * 1024 * 1024
_WARNED = set()
_NV_TWIDDLE_LIMIT = 32
_NV_TWIDDLE_CACHE = OrderedDict()
_NV_TWIDDLE_LOCK = threading.RLock()
_NV_TWIDDLE_PID = os.getpid()
_REAL_DTYPES = (torch.float16, torch.float32, torch.float64)
_COMPLEX_DTYPES = (torch.complex32, torch.complex64, torch.complex128)


@triton.jit
def _pad_index(p, L: tl.constexpr, MODE: tl.constexpr):
    valid = (p >= 0) & (p < L)
    if MODE == "reflect":
        p = tl.where(p < 0, -p, p)
        p = tl.where(p >= L, 2 * L - 2 - p, p)
        valid = tl.full(p.shape, True, tl.int1)
    elif MODE == "replicate":
        p = tl.minimum(tl.maximum(p, 0), L - 1)
        valid = tl.full(p.shape, True, tl.int1)
    elif MODE == "circular":
        p = (p % L + L) % L
        valid = tl.full(p.shape, True, tl.int1)
    return p, valid


@triton.jit
def _load_pair(
    P,
    offset,
    mask,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    NEG: tl.constexpr = False,
):
    if COMPLEX:
        re = tl.load(P + 2 * offset, mask, other=0)
        im = tl.load(P + 2 * offset + 1, mask, other=0)
        if CONJ:
            im = -im
    else:
        re = tl.load(P + offset, mask, other=0)
        im = tl.full(offset.shape, 0, re.dtype)
    if NEG:
        re, im = -re, -im
    return re, im


@triton.jit
def _nv_twiddle_kernel(TW, N: tl.constexpr, FP64: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    angle = k.to(dtype) * (-6.283185307179586 / N)
    if FP64:
        c, s = libdevice.cos(angle), libdevice.sin(angle)
    else:
        c, s = tl.cos(angle), tl.sin(angle)
    tl.store(TW + 2 * k, c, k < N // 2)
    tl.store(TW + 2 * k + 1, s, k < N // 2)


def _nv_twiddle(n, fp64, input_device):
    # Check before acquiring a possibly inherited lock. CUDA state and cached
    # tensors cannot be reused in a fork child; a spawned process starts fresh.
    if os.getpid() != _NV_TWIDDLE_PID:
        raise RuntimeError("STFT CUDA FFT cannot be used after fork; use spawn instead")
    stream = torch.cuda.current_stream(input_device)
    capturing = torch.cuda.is_current_stream_capturing()
    key = (input_device.index, stream.cuda_stream, threading.get_ident(), n, fp64)
    with _NV_TWIDDLE_LOCK:
        entry = None if capturing else _NV_TWIDDLE_CACHE.get(key)
        if entry is None:
            twiddle = torch.empty(
                (n,),
                device=input_device,
                dtype=torch.float64 if fp64 else torch.float32,
            )
            _nv_twiddle_kernel[(triton.cdiv(n // 2, 256),)](twiddle, n, fp64, 256)
            if not capturing:
                # Retain the stream so its handle cannot be recycled while its
                # cached table survives. Eviction never synchronizes the device.
                _NV_TWIDDLE_CACHE[key] = (twiddle, stream)
                if len(_NV_TWIDDLE_CACHE) > _NV_TWIDDLE_LIMIT:
                    _NV_TWIDDLE_CACHE.popitem(last=False)
        else:
            twiddle = entry[0]
            _NV_TWIDDLE_CACHE.move_to_end(key)
        if not capturing:
            # A local reference survives eviction until the launch finishes;
            # allocator stream tracking protects asynchronous consumers after it.
            twiddle.record_stream(stream)
        # Capture always uses a graph-pool allocation and captures initialization.
        # The graph's private pool owns its storage lifetime, so later LRU
        # eviction or replay cannot refer to an ordinary cached allocation.
        return twiddle


@triton.jit
def _nv_butterfly(r, i, TW, N: tl.constexpr, HALF: tl.constexpr):
    ur, vr = tl.split(tl.permute(tl.reshape(r, (N // (2 * HALF), 2, HALF)), (0, 2, 1)))
    ui, vi = tl.split(tl.permute(tl.reshape(i, (N // (2 * HALF), 2, HALF)), (0, 2, 1)))
    j = tl.arange(0, HALF)[None, :]
    c = tl.load(TW + 2 * j * (N // (2 * HALF)))
    s = tl.load(TW + 2 * j * (N // (2 * HALF)) + 1)
    tr, ti = vr * c - vi * s, vr * s + vi * c
    r = tl.reshape(tl.permute(tl.join(ur + tr, ur - tr), (0, 2, 1)), (N,))
    i = tl.reshape(tl.permute(tl.join(ui + ti, ui - ti), (0, 2, 1)), (N,))
    return r, i


# Fuse framing and bounded radix-2 transforms to avoid host launch gaps.
# General lengths and large FFTs use the vendor engine below.
@triton.jit
def _fused_fft_kernel(
    X,
    W,
    Y,
    T: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    WIN: tl.constexpr,
    PAD: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    XC: tl.constexpr,
    WC: tl.constexpr,
    XCONJ: tl.constexpr,
    WCONJ: tl.constexpr,
    XNEG: tl.constexpr,
    WNEG: tl.constexpr,
    HAS_W: tl.constexpr,
    MODE: tl.constexpr,
    INDEX64: tl.constexpr,
    SCALE: tl.constexpr,
    FP64: tl.constexpr,
    F: tl.constexpr,
    LOG_N: tl.constexpr,
    NV: tl.constexpr = False,
    TW=None,
):
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    row = tl.program_id(0)
    if INDEX64:
        row = row.to(tl.int64)
    lane = tl.arange(0, N)
    k = tl.full((N,), 0, tl.int32)
    for bit in tl.static_range(LOG_N):
        k = k | (((lane >> bit) & 1) << (LOG_N - bit - 1))
    if INDEX64:
        k = k.to(tl.int64)
    p, valid = _pad_index(row % T * H + k - PAD, L, MODE)
    if L == 0:
        xr, xi = tl.full((N,), 0, dtype), tl.full((N,), 0, dtype)
    else:
        xr, xi = _load_pair(X, row // T * XS0 + p * XS1, valid, XC, XCONJ, XNEG)
        xr, xi = xr.to(dtype), xi.to(dtype)
    w = k - (N - WIN) // 2
    wm = (w >= 0) & (w < WIN)
    if HAS_W:
        wr, wi = _load_pair(W, w * WS, wm, WC, WCONJ, WNEG)
        wr, wi = wr.to(dtype), wi.to(dtype)
    else:
        wr, wi = wm.to(dtype), tl.full((N,), 0, dtype)
    if not HAS_W and WIN == N:
        r, i = xr, xi
    elif not XC and not WC:
        r, i = xr * wr, xi
    else:
        r, i = xr * wr - xi * wi, xr * wi + xi * wr
    if NV:
        for stage in tl.static_range(1, LOG_N + 1):
            r, i = _nv_butterfly(r, i, TW, N, 1 << (stage - 1))
    else:
        for stage in tl.static_range(1, LOG_N + 1):
            half = 1 << (stage - 1)
            j = lane & (half - 1)
            even = lane & ~half
            odd = even | half
            ur, ui = tl.gather(r, even, 0), tl.gather(i, even, 0)
            vr, vi = tl.gather(r, odd, 0), tl.gather(i, odd, 0)
            angle = j.to(dtype) * (-3.141592653589793 / half)
            if FP64:
                c, s = libdevice.cos(angle), libdevice.sin(angle)
            else:
                c, s = tl.cos(angle), tl.sin(angle)
            tr, ti = vr * c - vi * s, vr * s + vi * c
            lower = (lane & half) == 0
            r, i = tl.where(lower, ur + tr, ur - tr), tl.where(lower, ui + ti, ui - ti)
    tl.store(Y + 2 * (row * F + lane), r * SCALE, lane < F)
    tl.store(Y + 2 * (row * F + lane) + 1, i * SCALE, lane < F)


@libentry()
@triton.jit
def _zero_kernel(Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, 0, i < SIZE)


@libentry()
@triton.jit
def _frame_kernel(
    X,
    W,
    Y,
    START,
    ROWS,
    T: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    WIN: tl.constexpr,
    PAD: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    XC: tl.constexpr,
    WC: tl.constexpr,
    YC: tl.constexpr,
    XCONJ: tl.constexpr,
    WCONJ: tl.constexpr,
    XNEG: tl.constexpr,
    WNEG: tl.constexpr,
    HAS_W: tl.constexpr,
    MODE: tl.constexpr,
    FP64: tl.constexpr,
    INDEX64: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    FLAT_GRID: tl.constexpr = False,
):
    if FLAT_GRID:
        tiles: tl.constexpr = triton.cdiv(N * (2 if YC else 1), BLOCK)
        row, tile = tl.program_id(0) // tiles, tl.program_id(0) % tiles
    else:
        row, tile = tl.program_id(0), tl.program_id(1)
    if INDEX64:
        row, tile = row.to(tl.int64), tile.to(tl.int64)
    lane = tile * BLOCK + tl.arange(0, BLOCK)
    if YC:
        k = lane // 2
    else:
        k = lane
    i = row * N + k
    global_row = START + row
    b, t = global_row // T, global_row % T
    p, valid = _pad_index(t * H + k - PAD, L, MODE)
    acc: tl.constexpr = tl.float64 if FP64 else tl.float32
    if L == 0:
        xr, xi = tl.full((BLOCK,), 0, acc), tl.full((BLOCK,), 0, acc)
    else:
        xr, xi = _load_pair(X, b * XS0 + p * XS1, (k < N) & valid, XC, XCONJ, XNEG)
        xr, xi = xr.to(acc), xi.to(acc)
    w = k - (N - WIN) // 2
    wm = (w >= 0) & (w < WIN)
    if HAS_W:
        wr, wi = _load_pair(W, w * WS, wm, WC, WCONJ, WNEG)
        wr, wi = wr.to(acc), wi.to(acc)
    else:
        wr, wi = wm.to(acc), tl.full((BLOCK,), 0, acc)
    if not HAS_W and WIN == N:
        yr, yi = xr, xi
    elif not XC and not WC:
        yr, yi = xr * wr, xi
    else:
        yr, yi = xr * wr - xi * wi, xr * wi + xi * wr
    if SCALE != 1:
        yr, yi = yr * SCALE, yi * SCALE
    if YC:
        value = tl.where(lane % 2 == 0, yr, yi)
        tl.store(Y + row * (2 * N) + lane, value, lane < 2 * N)
    else:
        tl.store(Y + i, yr, k < N)


@libentry()
@triton.jit
def _spectrum_kernel(
    X,
    Y,
    START,
    ROWS,
    N: tl.constexpr,
    F: tl.constexpr,
    IN_F: tl.constexpr,
    SCALE: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK: tl.constexpr,
    FLAT_GRID: tl.constexpr = False,
):
    if FLAT_GRID:
        tiles: tl.constexpr = triton.cdiv(2 * F, BLOCK)
        row, tile = tl.program_id(0) // tiles, tl.program_id(0) % tiles
    else:
        row, tile = tl.program_id(0), tl.program_id(1)
    if INDEX64:
        row, tile = row.to(tl.int64), tile.to(tl.int64)
    lane = tile * BLOCK + tl.arange(0, BLOCK)
    f, imag = lane // 2, lane % 2
    mirror = f >= IN_F
    src = row * IN_F + tl.where(mirror, N - f, f)
    value = tl.load(X + 2 * src + imag, f < F, other=0)
    value = tl.where(mirror & (imag != 0), -value, value) * SCALE
    dst = (START + row) * (2 * F) + lane
    tl.store(Y + dst, value, lane < 2 * F)


@libentry()
@triton.jit
def _gradient_spectrum_kernel(
    G,
    Y,
    START,
    ROWS,
    T: tl.constexpr,
    N: tl.constexpr,
    F: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    CONJ: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, f = i // N, i % N
    r = START + row
    src = r // T * GS0 + r % T * GS1 + f * GS2
    re, im = _load_pair(G, src, (row < ROWS) & (f < F), True, CONJ, NEG)
    tl.store(Y + 2 * i, re, row < ROWS)
    tl.store(Y + 2 * i + 1, im, row < ROWS)


@libentry()
@triton.jit
def _overlap_kernel(
    G,
    W,
    DX,
    START,
    ROWS,
    B: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    WIN: tl.constexpr,
    LP: tl.constexpr,
    WS: tl.constexpr,
    WC: tl.constexpr,
    WCONJ: tl.constexpr,
    WNEG: tl.constexpr,
    HAS_W: tl.constexpr,
    XC: tl.constexpr,
    SCALE: tl.constexpr,
    FIRST: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    b, p = i // LP, i % LP
    acc: tl.constexpr = tl.float64 if FP64 else tl.float32
    re, im = tl.full((BLOCK,), 0, acc), tl.full((BLOCK,), 0, acc)
    # Input-centric overlap-add avoids floating-point atomic operations.
    last = p // H
    for overlap in range(triton.cdiv(N, H)):
        t = last - overlap
        k = p - t * H
        row = b * T + t - START
        w = k - (N - WIN) // 2
        mask = (b < B) & (t >= 0) & (t < T) & (k < N)
        mask = mask & (row >= 0) & (row < ROWS) & (w >= 0) & (w < WIN)
        gr = tl.load(G + 2 * (row * N + k), mask, other=0).to(acc)
        gi = tl.load(G + 2 * (row * N + k) + 1, mask, other=0).to(acc)
        if HAS_W:
            wr, wi = _load_pair(W, w * WS, mask, WC, WCONJ, WNEG)
            wr, wi = wr.to(acc), wi.to(acc)
        else:
            wr, wi = 1.0, 0.0
        re += gr * wr + gi * wi
        im += gi * wr - gr * wi
    if XC:
        if not FIRST:
            re += tl.load(DX + 2 * i, b < B, other=0) / SCALE
            im += tl.load(DX + 2 * i + 1, b < B, other=0) / SCALE
        tl.store(DX + 2 * i, re * SCALE, b < B)
        tl.store(DX + 2 * i + 1, im * SCALE, b < B)
    else:
        if not FIRST:
            re += tl.load(DX + i, b < B, other=0) / SCALE
        tl.store(DX + i, re * SCALE, b < B)


@libentry()
@triton.jit
def _unpad_kernel(
    G,
    DX,
    B: tl.constexpr,
    L: tl.constexpr,
    PAD: tl.constexpr,
    COMPLEX: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    C: tl.constexpr = 2 if COMPLEX else 1
    b, p, c = i // (L * C), (i // C) % L, i % C
    LP: tl.constexpr = L + 2 * PAD
    base = b * LP * C + c
    value = tl.load(G + base + (p + PAD) * C, b < B, other=0)
    if MODE == "reflect":
        left = PAD - p
        right = PAD + 2 * L - 2 - p
        value += tl.load(
            G + base + left * C, (b < B) & (p > 0) & (left >= 0) & (left < PAD), other=0
        )
        value += tl.load(
            G + base + right * C,
            (b < B) & (p < L - 1) & (right >= PAD + L) & (right < LP),
            other=0,
        )
    elif MODE == "circular":
        left, right = p + PAD - L, p + PAD + L
        value += tl.load(
            G + base + left * C, (b < B) & (left >= 0) & (left < PAD), other=0
        )
        value += tl.load(
            G + base + right * C, (b < B) & (right >= PAD + L) & (right < LP), other=0
        )
    elif MODE == "replicate":
        for j in range(PAD):
            value += tl.load(G + base + j * C, (b < B) & (p == 0), other=0)
            value += tl.load(
                G + base + (PAD + L + j) * C, (b < B) & (p == L - 1), other=0
            )
    tl.store(DX + i, value, b < B)


@libentry()
@triton.jit
def _window_gradient_kernel(
    X,
    G,
    DW,
    START,
    ROWS,
    T: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    WIN: tl.constexpr,
    PAD: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XC: tl.constexpr,
    XCONJ: tl.constexpr,
    XNEG: tl.constexpr,
    WC: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
    FIRST: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    w = tl.program_id(0)
    k = w + (N - WIN) // 2
    r = tl.arange(0, BLOCK).to(tl.int64)
    acc: tl.constexpr = tl.float64 if FP64 else tl.float32
    re, im = tl.full((BLOCK,), 0, acc), tl.full((BLOCK,), 0, acc)
    for start in range(0, ROWS, BLOCK):
        row = start + r
        global_row = START + row
        b, t = global_row // T, global_row % T
        p, valid = _pad_index(t * H + k - PAD, L, MODE)
        xr, xi = _load_pair(X, b * XS0 + p * XS1, (row < ROWS) & valid, XC, XCONJ, XNEG)
        xr, xi = xr.to(acc), xi.to(acc)
        gr = tl.load(G + 2 * (row * N + k), row < ROWS, other=0).to(acc)
        gi = tl.load(G + 2 * (row * N + k) + 1, row < ROWS, other=0).to(acc)
        re += gr * xr + gi * xi
        im += gi * xr - gr * xi
    sr, si = tl.sum(re, 0) * SCALE, tl.sum(im, 0) * SCALE
    if WC:
        if not FIRST:
            sr += tl.load(DW + 2 * w)
            si += tl.load(DW + 2 * w + 1)
        tl.store(DW + 2 * w, sr)
        tl.store(DW + 2 * w + 1, si)
    else:
        if not FIRST:
            sr += tl.load(DW + w)
        tl.store(DW + w, sr)


def _storage(x):
    if x is None:
        return None, False
    conj = x.is_conj()
    if conj:
        x = x.conj()
    return (torch.view_as_real(x) if x.is_complex() else x), conj


@dataclass(frozen=True)
class _Plan:
    n: int
    hop: int
    win: int
    pad: int
    mode: str
    frames: int
    freq: int
    dtype: torch.dtype
    scale: float


def _frame_args(x, w, plan):
    # Bound scalar (real/imag) offsets, including padded masked addresses.
    # Keep the 64-bit path for large signals, strides, or frame matrices.
    max_index = max(
        2 * (x.shape[0] * x.stride(0) + (x.shape[1] + 2 * plan.pad) * x.stride(1)),
        2 * x.shape[0] * plan.frames * max(plan.n, plan.freq),
        0 if w is None else 2 * plan.n * w.stride(0),
    )
    return dict(
        T=plan.frames,
        N=plan.n,
        L=x.shape[-1],
        H=plan.hop,
        WIN=plan.win,
        PAD=plan.pad,
        XS0=x.stride(0),
        XS1=x.stride(1),
        WS=w.stride(0) if w is not None else 0,
        XC=x.is_complex(),
        WC=w.is_complex() if w is not None else False,
        XCONJ=x.is_conj(),
        WCONJ=w.is_conj() if w is not None else False,
        XNEG=x.is_neg(),
        WNEG=w.is_neg() if w is not None else False,
        HAS_W=w is not None,
        MODE=plan.mode,
        FP64=plan.dtype in (torch.float64, torch.complex128),
        INDEX64=max_index >= (1 << 31) - 256,
        SCALE=plan.scale,
    )


class _Stft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, plan):
        if ctx is not None:
            ctx.plan = plan
            ctx.save_for_backward(x, w)
        if device.vendor_name == "ascend":
            from flag_gems.runtime.backend._ascend.stft_fft import try_fused_stft

            result = try_fused_stft(x, w, plan)
            if result is not None:
                return result
        rows = x.shape[0] * plan.frames
        complex_dtype = (
            _COMPLEX_DTYPES[_REAL_DTYPES.index(plan.dtype)]
            if plan.dtype in _REAL_DTYPES
            else plan.dtype
        )
        if (
            device.vendor_name in ("nvidia", "hygon", "mthreads")
            and 64 <= plan.n <= 2048
            and plan.n & (plan.n - 1) == 0
        ):
            out = torch.empty(
                (x.shape[0], plan.frames, plan.freq),
                device=x.device,
                dtype=complex_dtype,
            )
            if rows:
                xp, _ = _storage(x)
                wp, _ = _storage(w)
                op, _ = _storage(out)
                with torch_device_fn.device(x.device):
                    nvidia = device.vendor_name == "nvidia"
                    twiddle = (
                        _nv_twiddle(
                            plan.n,
                            plan.dtype in (torch.float64, torch.complex128),
                            x.device,
                        )
                        if nvidia
                        else None
                    )
                    _fused_fft_kernel[(rows,)](
                        xp if x.shape[1] else None,
                        wp,
                        op,
                        **_frame_args(x, w, plan),
                        F=plan.freq,
                        LOG_N=plan.n.bit_length() - 1,
                        NV=nvidia,
                        TW=twiddle,
                        num_warps=4,
                    )
            return out
        # cuFFT's half butterflies accumulate rounding error at every stage.
        # Promote the internal frame/FFT, then round only the final spectrum.
        frame_dtype = {
            torch.float16: torch.float32,
            torch.complex32: torch.complex64,
        }.get(plan.dtype, plan.dtype)
        # A full real spectrum can use C2C with zero imaginary input. This
        # avoids a separate Hermitian completion launch for small STFT frames.
        if frame_dtype in _REAL_DTYPES and plan.freq > plan.n // 2 + 1:
            frame_dtype = _COMPLEX_DTYPES[_REAL_DTYPES.index(frame_dtype)]
        itemsize = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.complex32: 4,
            torch.complex64: 8,
            torch.complex128: 16,
        }[frame_dtype]
        step = max(1, _FRAME_BYTES // (plan.n * itemsize))
        fft_freq = plan.n if frame_dtype in _COMPLEX_DTYPES else plan.n // 2 + 1
        direct_output = (
            0 < rows <= step
            and plan.freq == fft_freq
            and complex_dtype not in (torch.complex32,)
        )
        out = (
            None
            if direct_output
            else torch.empty(
                (x.shape[0], plan.frames, plan.freq),
                device=x.device,
                dtype=complex_dtype,
            )
        )
        xp, _ = _storage(x)
        wp, _ = _storage(w)
        op, _ = _storage(out)
        args = _frame_args(x, w, plan)
        frame_kernel = (
            _frame_kernel.fn if device.vendor_name == "nvidia" else _frame_kernel
        )
        spectrum_kernel = (
            _spectrum_kernel.fn if device.vendor_name == "nvidia" else _spectrum_kernel
        )
        with torch_device_fn.device(x.device):
            for start in range(0, rows, step):
                count = min(step, rows - start)
                frames = torch.empty(
                    (count, plan.n), device=x.device, dtype=frame_dtype
                )
                fp, _ = _storage(frames)
                lanes = plan.n * (2 if frames.is_complex() else 1)
                frame_tiles = triton.cdiv(lanes, 256)
                flat_frame = frame_tiles > 65535
                frame_grid = (
                    (count * frame_tiles,) if flat_frame else (count, frame_tiles)
                )
                frame_kernel[frame_grid](
                    xp if x.shape[1] else None,
                    wp,
                    fp,
                    start,
                    count,
                    YC=frames.is_complex(),
                    BLOCK=256,
                    FLAT_GRID=flat_frame,
                    **args,
                )
                spectrum = fft_frames(frames)
                if direct_output:
                    return spectrum.view(x.shape[0], plan.frames, plan.freq)
                sp, _ = _storage(spectrum)
                spectrum_tiles = triton.cdiv(2 * plan.freq, 256)
                flat_spectrum = spectrum_tiles > 65535
                spectrum_grid = (
                    (count * spectrum_tiles,)
                    if flat_spectrum
                    else (count, spectrum_tiles)
                )
                spectrum_kernel[spectrum_grid](
                    sp,
                    op,
                    start,
                    count,
                    plan.n,
                    plan.freq,
                    spectrum.shape[-1],
                    1.0,
                    args["INDEX64"],
                    256,
                    FLAT_GRID=flat_spectrum,
                )
        return out

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        p = ctx.plan
        rows = x.shape[0] * p.frames
        fp64 = p.dtype in (torch.float64, torch.complex128)
        # Accumulate low precision gradients in float32 until the final cast.
        real_dtype = torch.float64 if fp64 else torch.float32
        complex_dtype = torch.complex128 if fp64 else torch.complex64
        dx = (
            torch.empty(x.shape, device=x.device, dtype=x.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = (
            torch.empty(
                w.shape,
                device=w.device,
                dtype=complex_dtype if w.is_complex() else real_dtype,
            )
            if w is not None and ctx.needs_input_grad[1]
            else None
        )
        if x.numel() == 0:
            # Empty signals/batches have no input contribution to the window
            # gradient. Avoid both NULL tensor arguments and zero-size launches.
            if dw is not None:
                dw = torch.empty(w.shape, device=w.device, dtype=w.dtype)
                dwp, _ = _storage(dw)
                size = dw.numel() * (2 if dw.is_complex() else 1)
                with torch_device_fn.device(x.device):
                    _zero_kernel[(triton.cdiv(size, 256),)](dwp, size, 256)
            return dx, dw, None
        padded = (
            torch.empty(
                (x.shape[0], x.shape[1] + 2 * p.pad),
                device=x.device,
                dtype=complex_dtype if x.is_complex() else real_dtype,
            )
            if dx is not None
            else None
        )
        gp, gc = _storage(grad)
        xp, _ = _storage(x)
        wp, wc = _storage(w)
        pp, _ = _storage(padded)
        dwp, _ = _storage(dw)
        step = max(1, _FRAME_BYTES // (p.n * (16 if fp64 else 8)))
        with torch_device_fn.device(x.device):
            if rows == 0 and dw is not None:
                size = dw.numel() * (2 if dw.is_complex() else 1)
                _zero_kernel[(triton.cdiv(size, 256),)](dwp, size, 256)
            for start in range(0, rows, step):
                count = min(step, rows - start)
                spectrum = torch.empty(
                    (count, p.n), device=x.device, dtype=complex_dtype
                )
                sp, _ = _storage(spectrum)
                _gradient_spectrum_kernel[(triton.cdiv(count * p.n, 256),)](
                    gp,
                    sp,
                    start,
                    count,
                    p.frames,
                    p.n,
                    p.freq,
                    *grad.stride(),
                    gc,
                    grad.is_neg(),
                    256,
                )
                time_grad = fft_frames(spectrum, inverse=True)
                tp, _ = _storage(time_grad)
                if dx is not None:
                    _overlap_kernel[(triton.cdiv(padded.numel(), 256),)](
                        tp,
                        wp,
                        pp,
                        start,
                        count,
                        x.shape[0],
                        p.frames,
                        p.n,
                        p.hop,
                        p.win,
                        padded.shape[1],
                        w.stride(0) if w is not None else 0,
                        w.is_complex() if w is not None else False,
                        wc,
                        w.is_neg() if w is not None else False,
                        w is not None,
                        x.is_complex(),
                        p.scale,
                        start == 0,
                        fp64,
                        256,
                    )
                if dw is not None:
                    _window_gradient_kernel[(p.win,)](
                        xp,
                        tp,
                        dwp,
                        start,
                        count,
                        p.frames,
                        p.n,
                        x.shape[1],
                        p.hop,
                        p.win,
                        p.pad,
                        *x.stride(),
                        x.is_complex(),
                        x.is_conj(),
                        x.is_neg(),
                        w.is_complex(),
                        p.mode,
                        p.scale,
                        start == 0,
                        fp64,
                        256,
                    )
            if dx is not None:
                dp, _ = _storage(dx)
                _unpad_kernel[
                    (triton.cdiv(dx.numel() * (2 if x.is_complex() else 1), 256),)
                ](
                    pp,
                    dp,
                    x.shape[0],
                    x.shape[1],
                    p.pad,
                    x.is_complex(),
                    p.mode,
                    256,
                )
        # Use the same device copy kernel for the final gradient cast.
        if dw is not None and dw.dtype != w.dtype:
            cast = torch.empty(w.shape, dtype=w.dtype, device=w.device)
            cp, _ = _storage(cast)
            with torch_device_fn.device(w.device):
                _unpad_kernel[
                    (triton.cdiv(w.numel() * (2 if w.is_complex() else 1), 256),)
                ](
                    dwp,
                    cp,
                    1,
                    w.numel(),
                    0,
                    w.is_complex(),
                    "constant",
                    256,
                )
            dw = cast
        return dx, dw, None


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


def stft_center(
    self,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    pad_mode="reflect",
    normalized=False,
    onesided=None,
    return_complex=None,
    align_to_window=None,
):
    logger.debug("GEMS STFT.CENTER")
    if window is None:
        _warn_once(
            "window",
            "A window was not provided. A rectangular window will be applied, "
            "which is known to cause spectral leakage.",
        )
    if window is not None and window.device != self.device:
        raise RuntimeError("stft input and window must be on the same device")
    if center and align_to_window is not None:
        raise RuntimeError(
            "stft align_to_window should only be set when center = false"
        )
    complex_input = self.is_complex() or (window is not None and window.is_complex())
    if return_complex is None:
        if not complex_input:
            raise RuntimeError(
                "stft requires the return_complex parameter be given for real inputs"
            )
        return_complex = True
    if not return_complex:
        _warn_once(
            "real",
            "stft with return_complex=False is deprecated. Use torch.view_as_real on the complex output instead.",
        )
    # BF16 may be promoted by a window, but shell dtypes such as FP8/FP4
    # have no STFT input contract and must fail before promote_types.
    if self.dtype not in _REAL_DTYPES + _COMPLEX_DTYPES + (torch.bfloat16,):
        raise NotImplementedError("stft expects floating point or complex input")
    if self.ndim not in (1, 2):
        raise RuntimeError("stft expects a 1D or 2D tensor")
    hop = n_fft // 4 if hop_length is None else hop_length
    win = n_fft if win_length is None else win_length
    length = self.shape[-1]
    pad = n_fft // 2 if center else 0
    if n_fft <= 0 or n_fft > length + 2 * pad:
        raise RuntimeError("stft requires 0 < n_fft <= padded input length")
    if hop <= 0:
        raise RuntimeError("stft requires hop_length > 0")
    if win <= 0 or win > n_fft:
        raise RuntimeError("stft requires 0 < win_length <= n_fft")
    if window is not None and (window.ndim != 1 or window.numel() != win):
        raise RuntimeError("stft expects a 1D window of length win_length")
    frames = 1 + (length + 2 * pad - n_fft) // hop
    if not center and align_to_window:
        pad = (n_fft - win) // 2
        frames = 1 + (length - win) // hop
        if (frames - 1) * hop + n_fft > length + 2 * pad:
            raise RuntimeError("stft aligned frame exceeds padded input storage")
    if center or align_to_window:
        if pad_mode not in ("constant", "reflect", "replicate", "circular"):
            raise NotImplementedError(f"Unrecognised padding mode {pad_mode}")
        if pad_mode == "reflect" and pad >= length:
            raise RuntimeError(
                "Padding size should be less than the corresponding input dimension"
            )
        if pad_mode == "circular" and pad > length:
            raise RuntimeError("Padding value causes wrapping around more than once")
        if pad_mode == "replicate" and length == 0:
            raise RuntimeError(
                "Expected a nonempty input dimension for replicate padding"
            )
    dtype = (
        torch.promote_types(self.dtype, window.dtype)
        if window is not None
        else self.dtype
    )
    if dtype not in _REAL_DTYPES + _COMPLEX_DTYPES:
        raise NotImplementedError(f"stft FFT dtype {dtype} is not supported")
    if device.vendor_name != "nvidia" and dtype not in (torch.float32, torch.complex64):
        raise NotImplementedError(
            f"stft FFT dtype {dtype} is not supported on {device.vendor_name}"
        )
    if dtype in (torch.float16, torch.complex32) and n_fft & (n_fft - 1):
        raise NotImplementedError("stft half precision requires a power-of-two n_fft")
    if onesided is None:
        onesided = not complex_input
    if complex_input and onesided:
        raise RuntimeError("Cannot have onesided output if window or input is complex")
    freq = n_fft // 2 + 1 if onesided else n_fft
    plan = _Plan(
        n_fft,
        hop,
        win,
        pad,
        pad_mode if pad else "constant",
        frames,
        freq,
        dtype,
        1 / math.sqrt(n_fft) if normalized else 1.0,
    )
    inp = self.unsqueeze(0) if self.ndim == 1 else self
    if torch.is_grad_enabled() and (
        self.requires_grad or (window is not None and window.requires_grad)
    ):
        result = _Stft.apply(inp, window, plan)
    else:
        result = _Stft.forward(None, inp, window, plan)
    result = result.transpose(1, 2)
    if self.ndim == 1:
        result = result.squeeze(0)
    return result if return_complex else torch.view_as_real(result)


def stft(
    self,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    normalized=False,
    onesided=None,
    return_complex=None,
    align_to_window=None,
):
    logger.debug("GEMS STFT")
    return stft_center(
        self,
        n_fft,
        hop_length,
        win_length,
        window,
        False,
        "constant",
        normalized,
        onesided,
        return_complex,
        align_to_window,
    )
