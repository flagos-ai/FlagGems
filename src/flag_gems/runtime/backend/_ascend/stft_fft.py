# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Direct AsdSip FFT bridge; both directions are unnormalized.

ABI: CANN 8.5/9.0 fft_api.h, utils/aspb_status.h and aclnn/acl_meta.h.
The SDK guarantees accuracy for finite float32/complex64 input; nonfinite
propagation is validated separately. No ATen FFT or CPU computation is used.
"""

import ctypes
import os
import threading
from collections import OrderedDict
from functools import lru_cache

import torch
import torch_npu
import triton
import triton.language as tl

from flag_gems.ops.stft import _frame_args, _load_pair, _pad_index
from flag_gems.runtime import torch_device_fn

_LOCK = threading.RLock()
_PLANS = OrderedDict()
_CHIRPS = OrderedDict()
_LIBRARY = None
_PID = os.getpid()
_MAX_PLANS = 32


class _Library:
    def __init__(self):
        root = os.environ.get(
            "ASDSIP_HOME_PATH", "/usr/local/Ascend/nnal/asdsip/latest"
        )
        libdir = os.path.join(root, "lib")
        self.dependencies = [
            ctypes.CDLL(os.path.join(libdir, name), mode=ctypes.RTLD_GLOBAL)
            for name in ("libasdsip_host.so", "libasdsip_core.so")
        ]
        self.lib = ctypes.CDLL(os.path.join(libdir, "libasdsip.so"))
        self.meta = ctypes.CDLL("libnnopbase.so")
        p, i, q = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
        self.create = self.bind(self.lib, "asdFftCreate", [ctypes.POINTER(p)])
        self.destroy = self.bind(self.lib, "asdFftDestroy", [p])
        self.plan = self.bind(self.lib, "asdFftMakePlan1D", [p, q, i, i, q, i])
        self.size = self.bind(
            self.lib, "asdFftGetWorkspaceSize", [p, ctypes.POINTER(ctypes.c_size_t)]
        )
        self.workspace = self.bind(self.lib, "asdFftSetWorkspace", [p, p])
        self.stream = self.bind(self.lib, "asdFftSetStream", [p, p])
        self.r2c = self.bind(self.lib, "asdFftExecR2C", [p, p, p])
        self.c2c = self.bind(self.lib, "asdFftExecC2C", [p, p, p])
        qp = ctypes.POINTER(q)
        self.tensor = self.bind(
            self.meta,
            "aclCreateTensor",
            [qp, ctypes.c_uint64, i, qp, q, i, qp, ctypes.c_uint64, p],
            p,
        )
        self.free_tensor = self.bind(self.meta, "aclDestroyTensor", [p])

    @staticmethod
    def bind(lib, name, args, result=ctypes.c_int32):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
        return fn

    @staticmethod
    def check(fn, *args):
        status = fn(*args)
        if status:
            raise RuntimeError(f"STFT {fn.__name__} failed with status {status}")

    def descriptor(self, value):
        shape = (ctypes.c_int64 * 2)(*value.shape)
        strides = (ctypes.c_int64 * 2)(*value.stride())
        desc = self.tensor(
            shape,
            2,
            16 if value.is_complex() else 0,
            strides,
            0,
            2,
            shape,
            2,
            value.data_ptr(),
        )
        if not desc:
            raise RuntimeError("STFT aclCreateTensor returned null")
        return desc


class _Plan:
    def __init__(self, lib, frames, stream, raw, inverse):
        self.lib, self.stream, self.device = lib, stream, frames.device
        self.handle = ctypes.c_void_p()
        self.workspace = None
        lib.check(lib.create, ctypes.byref(self.handle))
        try:
            lib.check(lib.stream, self.handle, raw)
            lib.check(
                lib.plan,
                self.handle,
                frames.shape[1],
                0x10 if frames.is_complex() else 0x12,
                0x11 if inverse else 0x10,
                frames.shape[0],
                0x10,
            )
            size = ctypes.c_size_t()
            lib.check(lib.size, self.handle, ctypes.byref(size))
            self.workspace = torch.empty(
                size.value, dtype=torch.uint8, device=frames.device
            )
            lib.check(lib.workspace, self.handle, self.workspace.data_ptr())
        except BaseException:
            self.close()
            raise

    def close(self):
        with torch.npu.device(self.device.index):
            self.stream.synchronize()
            self.lib.check(self.lib.destroy, self.handle)
        self.workspace = None

    def run(self, frames, output):
        a = self.lib.descriptor(frames)
        b = None
        try:
            b = self.lib.descriptor(output)
            frames.record_stream(self.stream)
            output.record_stream(self.stream)
            self.lib.check(
                self.lib.c2c if frames.is_complex() else self.lib.r2c, self.handle, a, b
            )
        finally:
            self.lib.check(self.lib.free_tensor, a)
            if b:
                self.lib.check(self.lib.free_tensor, b)


@triton.jit
def _chirp_kernel(
    C, N: tl.constexpr, M: tl.constexpr, SIGN: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = lane // 2
    k = tl.minimum(j, M - j).to(tl.int64)
    phase = ((k * k) % (2 * N)).to(tl.float32) * (3.141592653589793 / N)
    real = tl.where(k < N, tl.cos(phase), 0.0)
    imag = tl.where(k < N, -SIGN * tl.sin(phase), 0.0)
    value = tl.where(lane % 2 == 0, real, imag)
    tl.store(C + lane, value, lane < 2 * M)


@triton.jit
def _prepare_kernel(
    X,
    A,
    N: tl.constexpr,
    M: tl.constexpr,
    BATCH: tl.constexpr,
    COMPLEX: tl.constexpr,
    SIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = lane // 2
    row, j = idx // M, idx % M
    k = j.to(tl.int64)
    phase = ((k * k) % (2 * N)).to(tl.float32) * (3.141592653589793 / N)
    c, s = tl.cos(phase), SIGN * tl.sin(phase)
    valid = (row < BATCH) & (j < N)
    if COMPLEX:
        xr = tl.load(X + 2 * (row * N + j), valid, other=0.0)
        xi = tl.load(X + 2 * (row * N + j) + 1, valid, other=0.0)
    else:
        xr = tl.load(X + row * N + j, valid, other=0.0)
        xi = 0.0
    value = tl.where(lane % 2 == 0, xr * c - xi * s, xr * s + xi * c)
    tl.store(A + lane, value, lane < 2 * BATCH * M)


@triton.jit
def _multiply_kernel(
    A, B, C, M: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = lane // 2
    ar = tl.load(A + 2 * j, j < TOTAL, other=0.0)
    ai = tl.load(A + 2 * j + 1, j < TOTAL, other=0.0)
    br = tl.load(B + 2 * (j % M))
    bi = tl.load(B + 2 * (j % M) + 1)
    value = tl.where(lane % 2 == 0, ar * br - ai * bi, ar * bi + ai * br)
    tl.store(C + lane, value, lane < 2 * TOTAL)


@triton.jit
def _finish_kernel(
    X,
    Y,
    N: tl.constexpr,
    M: tl.constexpr,
    OUT: tl.constexpr,
    TOTAL: tl.constexpr,
    SIGN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = lane // 2
    row, j = idx // OUT, idx % OUT
    k = j.to(tl.int64)
    phase = ((k * k) % (2 * N)).to(tl.float32) * (3.141592653589793 / N)
    c, s = tl.cos(phase), SIGN * tl.sin(phase)
    xr = tl.load(X + 2 * (row * M + j), idx < TOTAL, other=0.0)
    xi = tl.load(X + 2 * (row * M + j) + 1, idx < TOTAL, other=0.0)
    value = tl.where(lane % 2 == 0, xr * c - xi * s, xr * s + xi * c)
    tl.store(Y + lane, value * SCALE, lane < 2 * TOTAL)


def _bluestein(frames, inverse):
    batch, n = frames.shape
    m = triton.next_power_of_2(2 * n - 1)
    if m > (1 << 27) or batch * m > (1 << 30):
        raise NotImplementedError(
            "Ascend STFT Bluestein workspace exceeds indexing limits"
        )
    sign = 1 if inverse else -1
    a = torch.empty((batch, m), dtype=torch.complex64, device=frames.device)
    raw = torch_npu._C._npu_getCurrentRawStream(frames.device.index)
    key = (frames.device.index, n, inverse, raw, threading.get_ident())
    with _LOCK:
        entry = _CHIRPS.get(key)
        if entry is None:
            stream = torch.npu.current_stream(frames.device.index)
            chirp = torch.empty((1, m), dtype=torch.complex64, device=frames.device)
            _chirp_kernel[(triton.cdiv(2 * m, 512),)](
                chirp.view(torch.float32), n, m, sign, 512
            )
            bf = fft_frames(chirp)
            bf.record_stream(stream)
            if len(_CHIRPS) >= _MAX_PLANS:
                _CHIRPS.popitem(last=False)
            _CHIRPS[key] = (bf, stream)
        else:
            bf, stream = entry
        _CHIRPS.move_to_end(key)
    frames.record_stream(stream)
    _prepare_kernel[(triton.cdiv(2 * batch * m, 512),)](
        frames.view(torch.float32),
        a.view(torch.float32),
        n,
        m,
        batch,
        frames.is_complex(),
        sign,
        512,
    )
    af = fft_frames(a)
    product = torch.empty_like(af)
    _multiply_kernel[(triton.cdiv(2 * batch * m, 512),)](
        af.view(torch.float32),
        bf.view(torch.float32),
        product.view(torch.float32),
        m,
        batch * m,
        512,
    )
    conv = fft_frames(product, inverse=True)
    out_n = n if frames.is_complex() else n // 2 + 1
    output = torch.empty((batch, out_n), dtype=torch.complex64, device=frames.device)
    _finish_kernel[(triton.cdiv(2 * batch * out_n, 512),)](
        conv.view(torch.float32),
        output.view(torch.float32),
        n,
        m,
        out_n,
        batch * out_n,
        sign,
        1.0 / m,
        512,
    )
    return output


@lru_cache(maxsize=128)
def _needs_bluestein(n):
    for factor in range(2, 200):
        while n % factor == 0:
            n //= factor
    return n != 1


def fft_frames(frames, *, inverse=False):
    """Transform contiguous [batch, N] f32/c64 frames on Ascend."""
    global _LIBRARY
    if frames.device.type != "npu" or frames.ndim != 2 or not frames.is_contiguous():
        raise ValueError("STFT FFT requires contiguous [batch, N] NPU frames")
    if frames.dtype not in (torch.float32, torch.complex64):
        raise NotImplementedError("Ascend STFT FFT requires float32 or complex64")
    if frames.is_conj() or frames.is_neg():
        raise ValueError("STFT FFT requires resolved conjugate and negative bits")
    if inverse and not frames.is_complex():
        raise ValueError("STFT inverse FFT requires complex input")
    batch, n = frames.shape
    if batch < 1 or n < 1:
        raise ValueError("STFT FFT requires nonempty frames")
    if os.getpid() != _PID:
        raise RuntimeError("STFT FFT cannot be used after fork; use spawn")
    if n > (1 << 27) or batch * n > (1 << 30):
        raise NotImplementedError("Ascend STFT FFT exceeds supported indexing limits")
    if _needs_bluestein(n):
        with torch.npu.device(frames.device.index):
            capturing = getattr(torch.npu, "is_current_stream_capturing", None)
            if capturing is not None and capturing():
                raise NotImplementedError(
                    "Ascend STFT FFT does not support graph capture"
                )
            return _bluestein(frames, inverse)
    output = torch.empty(
        (batch, n if frames.is_complex() else n // 2 + 1),
        dtype=torch.complex64,
        device=frames.device,
    )
    with torch.npu.device(frames.device.index), _LOCK:
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise NotImplementedError("Ascend STFT FFT does not support graph capture")
        raw = torch_npu._C._npu_getCurrentRawStream(frames.device.index)
        key = (
            frames.device.index,
            frames.dtype,
            batch,
            n,
            inverse,
            raw,
            threading.get_ident(),
        )
        plan = _PLANS.get(key)
        if plan is None:
            if _LIBRARY is None:
                _LIBRARY = _Library()
            if len(_PLANS) >= _MAX_PLANS:
                old_key, old = next(iter(_PLANS.items()))
                old.close()
                del _PLANS[old_key]
            # A cached plan owns its stream; avoid rebuilding Python Stream
            # wrappers and querying every device on the warm execution path.
            stream = torch.npu.current_stream(frames.device.index)
            plan = _Plan(_LIBRARY, frames, stream, raw, inverse)
            _PLANS[key] = plan
        _PLANS.move_to_end(key)
        plan.run(frames, output)
    return output


@lru_cache(maxsize=1)
def _fused_stft_supported():
    """Conservatively enable the compiler/runtime combination tested on 910B."""
    try:
        return triton.__version__ == "3.5.1" and all(
            torch_npu.utils.get_cann_version(component) == "9.0.0"
            for component in ("CANN", "RUNTIME", "COMPILER", "TOOLKIT")
        )
    except Exception:
        # Missing metadata must retain the independently validated SiP path.
        return False


# Keep the two real-spectrum layouts separate: full spectra store N lanes,
# while half spectra reconstruct M=N/2 lanes and write Nyquist separately.
# This avoids the descending full-spectrum stores that regress on 910B.
@triton.jit
def _fused_stft(X, W, Y, META: tl.constexpr):
    T: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    L: tl.constexpr = META[2]
    H: tl.constexpr = META[3]
    WIN: tl.constexpr = META[4]
    PAD: tl.constexpr = META[5]
    XS0: tl.constexpr = META[6]
    XS1: tl.constexpr = META[7]
    WS: tl.constexpr = META[8]
    XC: tl.constexpr = META[9]
    WC: tl.constexpr = META[10]
    XCONJ: tl.constexpr = META[11]
    WCONJ: tl.constexpr = META[12]
    XNEG: tl.constexpr = META[13]
    WNEG: tl.constexpr = META[14]
    HAS_W: tl.constexpr = META[15]
    MODE: tl.constexpr = META[16]
    INDEX64: tl.constexpr = META[17]
    SCALE: tl.constexpr = META[18]
    F: tl.constexpr = META[19]
    LOG_N: tl.constexpr = META[20]
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
        xr, xi = tl.full((N,), 0, tl.float32), tl.full((N,), 0, tl.float32)
    elif XC and XS1 == 1 and PAD == 0 and WIN == N:
        # Read a full valid frame contiguously, then permute only in UB.
        scalar_lane = tl.arange(0, 2 * N)
        frame_base = row // T * XS0 + row % T * H
        raw = tl.load(X + 2 * frame_base + scalar_lane)
        xr = tl.gather(raw, (2 * k).to(tl.int32), 0)
        xi = tl.gather(raw, (2 * k + 1).to(tl.int32), 0)
        if XCONJ:
            xi = -xi
        if XNEG:
            xr, xi = -xr, -xi
    else:
        xr, xi = _load_pair(X, row // T * XS0 + p * XS1, valid, XC, XCONJ, XNEG)
        xr, xi = xr.to(tl.float32), xi.to(tl.float32)
    w = k - (N - WIN) // 2
    wm = (w >= 0) & (w < WIN)
    if HAS_W:
        wr, wi = _load_pair(W, w * WS, wm, WC, WCONJ, WNEG)
        wr, wi = wr.to(tl.float32), wi.to(tl.float32)
    else:
        wr, wi = wm.to(tl.float32), tl.full((N,), 0, tl.float32)
    if not HAS_W and WIN == N:
        r, i = xr, xi
    elif not XC and not WC:
        r, i = xr * wr, xi
    else:
        r, i = xr * wr - xi * wi, xr * wi + xi * wr
    for stage in tl.static_range(1, LOG_N + 1):
        half = 1 << (stage - 1)
        j = lane & (half - 1)
        even = lane & ~half
        odd = even | half
        ur, ui = tl.gather(r, even, 0), tl.gather(i, even, 0)
        vr, vi = tl.gather(r, odd, 0), tl.gather(i, odd, 0)
        angle = j.to(tl.float32) * (-3.141592653589793 / half)
        c, s = tl.cos(angle), tl.sin(angle)
        tr, ti = vr * c - vi * s, vr * s + vi * c
        lower = (lane & half) == 0
        r, i = tl.where(lower, ur + tr, ur - tr), tl.where(lower, ui + ti, ui - ti)
    tl.store(Y + 2 * (row * F + lane), r * SCALE, lane < F)
    tl.store(Y + 2 * (row * F + lane) + 1, i * SCALE, lane < F)


@triton.jit
def _fused_real_full_stft(X, W, Y, META: tl.constexpr):
    T: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    L: tl.constexpr = META[2]
    H: tl.constexpr = META[3]
    WIN: tl.constexpr = META[4]
    PAD: tl.constexpr = META[5]
    XS0: tl.constexpr = META[6]
    XS1: tl.constexpr = META[7]
    WS: tl.constexpr = META[8]
    XC: tl.constexpr = META[9]  # noqa: F841 - shared metadata layout
    WC: tl.constexpr = META[10]  # noqa: F841 - shared metadata layout
    XCONJ: tl.constexpr = META[11]
    WCONJ: tl.constexpr = META[12]
    XNEG: tl.constexpr = META[13]
    WNEG: tl.constexpr = META[14]
    HAS_W: tl.constexpr = META[15]
    MODE: tl.constexpr = META[16]
    INDEX64: tl.constexpr = META[17]
    SCALE: tl.constexpr = META[18]
    F: tl.constexpr = META[19]
    LOG_N: tl.constexpr = META[20]
    row = tl.program_id(0)
    if INDEX64:
        row = row.to(tl.int64)
    lane = tl.arange(0, N // 2)
    k = tl.full((N // 2,), 0, tl.int32)
    for bit in tl.static_range(LOG_N - 1):
        k = k | (((lane >> bit) & 1) << (LOG_N - 2 - bit))
    if INDEX64:
        k = k.to(tl.int64)
    p0, valid0 = _pad_index(row % T * H + 2 * k - PAD, L, MODE)
    p1, valid1 = _pad_index(row % T * H + 2 * k + 1 - PAD, L, MODE)
    if L == 0:
        r, i = tl.full((N // 2,), 0, tl.float32), tl.full((N // 2,), 0, tl.float32)
    elif XS1 == 1 and MODE != "circular":
        # The clamped N-sample interval covers every valid padded index.
        # PAD<=N/2; reflect additionally requires PAD<L in the host plan.
        frame_start = row % T * H - PAD
        anchor = tl.minimum(tl.maximum(frame_start, 0), tl.maximum(L - N, 0))
        scalar_lane = tl.arange(0, N)
        if L >= N:
            raw = tl.load(X + row // T * XS0 + anchor + scalar_lane)
        else:
            raw = tl.load(X + row // T * XS0 + scalar_lane, scalar_lane < L, 0.0)
        index0 = tl.where(valid0, p0 - anchor, 0).to(tl.int32)
        index1 = tl.where(valid1, p1 - anchor, 0).to(tl.int32)
        r = tl.gather(raw, index0, 0).to(tl.float32)
        i = tl.gather(raw, index1, 0).to(tl.float32)
        r, i = tl.where(valid0, r, 0.0), tl.where(valid1, i, 0.0)
        if XNEG:
            r, i = -r, -i
    else:
        r, _ = _load_pair(X, row // T * XS0 + p0 * XS1, valid0, False, XCONJ, XNEG)
        i, _ = _load_pair(X, row // T * XS0 + p1 * XS1, valid1, False, XCONJ, XNEG)
        r, i = r.to(tl.float32), i.to(tl.float32)
    w0, w1 = 2 * k - (N - WIN) // 2, 2 * k + 1 - (N - WIN) // 2
    wm0, wm1 = (w0 >= 0) & (w0 < WIN), (w1 >= 0) & (w1 < WIN)
    if HAS_W:
        if WS == 1:
            # Load the physical window contiguously, then permute inside UB.
            window_lane = tl.arange(0, N)
            window_raw = tl.load(W + window_lane, window_lane < WIN, 0.0)
            wi0 = tl.where(wm0, w0, 0).to(tl.int32)
            wi1 = tl.where(wm1, w1, 0).to(tl.int32)
            wr0 = tl.where(wm0, tl.gather(window_raw, wi0, 0), 0.0)
            wr1 = tl.where(wm1, tl.gather(window_raw, wi1, 0), 0.0)
            if WNEG:
                wr0, wr1 = -wr0, -wr1
        else:
            wr0, _ = _load_pair(W, w0 * WS, wm0, False, WCONJ, WNEG)
            wr1, _ = _load_pair(W, w1 * WS, wm1, False, WCONJ, WNEG)
        r, i = r * wr0.to(tl.float32), i * wr1.to(tl.float32)
    elif WIN != N:
        r, i = r * wm0.to(tl.float32), i * wm1.to(tl.float32)
    for stage in tl.static_range(1, LOG_N):
        half = 1 << (stage - 1)
        j = lane & (half - 1)
        even = lane & ~half
        odd = even | half
        ur, ui = tl.gather(r, even, 0), tl.gather(i, even, 0)
        vr, vi = tl.gather(r, odd, 0), tl.gather(i, odd, 0)
        angle = j.to(tl.float32) * (-3.141592653589793 / half)
        c, si = tl.cos(angle), tl.sin(angle)
        tr, ti = vr * c - vi * si, vr * si + vi * c
        lower = (lane & half) == 0
        r, i = tl.where(lower, ur + tr, ur - tr), tl.where(lower, ui + ti, ui - ti)
    freq = tl.arange(0, N)
    direct, mirror = freq % (N // 2), (-freq) & (N // 2 - 1)
    ar, ai = tl.gather(r, direct, 0), tl.gather(i, direct, 0)
    br, bi = tl.gather(r, mirror, 0), tl.gather(i, mirror, 0)
    er, ei = (ar + br) * 0.5, (ai - bi) * 0.5
    ore, oi = (ai + bi) * 0.5, (br - ar) * 0.5
    angle = freq.to(tl.float32) * (-6.283185307179586 / N)
    c, si = tl.cos(angle), tl.sin(angle)
    xr, xi = er + c * ore - si * oi, ei + c * oi + si * ore
    xr = tl.where(freq == 0, ar + ai, tl.where(freq == N // 2, ar - ai, xr))
    xi = tl.where((freq == 0) | (freq == N // 2), 0.0, xi)
    tl.store(Y + 2 * (row * F + freq), xr * SCALE, freq < F)
    tl.store(Y + 2 * (row * F + freq) + 1, xi * SCALE, freq < F)


@triton.jit
def _fused_real_stft(X, W, Y, META: tl.constexpr):
    T: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    L: tl.constexpr = META[2]
    H: tl.constexpr = META[3]
    WIN: tl.constexpr = META[4]
    PAD: tl.constexpr = META[5]
    XS0: tl.constexpr = META[6]
    XS1: tl.constexpr = META[7]
    WS: tl.constexpr = META[8]
    XC: tl.constexpr = META[9]  # noqa: F841 - shared metadata layout
    WC: tl.constexpr = META[10]  # noqa: F841 - shared metadata layout
    XCONJ: tl.constexpr = META[11]
    WCONJ: tl.constexpr = META[12]
    XNEG: tl.constexpr = META[13]
    WNEG: tl.constexpr = META[14]
    HAS_W: tl.constexpr = META[15]
    MODE: tl.constexpr = META[16]
    INDEX64: tl.constexpr = META[17]
    SCALE: tl.constexpr = META[18]
    F: tl.constexpr = META[19]
    LOG_N: tl.constexpr = META[20]
    row = tl.program_id(0)
    if INDEX64:
        row = row.to(tl.int64)
    lane = tl.arange(0, N // 2)
    k = tl.full((N // 2,), 0, tl.int32)
    for bit in tl.static_range(LOG_N - 1):
        k = k | (((lane >> bit) & 1) << (LOG_N - 2 - bit))
    if INDEX64:
        k = k.to(tl.int64)
    p0, valid0 = _pad_index(row % T * H + 2 * k - PAD, L, MODE)
    p1, valid1 = _pad_index(row % T * H + 2 * k + 1 - PAD, L, MODE)
    if L == 0:
        r, i = tl.full((N // 2,), 0, tl.float32), tl.full((N // 2,), 0, tl.float32)
    elif XS1 == 1 and MODE != "circular":
        # The clamped N-sample interval covers every valid padded index.
        # PAD<=N/2; reflect additionally requires PAD<L in the host plan.
        frame_start = row % T * H - PAD
        anchor = tl.minimum(tl.maximum(frame_start, 0), tl.maximum(L - N, 0))
        scalar_lane = tl.arange(0, N)
        if L >= N:
            raw = tl.load(X + row // T * XS0 + anchor + scalar_lane)
        else:
            raw = tl.load(X + row // T * XS0 + scalar_lane, scalar_lane < L, 0.0)
        index0 = tl.where(valid0, p0 - anchor, 0).to(tl.int32)
        index1 = tl.where(valid1, p1 - anchor, 0).to(tl.int32)
        r = tl.gather(raw, index0, 0).to(tl.float32)
        i = tl.gather(raw, index1, 0).to(tl.float32)
        r, i = tl.where(valid0, r, 0.0), tl.where(valid1, i, 0.0)
        if XNEG:
            r, i = -r, -i
    else:
        r, _ = _load_pair(X, row // T * XS0 + p0 * XS1, valid0, False, XCONJ, XNEG)
        i, _ = _load_pair(X, row // T * XS0 + p1 * XS1, valid1, False, XCONJ, XNEG)
        r, i = r.to(tl.float32), i.to(tl.float32)
    w0, w1 = 2 * k - (N - WIN) // 2, 2 * k + 1 - (N - WIN) // 2
    wm0, wm1 = (w0 >= 0) & (w0 < WIN), (w1 >= 0) & (w1 < WIN)
    if HAS_W:
        if WS == 1:
            # Load the physical window contiguously, then permute inside UB.
            window_lane = tl.arange(0, N)
            window_raw = tl.load(W + window_lane, window_lane < WIN, 0.0)
            wi0 = tl.where(wm0, w0, 0).to(tl.int32)
            wi1 = tl.where(wm1, w1, 0).to(tl.int32)
            wr0 = tl.where(wm0, tl.gather(window_raw, wi0, 0), 0.0)
            wr1 = tl.where(wm1, tl.gather(window_raw, wi1, 0), 0.0)
            if WNEG:
                wr0, wr1 = -wr0, -wr1
        else:
            wr0, _ = _load_pair(W, w0 * WS, wm0, False, WCONJ, WNEG)
            wr1, _ = _load_pair(W, w1 * WS, wm1, False, WCONJ, WNEG)
        r, i = r * wr0.to(tl.float32), i * wr1.to(tl.float32)
    elif WIN != N:
        r, i = r * wm0.to(tl.float32), i * wm1.to(tl.float32)
    for stage in tl.static_range(1, LOG_N):
        half = 1 << (stage - 1)
        j = lane & (half - 1)
        even = lane & ~half
        odd = even | half
        ur, ui = tl.gather(r, even, 0), tl.gather(i, even, 0)
        vr, vi = tl.gather(r, odd, 0), tl.gather(i, odd, 0)
        angle = j.to(tl.float32) * (-3.141592653589793 / half)
        c, si = tl.cos(angle), tl.sin(angle)
        tr, ti = vr * c - vi * si, vr * si + vi * c
        lower = (lane & half) == 0
        r, i = tl.where(lower, ur + tr, ur - tr), tl.where(lower, ui + ti, ui - ti)
    # Reconstruct only M lanes. The upper real spectrum is conjugate symmetric.
    mirror = (-lane) & (N // 2 - 1)
    br, bi = tl.gather(r, mirror, 0), tl.gather(i, mirror, 0)
    er, ei = (r + br) * 0.5, (i - bi) * 0.5
    ore, oi = (i + bi) * 0.5, (br - r) * 0.5
    angle = lane.to(tl.float32) * (-6.283185307179586 / N)
    c, si = tl.cos(angle), tl.sin(angle)
    xr, xi = er + c * ore - si * oi, ei + c * oi + si * ore
    xr, xi = tl.where(lane == 0, r + i, xr), tl.where(lane == 0, 0.0, xi)
    tl.store(Y + 2 * (row * F + lane), xr * SCALE)
    tl.store(Y + 2 * (row * F + lane) + 1, xi * SCALE)
    # lane zero writes Nyquist, others write N-k only for a full spectrum.
    high = tl.where(lane == 0, N // 2, N - lane)
    high_r = tl.where(lane == 0, r - i, xr)
    mask = (lane == 0) | (F == N)
    tl.store(Y + 2 * (row * F + high), high_r * SCALE, mask)
    tl.store(Y + 2 * (row * F + high) + 1, -xi * SCALE, mask)


def try_fused_stft(x, w, plan):
    """Return fused B,T,F output, or None to use the owned SiP implementation."""
    if os.getpid() != _PID:
        raise RuntimeError("STFT FFT cannot be used after fork; use spawn")
    n = plan.n
    if (
        n < 64
        or n > 1024
        or n & (n - 1)
        or plan.dtype not in (torch.float32, torch.complex64)
        or not _fused_stft_supported()
    ):
        return None
    # Promotion can yield complex64 from complex-half physical storage. The
    # fused reinterpret pointers require 32-bit components; the SiP framing
    # path preserves each input's actual component dtype before promotion.
    if x.dtype == torch.complex32 or (w is not None and w.dtype == torch.complex32):
        return None
    with torch_device_fn.device(x.device):
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise NotImplementedError("Ascend STFT FFT does not support graph capture")
        out = torch.empty(
            (x.shape[0], plan.frames, plan.freq),
            device=x.device,
            dtype=torch.complex64,
        )
        rows = x.shape[0] * plan.frames
        if rows:
            # Preserve the original storage address and defer lazy flags to JIT.
            xp = triton.reinterpret(x, tl.float32) if x.is_complex() else x
            wp = (
                triton.reinterpret(w, tl.float32)
                if w is not None and w.is_complex()
                else w
            )
            op = triton.reinterpret(out, tl.float32)
            args = _frame_args(x, w, plan)
            args.pop("FP64")
            if not x.is_complex() and (w is None or not w.is_complex()):
                kernel = _fused_real_full_stft if plan.freq == n else _fused_real_stft
            else:
                kernel = _fused_stft
            kernel[(rows,)](
                xp if x.shape[1] else None,
                wp,
                op,
                # One explicit constexpr tuple reduces host parameter binding.
                # Keep this order aligned with all three kernels' META unpacking.
                (
                    args["T"],
                    args["N"],
                    args["L"],
                    args["H"],
                    args["WIN"],
                    args["PAD"],
                    args["XS0"],
                    args["XS1"],
                    args["WS"],
                    args["XC"],
                    args["WC"],
                    args["XCONJ"],
                    args["WCONJ"],
                    args["XNEG"],
                    args["WNEG"],
                    args["HAS_W"],
                    args["MODE"],
                    args["INDEX64"],
                    args["SCALE"],
                    plan.freq,
                    n.bit_length() - 1,
                ),
                num_warps=4,
            )
    return out
