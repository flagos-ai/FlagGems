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
from triton.language.extra import libdevice as _ld

# The MetaX warp is 64 threads wide, so BLOCK=2048 over 4 warps (256 threads)
# gives 8 elements per thread: one full 16-byte access per thread for 2-byte
# dtypes (two 128-bit accesses for fp32).  Measured effect: fp16/bf16 streaming
# went from 1.35/1.25 TB/s to 1.50 TB/s, matching fp32 and the practical
# streaming ceiling of this device.
_BLOCK = 2048
# Latency-bound small tensors instead want their few kilobytes spread over many
# CTAs, so they get one element per thread.
_BLOCK_SMALL = 256
_SMALL_MAX = 16384
_NUM_WARPS = 4
_MAX_BATCH = 16


@triton.jit
def _rsqrt_flat(
    x_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        # numel is a multiple of BLOCK: no bound check and no masked store.
        x = tl.load(x_ptr + offs)
        if F64:
            y = 1.0 / tl.sqrt(x)
        else:
            y = tl.rsqrt(x.to(tl.float32))
        tl.store(o_ptr + offs, y)
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        if F64:
            y = 1.0 / tl.sqrt(x)
        else:
            y = tl.rsqrt(x.to(tl.float32))
        tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _rsqrt_multi(
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
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    EXACT: tl.constexpr,
):
    """One launch for up to 16 equal-shaped contiguous tensors.

    Axis 1 selects the source tensor, so the per-tensor launch cost is paid once.
    """
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    xp = x0
    if tid == 1:
        xp = x1
    elif tid == 2:
        xp = x2
    elif tid == 3:
        xp = x3
    elif tid == 4:
        xp = x4
    elif tid == 5:
        xp = x5
    elif tid == 6:
        xp = x6
    elif tid == 7:
        xp = x7
    elif tid == 8:
        xp = x8
    elif tid == 9:
        xp = x9
    elif tid == 10:
        xp = x10
    elif tid == 11:
        xp = x11
    elif tid == 12:
        xp = x12
    elif tid == 13:
        xp = x13
    elif tid == 14:
        xp = x14
    elif tid == 15:
        xp = x15
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ob = o_ptr + tid * n
    if EXACT:
        x = tl.load(xp + offs)
        if F64:
            y = 1.0 / tl.sqrt(x)
        else:
            y = tl.rsqrt(x.to(tl.float32))
        tl.store(ob + offs, y)
    else:
        mask = offs < n
        x = tl.load(xp + offs, mask=mask, other=1.0)
        if F64:
            y = 1.0 / tl.sqrt(x)
        else:
            y = tl.rsqrt(x.to(tl.float32))
        tl.store(ob + offs, y, mask=mask)


@triton.jit
def _rsqrt_strided(
    x_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.int32)
    mask = offs < n
    sh = (s0, s1, s2, s3, s4, s5, s6, s7)
    st = (t0, t1, t2, t3, t4, t5, t6, t7)
    rem = offs
    if RANK > 7:
        acc += (rem % sh[7]) * st[7]
        rem = rem // sh[7]
    if RANK > 6:
        acc += (rem % sh[6]) * st[6]
        rem = rem // sh[6]
    if RANK > 5:
        acc += (rem % sh[5]) * st[5]
        rem = rem // sh[5]
    if RANK > 4:
        acc += (rem % sh[4]) * st[4]
        rem = rem // sh[4]
    if RANK > 3:
        acc += (rem % sh[3]) * st[3]
        rem = rem // sh[3]
    if RANK > 2:
        acc += (rem % sh[2]) * st[2]
        rem = rem // sh[2]
    if RANK > 1:
        acc += (rem % sh[1]) * st[1]
        rem = rem // sh[1]
    acc += rem * st[0]
    x = tl.load(x_ptr + acc, mask=mask, other=1.0)
    if F64:
        y = 1.0 / tl.sqrt(x)
    else:
        y = tl.rsqrt(x.to(tl.float32))
    tl.store(o_ptr + offs, y, mask=mask)


@triton.jit
def _rsqrt_complex(
    x_ptr,
    o_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    """rsqrt of complex values stored as interleaved (re, im) reals.

    ATen computes this as rsqrt(|z|) * exp(-i*arg(z)/2); reproducing that form
    (hypot + atan2 half angle) keeps value parity including the specials.
    """
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        acc = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.int32)
    mask = offs < n
    sh = (s0, s1, s2, s3, s4, s5, s6, s7)
    st = (t0, t1, t2, t3, t4, t5, t6, t7)
    rem = offs
    if RANK > 7:
        acc += (rem % sh[7]) * st[7]
        rem = rem // sh[7]
    if RANK > 6:
        acc += (rem % sh[6]) * st[6]
        rem = rem // sh[6]
    if RANK > 5:
        acc += (rem % sh[5]) * st[5]
        rem = rem // sh[5]
    if RANK > 4:
        acc += (rem % sh[4]) * st[4]
        rem = rem // sh[4]
    if RANK > 3:
        acc += (rem % sh[3]) * st[3]
        rem = rem // sh[3]
    if RANK > 2:
        acc += (rem % sh[2]) * st[2]
        rem = rem // sh[2]
    if RANK > 1:
        acc += (rem % sh[1]) * st[1]
        rem = rem // sh[1]
    acc += rem * st[0]
    a = tl.load(x_ptr + acc, mask=mask, other=1.0)
    b = tl.load(x_ptr + acc + 1, mask=mask, other=0.0)
    if not F64:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    r = _ld.hypot(a, b)
    inv = _ld.rsqrt(r)
    th = _ld.atan2(b, a) * 0.5
    tl.store(o_ptr + 2 * offs, inv * _ld.cos(th), mask=mask)
    tl.store(o_ptr + 2 * offs + 1, -(inv * _ld.sin(th)), mask=mask)


def _collapse(shape, stride):
    """Drop size-1 dims and merge adjacent dims that are jointly contiguous."""
    sizes = []
    strides = []
    for sz, sd in zip(shape, stride):
        if sz == 1:
            continue
        if sizes and strides[-1] == sz * sd:
            sizes[-1] *= sz
        else:
            sizes.append(sz)
            strides.append(sd)
    return sizes, strides


def _pad(sizes, strides):
    rank = len(sizes)
    return sizes + [1] * (8 - rank), strides + [0] * (8 - rank), rank


def _block(n):
    """Block size by working-set size, in elements (device independent)."""
    if n <= _SMALL_MAX:
        return _BLOCK_SMALL
    return _BLOCK


def _single(t):
    """rsqrt of one tensor into a freshly allocated contiguous output."""
    dt = t.dtype
    if dt.is_floating_point or dt.is_complex:
        odt = dt
    else:
        odt = torch.get_default_dtype()
    out = torch.empty(t.shape, dtype=odt, device=t.device)
    n = t.numel()
    if n == 0:
        return out
    if dt.is_complex:
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        xv = torch.view_as_real(t)
        ov = torch.view_as_real(out)
        sizes, strides = _collapse(xv.shape[:-1], xv.stride()[:-1])
        pad_s, pad_t, rank = _pad(sizes, strides)
        _rsqrt_complex[grid](
            xv,
            ov,
            n,
            pad_s[0],
            pad_s[1],
            pad_s[2],
            pad_s[3],
            pad_s[4],
            pad_s[5],
            pad_s[6],
            pad_s[7],
            pad_t[0],
            pad_t[1],
            pad_t[2],
            pad_t[3],
            pad_t[4],
            pad_t[5],
            pad_t[6],
            pad_t[7],
            BLOCK=_BLOCK,
            RANK=rank,
            F64=(dt is torch.complex128),
            I64=(2 * n > 2147483647),
            num_warps=_NUM_WARPS,
        )
    elif t.is_contiguous():
        blk = _block(n)
        _rsqrt_flat[((n + blk - 1) // blk,)](
            t,
            out,
            n,
            BLOCK=blk,
            F64=(dt is torch.float64),
            I64=(n > 2147483647),
            EXACT=(n % blk == 0),
            num_warps=_NUM_WARPS,
        )
    else:
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        sizes, strides = _collapse(t.shape, t.stride())
        pad_s, pad_t, rank = _pad(sizes, strides)
        _rsqrt_strided[grid](
            t,
            out,
            n,
            pad_s[0],
            pad_s[1],
            pad_s[2],
            pad_s[3],
            pad_s[4],
            pad_s[5],
            pad_s[6],
            pad_s[7],
            pad_t[0],
            pad_t[1],
            pad_t[2],
            pad_t[3],
            pad_t[4],
            pad_t[5],
            pad_t[6],
            pad_t[7],
            BLOCK=_BLOCK,
            RANK=rank,
            F64=(dt is torch.float64),
            I64=(n > 2147483647),
            num_warps=_NUM_WARPS,
        )
    return out


def foreach_rsqrt(self):
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    nr = len(tensors)
    if nr == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if nr == 1:
        return [_single(tensors[0])]
    outs = [None] * nr
    i = 0
    while i < nr:
        t = tensors[i]
        dt = t.dtype
        n = t.numel()
        if n > 0 and n < 2147483648 and dt.is_floating_point and t.is_contiguous():
            shape = t.shape
            lim = i + _MAX_BATCH
            if lim > nr:
                lim = nr
            j = i + 1
            while j < lim:
                u = tensors[j]
                if u.dtype is not dt or u.shape != shape or not u.is_contiguous():
                    break
                j += 1
            cnt = j - i
            if cnt > 1 and n * cnt < 2147483648:
                buf = torch.empty((cnt,) + shape, dtype=dt, device=t.device)
                xs = list(tensors[i:j])
                if cnt < _MAX_BATCH:
                    xs.extend([tensors[j - 1]] * (_MAX_BATCH - cnt))
                blk = _block(n)
                _rsqrt_multi[((n + blk - 1) // blk, cnt)](
                    *xs,
                    buf,
                    n,
                    BLOCK=blk,
                    F64=(dt is torch.float64),
                    EXACT=(n % blk == 0),
                    num_warps=_NUM_WARPS,
                )
                for k in range(cnt):
                    outs[i + k] = buf[k]
                i = j
                continue
        outs[i] = _single(t)
        i += 1
    return outs
