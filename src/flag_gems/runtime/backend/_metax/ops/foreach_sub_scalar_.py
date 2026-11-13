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

import struct

import torch
import triton
import triton.language as tl

_MAXOUT = 8
_BLOCK_C = 1024
_BLOCK_F = 1024
_NW_F = 4
_KF = 4
_FAST = {}


@triton.jit
def _sub_f4h(x0, x1, x2, x3, s, BLOCK: tl.constexpr):
    j = tl.program_id(1)
    p = x0
    if j == 1:
        p = x1
    if j == 2:
        p = x2
    if j == 3:
        p = x3
    poff = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(p + poff)
    tl.store(p + poff, x - s)


@triton.jit
def _sub_f4(
    x0,
    x1,
    x2,
    x3,
    n,
    s,
    s_ptr,
    SPTR: tl.constexpr,
    SBITS: tl.constexpr,
    SC: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(1)
    p = x0
    if j == 1:
        p = x1
    if j == 2:
        p = x2
    if j == 3:
        p = x3
    poff = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = poff < n
        x = tl.load(p + poff, mask=m)
    else:
        x = tl.load(p + poff)
    if SPTR:
        sv = tl.load(s_ptr)
        if SC == 1:
            sv = sv.to(tl.float32)
        elif SC == 2:
            sv = sv.to(tl.int64)
    else:
        sv = s
        if SBITS:
            sv = sv.to(tl.float64, bitcast=True)
    if MASKED:
        tl.store(p + poff, x - sv, mask=m)
    else:
        tl.store(p + poff, x - sv)


@triton.jit
def _sub_c(
    x_ptr,
    n,
    s,
    s_ptr,
    SPTR: tl.constexpr,
    SBITS: tl.constexpr,
    SC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    poff = pid * BLOCK + tl.arange(0, BLOCK)
    mask = poff < n
    x = tl.load(x_ptr + poff, mask=mask)
    if SPTR:
        sv = tl.load(s_ptr)
        if SC == 1:
            sv = sv.to(tl.float32)
        elif SC == 2:
            sv = sv.to(tl.int64)
    else:
        sv = s
        if SBITS:
            sv = sv.to(tl.float64, bitcast=True)
    tl.store(x_ptr + poff, x - sv, mask=mask)


@triton.jit
def _sub_s(
    x_ptr,
    P,
    st0,
    s,
    s_ptr,
    a1,
    b1,
    a2,
    b2,
    a3,
    b3,
    a4,
    b4,
    a5,
    b5,
    a6,
    b6,
    a7,
    b7,
    a8,
    b8,
    SPTR: tl.constexpr,
    SBITS: tl.constexpr,
    SC: tl.constexpr,
    NOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(1)
    base = idx.to(tl.int64) * 0
    if NOUT > 0:
        q = idx % a1
        idx = idx // a1
        base += q.to(tl.int64) * b1
    if NOUT > 1:
        q = idx % a2
        idx = idx // a2
        base += q.to(tl.int64) * b2
    if NOUT > 2:
        q = idx % a3
        idx = idx // a3
        base += q.to(tl.int64) * b3
    if NOUT > 3:
        q = idx % a4
        idx = idx // a4
        base += q.to(tl.int64) * b4
    if NOUT > 4:
        q = idx % a5
        idx = idx // a5
        base += q.to(tl.int64) * b5
    if NOUT > 5:
        q = idx % a6
        idx = idx // a6
        base += q.to(tl.int64) * b6
    if NOUT > 6:
        q = idx % a7
        idx = idx // a7
        base += q.to(tl.int64) * b7
    if NOUT > 7:
        q = idx % a8
        idx = idx // a8
        base += q.to(tl.int64) * b8

    poff = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = poff < P
    offs = base + poff * st0
    x = tl.load(x_ptr + offs, mask=mask)
    if SPTR:
        sv = tl.load(s_ptr)
        if SC == 1:
            sv = sv.to(tl.float32)
        elif SC == 2:
            sv = sv.to(tl.int64)
    else:
        sv = s
        if SBITS:
            sv = sv.to(tl.float64, bitcast=True)
    tl.store(x_ptr + offs, x - sv, mask=mask)


def _scalar_variants(dtype, scalar, sptr):
    """Return (s_arg, s_ptr_or_None, SPTR, SBITS, SC)."""
    if sptr is not None:
        if dtype is torch.float64:
            return 0, sptr, True, False, 0
        if dtype.is_floating_point:
            return 0, sptr, True, False, 1
        return 0, sptr, True, False, 2
    if dtype is torch.float64:
        bits = struct.unpack("<q", struct.pack("<d", float(scalar)))[0]
        return bits, None, False, True, 0
    if dtype.is_floating_point:
        return float(scalar), None, False, False, 0
    return int(scalar), None, False, False, 0


def _canonicalize(t):
    """Return (P, st0, outer_list) for a non-contiguous strided tensor."""
    merged = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if sz == 0:
            return 0, 1, []
        merged.append((sz, st))
    if not merged:
        return 1, 1, []
    merged.sort(key=lambda p: p[1])
    out = [merged[0]]
    for sz, st in merged[1:]:
        psz, pst = out[-1]
        if st == pst * psz:
            out[-1] = (psz * sz, pst)
        else:
            out.append((sz, st))
    p, st0 = out[0]
    return p, st0, out[1:]


def _fused_entry(t0, t1, t2, t3, n, sarg, sp, key, SPTR, SBITS, SC, MASKED):
    ent = _FAST.get(key)
    if ent is None:
        ck = _sub_f4.warmup(
            t0,
            t1,
            t2,
            t3,
            n,
            sarg,
            sp,
            SPTR=SPTR,
            SBITS=SBITS,
            SC=SC,
            MASKED=MASKED,
            BLOCK=_BLOCK_F,
            num_warps=_NW_F,
            grid=(1, 1, 1),
        )
        ck[(1, 1, 1)]
        ent = (ck.run, ck.function, ck.packed_metadata)
        _FAST[key] = ent
    return ent


def foreach_sub_scalar_(self, scalar):
    k = len(self)
    if k == 0:
        return None

    sptr = None
    if isinstance(scalar, torch.Tensor):
        if scalar.numel() != 1:
            raise RuntimeError(
                "foreach_sub_ expects a scalar, but the provided tensor has "
                "more than one element"
            )
        if scalar.is_cuda:
            sptr = scalar
        else:
            scalar = scalar.item()

    if k == _KF:
        t0 = self[0]
        t1 = self[1]
        t2 = self[2]
        t3 = self[3]
        dt = t0.dtype
        if (
            t1.dtype is dt
            and t2.dtype is dt
            and t3.dtype is dt
            and dt is not torch.bool
            and t0.is_contiguous()
            and t1.is_contiguous()
            and t2.is_contiguous()
            and t3.is_contiguous()
        ):
            n = t0.numel()
            if (
                n > 1
                and (n & 15) == 0
                and n == t1.numel()
                and n == t2.numel()
                and n == t3.numel()
            ):
                p0 = t0.data_ptr()
                p1 = t1.data_ptr()
                p2 = t2.data_ptr()
                p3 = t3.data_ptr()
                if ((p0 | p1 | p2 | p3) & 15) == 0:
                    if (
                        (n & (_BLOCK_F - 1)) == 0
                        and type(scalar) is float
                        and (
                            dt is torch.float32
                            or dt is torch.float16
                            or dt is torch.bfloat16
                        )
                    ):
                        nkey = (0, dt)
                        ent = _FAST.get(nkey)
                        if ent is None:
                            ck = _sub_f4h.warmup(
                                t0,
                                t1,
                                t2,
                                t3,
                                scalar,
                                BLOCK=_BLOCK_F,
                                num_warps=_NW_F,
                                grid=(1, 1, 1),
                            )
                            ck[(1, 1, 1)]
                            ent = (ck.run, ck.function, ck.packed_metadata)
                            _FAST[nkey] = ent
                        stream = torch.cuda.current_stream().cuda_stream
                        run_f, func, meta = ent
                        run_f(
                            n // _BLOCK_F,
                            _KF,
                            1,
                            stream,
                            func,
                            meta,
                            None,
                            None,
                            None,
                            p0,
                            p1,
                            p2,
                            p3,
                            scalar,
                            _BLOCK_F,
                        )
                        return None
                    masked = 1 if (n & (_BLOCK_F - 1)) else 0
                    sarg, sp, SPTR, SBITS, SC = _scalar_variants(dt, scalar, sptr)
                    if not SPTR:
                        sp = p0
                    key = (
                        dt,
                        SPTR,
                        SBITS,
                        SC,
                        masked,
                        n & (_BLOCK_F - 1),
                        1 if sptr is not None else 0,
                    )
                    run_f, func, meta = _fused_entry(
                        t0, t1, t2, t3, n, sarg, sp, key, SPTR, SBITS, SC, masked
                    )
                    stream = torch.cuda.current_stream().cuda_stream
                    run_f(
                        (n + (_BLOCK_F - 1)) // _BLOCK_F,
                        _KF,
                        1,
                        stream,
                        func,
                        meta,
                        None,
                        None,
                        None,
                        p0,
                        p1,
                        p2,
                        p3,
                        n,
                        sarg,
                        sp,
                        SPTR,
                        SBITS,
                        SC,
                        masked,
                        _BLOCK_F,
                    )
                    return None

    for t in self:
        dt = t.dtype
        if dt is torch.bool:
            raise RuntimeError(
                "Subtraction, the `-` operator, with a bool tensor is not "
                "supported. If you are trying to invert a mask, use the `~` "
                "or `logical_not()` operator instead."
            )
        n = t.numel()
        if n == 0:
            continue
        sarg, sp, SPTR, SBITS, SC = _scalar_variants(dt, scalar, sptr)
        if t.is_contiguous():
            grid = (triton.cdiv(n, _BLOCK_C),)
            _sub_c[grid](
                t,
                n,
                sarg,
                sp if SPTR else t,
                SPTR=SPTR,
                SBITS=SBITS,
                SC=SC,
                BLOCK=_BLOCK_C,
            )
        else:
            p, st0, outer = _canonicalize(t)
            if p == 0:
                continue
            if len(outer) > _MAXOUT:
                raise RuntimeError("foreach_sub_: tensor rank too large")
            args = []
            for i in range(_MAXOUT):
                if i < len(outer):
                    args.append(outer[i][0])
                    args.append(outer[i][1])
                else:
                    args.append(1)
                    args.append(0)
            nout = len(outer)
            o = 1
            for sz, _st in outer:
                o *= sz
            block = 1024 if p >= 1024 else max(16, triton.next_power_of_2(p))
            grid = (triton.cdiv(p, block), o)
            _sub_s[grid](
                t,
                p,
                st0,
                sarg,
                sp if SPTR else t,
                *args,
                SPTR=SPTR,
                SBITS=SBITS,
                SC=SC,
                NOUT=nout,
                BLOCK=block,
            )
    return None
