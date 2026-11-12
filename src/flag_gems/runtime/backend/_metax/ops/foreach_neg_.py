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
from triton import knobs as _knobs
from triton.runtime import driver as _driver

_BLOCK = 1024
_WARPS = 4
_UBLOCK = 2048
_SLOTS = 16
_INT32_MAX = 2147483647
_UNIFORM_MAX = 1 << 16

_t_numel = torch.Tensor.numel
_t_ptr = torch.Tensor.data_ptr
_t_contig = torch.Tensor.is_contiguous

_REAL_OF = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


@triton.jit
def _neg_val(x):
    if x.dtype.is_floating() or x.dtype == tl.int64:
        return -x
    else:
        return (-x.to(tl.int32)).to(x.dtype)


@triton.jit
def _neg_flat(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, _neg_val(x), mask=mask)


@triton.jit
def _neg_multi16(
    a0,
    n0,
    a1,
    n1,
    a2,
    n2,
    a3,
    n3,
    a4,
    n4,
    a5,
    n5,
    a6,
    n6,
    a7,
    n7,
    a8,
    n8,
    a9,
    n9,
    a10,
    n10,
    a11,
    n11,
    a12,
    n12,
    a13,
    n13,
    a14,
    n14,
    a15,
    n15,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0) * BLOCK
    offs = base + tl.arange(0, BLOCK)
    if base < n0:
        m0 = offs < n0
        tl.store(a0 + offs, _neg_val(tl.load(a0 + offs, mask=m0)), mask=m0)
    if base < n1:
        m1 = offs < n1
        tl.store(a1 + offs, _neg_val(tl.load(a1 + offs, mask=m1)), mask=m1)
    if base < n2:
        m2 = offs < n2
        tl.store(a2 + offs, _neg_val(tl.load(a2 + offs, mask=m2)), mask=m2)
    if base < n3:
        m3 = offs < n3
        tl.store(a3 + offs, _neg_val(tl.load(a3 + offs, mask=m3)), mask=m3)
    if base < n4:
        m4 = offs < n4
        tl.store(a4 + offs, _neg_val(tl.load(a4 + offs, mask=m4)), mask=m4)
    if base < n5:
        m5 = offs < n5
        tl.store(a5 + offs, _neg_val(tl.load(a5 + offs, mask=m5)), mask=m5)
    if base < n6:
        m6 = offs < n6
        tl.store(a6 + offs, _neg_val(tl.load(a6 + offs, mask=m6)), mask=m6)
    if base < n7:
        m7 = offs < n7
        tl.store(a7 + offs, _neg_val(tl.load(a7 + offs, mask=m7)), mask=m7)
    if base < n8:
        m8 = offs < n8
        tl.store(a8 + offs, _neg_val(tl.load(a8 + offs, mask=m8)), mask=m8)
    if base < n9:
        m9 = offs < n9
        tl.store(a9 + offs, _neg_val(tl.load(a9 + offs, mask=m9)), mask=m9)
    if base < n10:
        m10 = offs < n10
        tl.store(a10 + offs, _neg_val(tl.load(a10 + offs, mask=m10)), mask=m10)
    if base < n11:
        m11 = offs < n11
        tl.store(a11 + offs, _neg_val(tl.load(a11 + offs, mask=m11)), mask=m11)
    if base < n12:
        m12 = offs < n12
        tl.store(a12 + offs, _neg_val(tl.load(a12 + offs, mask=m12)), mask=m12)
    if base < n13:
        m13 = offs < n13
        tl.store(a13 + offs, _neg_val(tl.load(a13 + offs, mask=m13)), mask=m13)
    if base < n14:
        m14 = offs < n14
        tl.store(a14 + offs, _neg_val(tl.load(a14 + offs, mask=m14)), mask=m14)
    if base < n15:
        m15 = offs < n15
        tl.store(a15 + offs, _neg_val(tl.load(a15 + offs, mask=m15)), mask=m15)


@triton.jit
def _neg_multi16u(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    n,
    BLOCK: tl.constexpr,
):
    """Same-numel variant: one tensor per program along axis 1."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tid = tl.program_id(1)
    if tid == 0:
        tl.store(a0 + offs, _neg_val(tl.load(a0 + offs, mask=mask)), mask=mask)
    if tid == 1:
        tl.store(a1 + offs, _neg_val(tl.load(a1 + offs, mask=mask)), mask=mask)
    if tid == 2:
        tl.store(a2 + offs, _neg_val(tl.load(a2 + offs, mask=mask)), mask=mask)
    if tid == 3:
        tl.store(a3 + offs, _neg_val(tl.load(a3 + offs, mask=mask)), mask=mask)
    if tid == 4:
        tl.store(a4 + offs, _neg_val(tl.load(a4 + offs, mask=mask)), mask=mask)
    if tid == 5:
        tl.store(a5 + offs, _neg_val(tl.load(a5 + offs, mask=mask)), mask=mask)
    if tid == 6:
        tl.store(a6 + offs, _neg_val(tl.load(a6 + offs, mask=mask)), mask=mask)
    if tid == 7:
        tl.store(a7 + offs, _neg_val(tl.load(a7 + offs, mask=mask)), mask=mask)
    if tid == 8:
        tl.store(a8 + offs, _neg_val(tl.load(a8 + offs, mask=mask)), mask=mask)
    if tid == 9:
        tl.store(a9 + offs, _neg_val(tl.load(a9 + offs, mask=mask)), mask=mask)
    if tid == 10:
        tl.store(a10 + offs, _neg_val(tl.load(a10 + offs, mask=mask)), mask=mask)
    if tid == 11:
        tl.store(a11 + offs, _neg_val(tl.load(a11 + offs, mask=mask)), mask=mask)
    if tid == 12:
        tl.store(a12 + offs, _neg_val(tl.load(a12 + offs, mask=mask)), mask=mask)
    if tid == 13:
        tl.store(a13 + offs, _neg_val(tl.load(a13 + offs, mask=mask)), mask=mask)
    if tid == 14:
        tl.store(a14 + offs, _neg_val(tl.load(a14 + offs, mask=mask)), mask=mask)
    if tid == 15:
        tl.store(a15 + offs, _neg_val(tl.load(a15 + offs, mask=mask)), mask=mask)


@triton.jit
def _neg_strided(
    x_ptr,
    n_in,
    si,
    n1,
    s1,
    n2,
    s2,
    n3,
    s3,
    n4,
    s4,
    n5,
    s5,
    n6,
    s6,
    n7,
    s7,
    n8,
    s8,
    BLOCK: tl.constexpr,
):
    pi = tl.program_id(0)
    po = tl.program_id(1).to(tl.int64)
    off = (po % n1) * s1
    po = po // n1
    off += (po % n2) * s2
    po = po // n2
    off += (po % n3) * s3
    po = po // n3
    off += (po % n4) * s4
    po = po // n4
    off += (po % n5) * s5
    po = po // n5
    off += (po % n6) * s6
    po = po // n6
    off += (po % n7) * s7
    po = po // n7
    off += (po % n8) * s8

    offs = pi * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_in
    ptr = x_ptr + off + offs * si
    x = tl.load(ptr, mask=mask)
    tl.store(ptr, _neg_val(x), mask=mask)


# ---------------------------------------------------------------------------
# launch plumbing
#
# Triton's Python launcher spends 0.6-1 us per kernel argument on argument
# binding, cache-key construction and launch-metadata bookkeeping, which
# dominates the small-list workloads.  Once the ordinary path has compiled a
# (dtype, kind) specialization, its completed C launcher can be reused
# directly.  The fast path is only taken when the specialization is
# reproducible from cheap checks - every pointer 16B aligned, every numel
# divisible by 16 (so Triton still infers the divisibility that drives access
# vectorization) and no launch hooks registered - and it falls back to the
# ordinary launch for anything else.
# ---------------------------------------------------------------------------

_FAST = {}


def _current_stream():
    active = _driver.active
    return active.get_current_stream(active.get_current_device())


def _hooks_clear():
    enter = _knobs.runtime.launch_enter_hook
    exit_ = _knobs.runtime.launch_exit_hook
    if enter is None and exit_ is None:
        return True
    for chain in (enter, exit_):
        calls = getattr(chain, "calls", None)
        if calls:
            return False
    return True


def _capture(key, kernel):
    try:
        launcher = kernel.run
        fn = kernel.function
        packed = kernel.packed_metadata
        if (
            getattr(launcher, "global_scratch_size", 1) == 0
            and getattr(launcher, "profile_scratch_size", 1) == 0
        ):
            launch = launcher.launch

            def go(g0, g1, g2, stream, argvals):
                launch(
                    g0,
                    g1,
                    g2,
                    stream,
                    fn,
                    None,
                    None,
                    packed,
                    None,
                    None,
                    None,
                    *argvals,
                )

        else:

            def go(g0, g1, g2, stream, argvals):
                launcher(g0, g1, g2, stream, fn, packed, None, None, None, *argvals)

        _FAST[key] = go
    except Exception:
        _FAST[key] = None


def _launch_single(t, n, por):
    grid = (n + _BLOCK - 1) // _BLOCK
    cacheable = ((por & 15) == 0) and ((n & 15) == 0) and n <= _INT32_MAX
    key = (t.dtype, 0)
    if cacheable:
        fast = _FAST.get(key)
        if fast is not None and _hooks_clear():
            try:
                fast(grid, 1, 1, _current_stream(), (t, n, _BLOCK))
                return
            except Exception:
                _FAST[key] = None
    kernel = _neg_flat[(grid,)](t, n, _BLOCK, num_warps=_WARPS)
    if cacheable and _hooks_clear():
        _capture(key, kernel)


def _launch_multi(lst, numels, peak, por, nor):
    cnt = len(lst)
    args = [None] * (2 * _SLOTS + 1)
    args[0 : 2 * cnt : 2] = lst
    args[1 : 2 * cnt : 2] = numels
    last = lst[cnt - 1]
    for i in range(2 * cnt, 2 * _SLOTS, 2):
        args[i] = last
        args[i + 1] = 0
    args[2 * _SLOTS] = _BLOCK
    grid = (peak + _BLOCK - 1) // _BLOCK
    cacheable = ((por & 15) == 0) and ((nor & 15) == 0) and peak <= _INT32_MAX
    key = (lst[0].dtype, 1)
    if cacheable:
        fast = _FAST.get(key)
        if fast is not None and _hooks_clear():
            try:
                fast(grid, 1, 1, _current_stream(), args)
                return
            except Exception:
                _FAST[key] = None
    kernel = _neg_multi16[(grid,)](*args, num_warps=_WARPS)
    if cacheable and _hooks_clear():
        _capture(key, kernel)


def _launch_multi_u(lst, n, por):
    cnt = len(lst)
    args = [None] * (_SLOTS + 2)
    args[0:cnt] = lst
    last = lst[cnt - 1]
    for i in range(cnt, _SLOTS):
        args[i] = last
    args[_SLOTS] = n
    args[_SLOTS + 1] = _UBLOCK
    grid = (n + _UBLOCK - 1) // _UBLOCK
    cacheable = ((por & 15) == 0) and ((n & 15) == 0) and n <= _INT32_MAX
    key = (lst[0].dtype, 2)
    if cacheable:
        fast = _FAST.get(key)
        if fast is not None and _hooks_clear():
            try:
                fast(grid, cnt, 1, _current_stream(), args)
                return
            except Exception:
                _FAST[key] = None
    kernel = _neg_multi16u[(grid, cnt)](*args, num_warps=_WARPS)
    if cacheable and _hooks_clear():
        _capture(key, kernel)


def _canonical(shape, strides):
    """Fastest-first (size, stride) pairs, adjacent dims merged when dense."""
    dims = sorted(
        ((sz, st) for sz, st in zip(shape, strides) if sz != 1),
        key=lambda p: p[1],
    )
    out = []
    for sz, st in dims:
        if out:
            psz, pst = out[-1]
            if pst * psz == st:
                out[-1] = (psz * sz, pst)
                continue
        out.append((sz, st))
    return out


def _launch_dims(t, dims):
    n_in, s_in = dims[0]
    outer = dims[1:]
    if len(outer) > 8:
        shape = t.shape
        for i in range(shape[0]):
            _negate_tensor(t.select(0, i))
        return
    if len(outer) == 0 and s_in == 1:
        _launch_single(t, n_in, _t_ptr(t))
        return
    nb = triton.next_power_of_2(n_in)
    blk = nb if nb < _BLOCK else _BLOCK
    if blk < 32:
        blk = 32
    warps = _WARPS if blk >= 256 else 1
    padded = list(outer) + [(1, 0)] * (8 - len(outer))
    outer_count = 1
    for sz, _ in outer:
        outer_count *= sz
    args = [t, n_in, s_in]
    for sz, st in padded:
        args.append(sz)
        args.append(st)
    _neg_strided[((n_in + blk - 1) // blk, outer_count)](*args, blk, num_warps=warps)


def _negate_tensor(t):
    n = _t_numel(t)
    if n == 0:
        return
    dtype = t.dtype
    if dtype.is_complex:
        if _t_contig(t):
            _launch_single(t.view(_REAL_OF[dtype]), n * 2, _t_ptr(t))
        else:
            _negate_tensor(t.real)
            _negate_tensor(t.imag)
        return
    if _t_contig(t):
        _launch_single(t, n, _t_ptr(t))
        return
    shape = t.shape
    strides = t.stride()
    for sz, st in zip(shape, strides):
        if st == 0 and sz > 1:
            raise RuntimeError(_OVERLAP_MSG)
    _launch_dims(t, _canonical(shape, strides))


def foreach_neg_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    nt = len(self)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Fast path: one homogeneous dtype group of flat-addressable tensors that
    # fits in a single launch.
    if nt <= _SLOTS:
        t0 = self[0]
        if type(t0) is torch.Tensor:
            dt0 = t0.dtype
            if dt0 is not torch.bool and not dt0.is_complex:
                try:
                    lst = []
                    numels = []
                    por = 0
                    nor = 0
                    ok = True
                    for t in self:
                        if t.dtype is not dt0 or not _t_contig(t):
                            ok = False
                            break
                        n = _t_numel(t)
                        lst.append(t)
                        numels.append(n)
                        por |= _t_ptr(t)
                        nor |= n
                    if ok:
                        first = numels[0]
                        if numels.count(first) == nt:
                            if first != 0:
                                if nt == 1:
                                    _launch_single(t0, first, por)
                                else:
                                    _launch_multi_u(lst, first, por)
                        else:
                            peak = max(numels)
                            if peak != 0:
                                _launch_multi(lst, numels, peak, por, nor)
                        return None
                except (AttributeError, TypeError):
                    pass

    groups = {}
    slow = None
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt == torch.bool:
            raise RuntimeError("\"foreach\" not implemented for 'bool'")
        if dt.is_complex or not _t_contig(t):
            if slow is None:
                slow = [t]
            else:
                slow.append(t)
        else:
            e = groups.get(dt)
            n = _t_numel(t)
            if e is None:
                groups[dt] = [[t], [n], _t_ptr(t), n]
            else:
                e[0].append(t)
                e[1].append(n)
                e[2] |= _t_ptr(t)
                e[3] |= n

    for lst, numels, por, nor in groups.values():
        peak = max(numels)
        if peak == 0:
            continue
        cnt = len(lst)
        i = 0
        while i < cnt:
            j = i + _SLOTS
            if j > cnt:
                j = cnt
            sub = lst[i:j]
            sn = numels[i:j]
            if j - i == 1:
                _launch_single(sub[0], sn[0], por)
            else:
                _launch_multi(sub, sn, max(sn), por, nor)
            i = j

    if slow is not None:
        for t in slow:
            _negate_tensor(t)
    return None
