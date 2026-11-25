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

"""Triton implementation of aten::_foreach_exp(Tensor[] self) -> Tensor[].

Each tensor is exponentiated elementwise into a freshly allocated, contiguous
output of the same shape; floating/complex dtypes are preserved and integral or
bool inputs are promoted to the default dtype (matching ATen's slow foreach
route).

Dispatch:
  * launch-bound contiguous tensors (numel <= _FUSE_MAX_NUMEL) are fused into
    one multi-slot kernel launch per (device, dtype) group, and >= 4-byte
    elements may always be fused because an 8KB block hides the slot-selection
    prologue;
  * large tensors use a streaming kernel with a per-dtype 16-byte-per-thread
    tile (measured winner; no per-call tuner dispatch);
  * other layouts use a generic collapsed-groups index decomposition;
  * complex tensors are handled as interleaved real pairs
    (exp(a)*(cos b + i sin b)).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_GROUPS = 8
_MAX_SLOTS = 16
_LARGE_NUMEL = 1 << 31
# The fused multi-slot launch executes its O(slot-depth) boundary/branch
# prologue once per block; that only pays off while it is hidden behind the
# block's memory time.  Round-2 profiling measured a 22% regression for 16-bit
# 16.7M-element lists (57.1us vs 46.7us per tensor), so those go back to
# per-tensor launches.
_FUSE_MAX_NUMEL = 1 << 20
# Above this size the streaming tile is chosen from the element size.
_TUNE_MIN_NUMEL = 1 << 22
# Profile evidence (rounds 1, 4 and 5): the winning streaming tile is ~16 bytes
# per thread; with 4-warp (256-thread) blocks that is BLOCK = 4096/itemsize
# (1024 fp32, 2048 16-bit), measured best in round 5 and better than 32 bytes
# per thread in round 6.  Round 7 keeps the 16-byte width but doubles the block
# to 512 threads to isolate the block-count dimension.
_BIG_BLOCK = {1: 8192, 2: 4096, 4: 2048, 8: 1024}
_BIG_WARPS = 8


@triton.jit
def _exp1(v, IS_F64: tl.constexpr):
    if IS_F64:
        r = tl.exp(v)
    else:
        r = tl.exp(v.to(tl.float32))
    return r


@triton.jit
def _flat_body(
    x_ptr, o_ptr, n, pid, BLOCK: tl.constexpr, IS_F64: tl.constexpr, PAIR: tl.constexpr
):
    base = pid * BLOCK
    idx = base + tl.arange(0, BLOCK)
    if base + BLOCK <= n:
        if PAIR:
            a = tl.load(x_ptr + 2 * idx)
            b = tl.load(x_ptr + 2 * idx + 1)
            ea = _exp1(a, IS_F64)
            tl.store(o_ptr + 2 * idx, ea * tl.cos(b))
            tl.store(o_ptr + 2 * idx + 1, ea * tl.sin(b))
        else:
            tl.store(o_ptr + idx, _exp1(tl.load(x_ptr + idx), IS_F64))
    else:
        mask = idx < n
        if PAIR:
            a = tl.load(x_ptr + 2 * idx, mask=mask)
            b = tl.load(x_ptr + 2 * idx + 1, mask=mask)
            ea = _exp1(a, IS_F64)
            tl.store(o_ptr + 2 * idx, ea * tl.cos(b), mask=mask)
            tl.store(o_ptr + 2 * idx + 1, ea * tl.sin(b), mask=mask)
        else:
            v = tl.load(x_ptr + idx, mask=mask)
            tl.store(o_ptr + idx, _exp1(v, IS_F64), mask=mask)


@triton.jit
def _exp_flat_kernel(
    x_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    PAIR: tl.constexpr,
    LARGE: tl.constexpr,
):
    if LARGE:
        pid = tl.program_id(0)
        base = pid.to(tl.int64) * BLOCK
        idx = base + tl.arange(0, BLOCK)
        mask = idx < n
        if PAIR:
            a = tl.load(x_ptr + 2 * idx, mask=mask)
            b = tl.load(x_ptr + 2 * idx + 1, mask=mask)
            ea = _exp1(a, IS_F64)
            tl.store(o_ptr + 2 * idx, ea * tl.cos(b), mask=mask)
            tl.store(o_ptr + 2 * idx + 1, ea * tl.sin(b), mask=mask)
        else:
            v = tl.load(x_ptr + idx, mask=mask)
            tl.store(o_ptr + idx, _exp1(v, IS_F64), mask=mask)
    else:
        _flat_body(x_ptr, o_ptr, n, tl.program_id(0), BLOCK, IS_F64, PAIR)


@triton.jit
def _exp_flat_large_kernel(
    x_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    PAIR: tl.constexpr,
    LARGE: tl.constexpr,
):
    if LARGE:
        pid = tl.program_id(0)
        base = pid.to(tl.int64) * BLOCK
        idx = base + tl.arange(0, BLOCK)
        mask = idx < n
        if PAIR:
            a = tl.load(x_ptr + 2 * idx, mask=mask)
            b = tl.load(x_ptr + 2 * idx + 1, mask=mask)
            ea = _exp1(a, IS_F64)
            tl.store(o_ptr + 2 * idx, ea * tl.cos(b), mask=mask)
            tl.store(o_ptr + 2 * idx + 1, ea * tl.sin(b), mask=mask)
        else:
            v = tl.load(x_ptr + idx, mask=mask)
            tl.store(o_ptr + idx, _exp1(v, IS_F64), mask=mask)
    else:
        _flat_body(x_ptr, o_ptr, n, tl.program_id(0), BLOCK, IS_F64, PAIR)


@triton.jit
def _exp_multi_kernel(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    x4,
    o4,
    x5,
    o5,
    x6,
    o6,
    x7,
    o7,
    x8,
    o8,
    x9,
    o9,
    x10,
    o10,
    x11,
    o11,
    x12,
    o12,
    x13,
    o13,
    x14,
    o14,
    x15,
    o15,
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
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    PAIR: tl.constexpr,
):
    """One launch for up to 16 contiguous tensors of the same dtype/device.

    Grid = sum of per-slot block counts; program ``pid`` belongs to the first
    slot whose cumulative block count exceeds it.  Unused slots are passed with
    numel 0, so their cumulative range is empty and they are never selected.
    """
    pid = tl.program_id(0)
    c0 = tl.cdiv(n0, BLOCK)
    if pid < c0:
        _flat_body(x0, o0, n0, pid, BLOCK, IS_F64, PAIR)
    else:
        c1 = c0 + tl.cdiv(n1, BLOCK)
        if pid < c1:
            _flat_body(x1, o1, n1, pid - c0, BLOCK, IS_F64, PAIR)
        else:
            c2 = c1 + tl.cdiv(n2, BLOCK)
            if pid < c2:
                _flat_body(x2, o2, n2, pid - c1, BLOCK, IS_F64, PAIR)
            else:
                c3 = c2 + tl.cdiv(n3, BLOCK)
                if pid < c3:
                    _flat_body(x3, o3, n3, pid - c2, BLOCK, IS_F64, PAIR)
                else:
                    c4 = c3 + tl.cdiv(n4, BLOCK)
                    if pid < c4:
                        _flat_body(x4, o4, n4, pid - c3, BLOCK, IS_F64, PAIR)
                    else:
                        c5 = c4 + tl.cdiv(n5, BLOCK)
                        if pid < c5:
                            _flat_body(x5, o5, n5, pid - c4, BLOCK, IS_F64, PAIR)
                        else:
                            c6 = c5 + tl.cdiv(n6, BLOCK)
                            if pid < c6:
                                _flat_body(x6, o6, n6, pid - c5, BLOCK, IS_F64, PAIR)
                            else:
                                c7 = c6 + tl.cdiv(n7, BLOCK)
                                if pid < c7:
                                    _flat_body(
                                        x7, o7, n7, pid - c6, BLOCK, IS_F64, PAIR
                                    )
                                else:
                                    c8 = c7 + tl.cdiv(n8, BLOCK)
                                    if pid < c8:
                                        _flat_body(
                                            x8, o8, n8, pid - c7, BLOCK, IS_F64, PAIR
                                        )
                                    else:
                                        c9 = c8 + tl.cdiv(n9, BLOCK)
                                        if pid < c9:
                                            _flat_body(
                                                x9,
                                                o9,
                                                n9,
                                                pid - c8,
                                                BLOCK,
                                                IS_F64,
                                                PAIR,
                                            )
                                        else:
                                            c10 = c9 + tl.cdiv(n10, BLOCK)
                                            if pid < c10:
                                                _flat_body(
                                                    x10,
                                                    o10,
                                                    n10,
                                                    pid - c9,
                                                    BLOCK,
                                                    IS_F64,
                                                    PAIR,
                                                )
                                            else:
                                                c11 = c10 + tl.cdiv(n11, BLOCK)
                                                if pid < c11:
                                                    _flat_body(
                                                        x11,
                                                        o11,
                                                        n11,
                                                        pid - c10,
                                                        BLOCK,
                                                        IS_F64,
                                                        PAIR,
                                                    )
                                                else:
                                                    c12 = c11 + tl.cdiv(n12, BLOCK)
                                                    if pid < c12:
                                                        _flat_body(
                                                            x12,
                                                            o12,
                                                            n12,
                                                            pid - c11,
                                                            BLOCK,
                                                            IS_F64,
                                                            PAIR,
                                                        )
                                                    else:
                                                        c13 = c12 + tl.cdiv(n13, BLOCK)
                                                        if pid < c13:
                                                            _flat_body(
                                                                x13,
                                                                o13,
                                                                n13,
                                                                pid - c12,
                                                                BLOCK,
                                                                IS_F64,
                                                                PAIR,
                                                            )
                                                        else:
                                                            c14 = c13 + tl.cdiv(
                                                                n14, BLOCK
                                                            )
                                                            if pid < c14:
                                                                _flat_body(
                                                                    x14,
                                                                    o14,
                                                                    n14,
                                                                    pid - c13,
                                                                    BLOCK,
                                                                    IS_F64,
                                                                    PAIR,
                                                                )
                                                            else:
                                                                if pid < c14 + tl.cdiv(
                                                                    n15, BLOCK
                                                                ):
                                                                    _flat_body(
                                                                        x15,
                                                                        o15,
                                                                        n15,
                                                                        pid - c14,
                                                                        BLOCK,
                                                                        IS_F64,
                                                                        PAIR,
                                                                    )


@triton.jit
def _strided_body(
    x_ptr,
    o_ptr,
    xo,
    idx,
    n,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    PAIR: tl.constexpr,
):
    mask = idx < n
    if PAIR:
        a = tl.load(x_ptr + 2 * xo, mask=mask)
        b = tl.load(x_ptr + 2 * xo + 1, mask=mask)
        ea = _exp1(a, IS_F64)
        tl.store(o_ptr + 2 * idx, ea * tl.cos(b), mask=mask)
        tl.store(o_ptr + 2 * idx + 1, ea * tl.sin(b), mask=mask)
    else:
        v = tl.load(x_ptr + xo, mask=mask)
        tl.store(o_ptr + idx, _exp1(v, IS_F64), mask=mask)


@triton.jit
def _exp_nd_kernel(
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
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    PAIR: tl.constexpr,
    LARGE: tl.constexpr,
):
    """Elementwise exp over a strided input; the output is contiguous.

    ``sK``/``tK`` are the size/stride of the K-th collapsed group ordered from
    the innermost (fastest) group outwards.
    """
    pid = tl.program_id(0)
    if LARGE:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    idx = base + tl.arange(0, BLOCK)
    r = idx
    xo = tl.zeros((BLOCK,), dtype=tl.int64)
    i = r % s0
    r = r // s0
    xo += i.to(tl.int64) * t0
    if ND > 1:
        i = r % s1
        r = r // s1
        xo += i.to(tl.int64) * t1
    if ND > 2:
        i = r % s2
        r = r // s2
        xo += i.to(tl.int64) * t2
    if ND > 3:
        i = r % s3
        r = r // s3
        xo += i.to(tl.int64) * t3
    if ND > 4:
        i = r % s4
        r = r // s4
        xo += i.to(tl.int64) * t4
    if ND > 5:
        i = r % s5
        r = r // s5
        xo += i.to(tl.int64) * t5
    if ND > 6:
        i = r % s6
        r = r // s6
        xo += i.to(tl.int64) * t6
    if ND > 7:
        i = r % s7
        r = r // s7
        xo += i.to(tl.int64) * t7
    _strided_body(x_ptr, o_ptr, xo, idx, n, BLOCK, IS_F64, PAIR)


def _collapse_groups(t):
    """Collapse logically adjacent dims into (size, stride) groups, outer first."""
    shape = t.shape
    strides = t.stride()
    pairs = [(int(sz), int(st)) for sz, st in zip(shape, strides) if int(sz) != 1]
    if not pairs:
        return [(1, 0)]
    merged = [pairs[-1]]
    for sz, st in reversed(pairs[:-1]):
        last_sz, last_st = merged[-1]
        if st == last_st * last_sz:
            merged[-1] = (sz * last_sz, last_st)
        else:
            merged.append((sz, st))
    merged.reverse()
    return merged


def _launch_fused(chunk, is_f64, pair):
    """chunk: list of (xv, ov, n) with identical dtype/device and contiguous."""
    xs = [c[0] for c in chunk]
    os_ = [c[1] for c in chunk]
    ns = [c[2] for c in chunk]
    k = len(chunk)
    pad = _MAX_SLOTS - k
    xp = xs + [xs[-1]] * pad
    op = os_ + [os_[-1]] * pad
    np_ = ns + [0] * pad
    total = 0
    for n in ns:
        total += (n + _BLOCK - 1) // _BLOCK
    args = []
    for a, b in zip(xp, op):
        args.append(a)
        args.append(b)
    args.extend(np_)
    _exp_multi_kernel[(total,)](
        *args,
        BLOCK=_BLOCK,
        IS_F64=is_f64,
        PAIR=pair,
        num_warps=_NUM_WARPS,
    )


def _exp_tensor(t, out):
    n = int(t.numel())
    if n == 0:
        return
    dt = t.dtype
    pair = False
    if t.is_complex():
        pair_dtype = torch.float64 if dt is torch.complex128 else torch.float32
        is_f64 = pair_dtype is torch.float64
        xv = t.view(pair_dtype)
        ov = out.view(pair_dtype)
        pair = True
    elif dt is torch.float64:
        is_f64 = True
        xv, ov = t, out
    else:
        is_f64 = False
        # same-itemsize view keeps bool tensors loadable as 0/1 integers
        xv = t.view(torch.uint8) if dt is torch.bool else t
        ov = out
    large = n > _LARGE_NUMEL

    if t.is_contiguous():
        if n > _TUNE_MIN_NUMEL:
            blk = _BIG_BLOCK.get(xv.element_size(), 1024)
            _exp_flat_large_kernel[(triton.cdiv(n, blk),)](
                xv,
                ov,
                n,
                BLOCK=blk,
                IS_F64=is_f64,
                PAIR=pair,
                LARGE=large,
                num_warps=_BIG_WARPS,
            )
            return
        _exp_flat_kernel[(triton.cdiv(n, _BLOCK),)](
            xv,
            ov,
            n,
            BLOCK=_BLOCK,
            IS_F64=is_f64,
            PAIR=pair,
            LARGE=large,
            num_warps=_NUM_WARPS,
        )
        return

    groups = _collapse_groups(t)
    if len(groups) > _MAX_GROUPS:
        raise NotImplementedError(
            "foreach_exp: unsupported tensor layout with %d stride groups" % len(groups)
        )
    groups = groups[::-1]  # innermost first
    sizes = [g[0] for g in groups] + [1] * (_MAX_GROUPS - len(groups))
    steps = [g[1] for g in groups] + [0] * (_MAX_GROUPS - len(groups))
    _exp_nd_kernel[(triton.cdiv(n, _BLOCK),)](
        xv,
        ov,
        n,
        *sizes,
        *steps,
        ND=len(groups),
        BLOCK=_BLOCK,
        IS_F64=is_f64,
        PAIR=pair,
        LARGE=large,
        num_warps=_NUM_WARPS,
    )


def _extract(args, kwargs):
    if len(args) == 1:
        return args[0]
    if len(args) > 1:
        return list(args)
    if len(kwargs) == 1:
        return next(iter(kwargs.values()))
    if len(kwargs) > 1:
        key = "self" if "self" in kwargs else next(iter(kwargs))
        return kwargs[key]
    raise TypeError("run() missing 1 required positional argument: 'self'")


def foreach_exp(*args, **kwargs):
    self = _extract(args, kwargs)

    if isinstance(self, torch.Tensor):
        raise TypeError(
            "_foreach_exp(): argument 'self' (position 1) must be tuple of "
            "Tensors, not Tensor"
        )
    tensors = list(self)
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = []
    fused = {}
    singles = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "_foreach_exp(): argument 'self' must be a list of Tensors, "
                "but found %s" % type(t).__name__
            )
        dt = t.dtype
        if t.is_complex() or dt.is_floating_point:
            out_dtype = dt
        else:
            out_dtype = torch.get_default_dtype()
        out = torch.empty(t.shape, dtype=out_dtype, device=t.device)
        outs.append(out)
        n = int(t.numel())
        if n == 0:
            continue
        if t.is_contiguous() and (n <= _FUSE_MAX_NUMEL or t.element_size() >= 4):
            key = (t.device, dt, t.is_complex())
            fused.setdefault(key, []).append((t, out, n))
        else:
            singles.append((t, out, n))

    for (dev, dt, is_complex), items in fused.items():
        if len(items) == 1:
            singles.append(items[0])
            continue
        is_f64 = dt is torch.float64
        pair = bool(is_complex)
        for start in range(0, len(items), _MAX_SLOTS):
            chunk = items[start : start + _MAX_SLOTS]
            if len(chunk) == 1:
                t, out, n = chunk[0]
                _exp_tensor(t, out)
            else:
                _launch_fused(chunk, is_f64, pair)

    for t, out, n in singles:
        _exp_tensor(t, out)

    return outs
