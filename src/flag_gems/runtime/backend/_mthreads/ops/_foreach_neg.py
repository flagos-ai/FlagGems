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
"""aten::_foreach_neg: negate every tensor of a TensorList, returning one
freshly allocated tensor per input (Triton kernels).

Host dispatch groups the contiguous, same-dtype members of one call into a
single multi-tensor launch (up to _KMAX tensors per launch) so that long lists
of small tensors do not pay one launch each.  Non-contiguous members use a
stride-aware kernel; empty members allocate and launch nothing.
"""

import itertools
import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_unary import _foreach_neg as default__foreach_neg

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)
_BLOCK = 1024
_WARPS = 4
_MAX_OUTER = 7
_KMAX = 16

_BOOL_MSG = (
    "Negation, the `-` operator, on a bool tensor is not supported. "
    "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
)


@triton.jit
def _neg_flat_kernel(in_ptr, out_ptr, numel, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(in_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
        tl.store(
            out_ptr + offs, -x, cache_modifier=".cs", eviction_policy="evict_first"
        )
    else:
        mask = offs < numel
        x = tl.load(
            in_ptr + offs,
            mask=mask,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + offs,
            -x,
            mask=mask,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


@triton.jit
def _neg_slice(in_ptr, out_ptr, numel, pid0, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(in_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
        tl.store(
            out_ptr + offs, -x, cache_modifier=".cs", eviction_policy="evict_first"
        )
    else:
        mask = offs < numel
        x = tl.load(
            in_ptr + offs,
            mask=mask,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.store(
            out_ptr + offs,
            -x,
            mask=mask,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


@triton.jit
def _neg_multi_kernel(
    a0,
    o0,
    n0,
    a1,
    o1,
    n1,
    a2,
    o2,
    n2,
    a3,
    o3,
    n3,
    a4,
    o4,
    n4,
    a5,
    o5,
    n5,
    a6,
    o6,
    n6,
    a7,
    o7,
    n7,
    a8,
    o8,
    n8,
    a9,
    o9,
    n9,
    a10,
    o10,
    n10,
    a11,
    o11,
    n11,
    a12,
    o12,
    n12,
    a13,
    o13,
    n13,
    a14,
    o14,
    n14,
    a15,
    o15,
    n15,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    c0 = (n0 + BLOCK - 1) // BLOCK
    c1 = c0 + (n1 + BLOCK - 1) // BLOCK
    c2 = c1 + (n2 + BLOCK - 1) // BLOCK
    c3 = c2 + (n3 + BLOCK - 1) // BLOCK
    c4 = c3 + (n4 + BLOCK - 1) // BLOCK
    c5 = c4 + (n5 + BLOCK - 1) // BLOCK
    c6 = c5 + (n6 + BLOCK - 1) // BLOCK
    c7 = c6 + (n7 + BLOCK - 1) // BLOCK
    c8 = c7 + (n8 + BLOCK - 1) // BLOCK
    c9 = c8 + (n9 + BLOCK - 1) // BLOCK
    c10 = c9 + (n10 + BLOCK - 1) // BLOCK
    c11 = c10 + (n11 + BLOCK - 1) // BLOCK
    c12 = c11 + (n12 + BLOCK - 1) // BLOCK
    c13 = c12 + (n13 + BLOCK - 1) // BLOCK
    c14 = c13 + (n14 + BLOCK - 1) // BLOCK
    if pid < c0:
        _neg_slice(a0, o0, n0, pid, BLOCK, EVEN)
    elif pid < c1:
        _neg_slice(a1, o1, n1, pid - c0, BLOCK, EVEN)
    elif pid < c2:
        _neg_slice(a2, o2, n2, pid - c1, BLOCK, EVEN)
    elif pid < c3:
        _neg_slice(a3, o3, n3, pid - c2, BLOCK, EVEN)
    elif pid < c4:
        _neg_slice(a4, o4, n4, pid - c3, BLOCK, EVEN)
    elif pid < c5:
        _neg_slice(a5, o5, n5, pid - c4, BLOCK, EVEN)
    elif pid < c6:
        _neg_slice(a6, o6, n6, pid - c5, BLOCK, EVEN)
    elif pid < c7:
        _neg_slice(a7, o7, n7, pid - c6, BLOCK, EVEN)
    elif pid < c8:
        _neg_slice(a8, o8, n8, pid - c7, BLOCK, EVEN)
    elif pid < c9:
        _neg_slice(a9, o9, n9, pid - c8, BLOCK, EVEN)
    elif pid < c10:
        _neg_slice(a10, o10, n10, pid - c9, BLOCK, EVEN)
    elif pid < c11:
        _neg_slice(a11, o11, n11, pid - c10, BLOCK, EVEN)
    elif pid < c12:
        _neg_slice(a12, o12, n12, pid - c11, BLOCK, EVEN)
    elif pid < c13:
        _neg_slice(a13, o13, n13, pid - c12, BLOCK, EVEN)
    elif pid < c14:
        _neg_slice(a14, o14, n14, pid - c13, BLOCK, EVEN)
    else:
        _neg_slice(a15, o15, n15, pid - c14, BLOCK, EVEN)


@triton.jit
def _neg_multi_uni_kernel(
    a0,
    o0,
    a1,
    o1,
    a2,
    o2,
    a3,
    o3,
    a4,
    o4,
    a5,
    o5,
    a6,
    o6,
    a7,
    o7,
    a8,
    o8,
    a9,
    o9,
    a10,
    o10,
    a11,
    o11,
    a12,
    o12,
    a13,
    o13,
    a14,
    o14,
    a15,
    o15,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch for a batch of equal-length contiguous tensors.

    Slot = pid // blocks_per_tensor, block = pid % blocks_per_tensor.  The
    shared numel keeps the argument count at 33 instead of 3 * _KMAX.
    """
    pid = tl.program_id(0)
    nb = (numel + BLOCK - 1) // BLOCK
    tid = pid // nb
    blk = pid % nb
    if tid == 0:
        _neg_slice(a0, o0, numel, blk, BLOCK, EVEN)
    elif tid == 1:
        _neg_slice(a1, o1, numel, blk, BLOCK, EVEN)
    elif tid == 2:
        _neg_slice(a2, o2, numel, blk, BLOCK, EVEN)
    elif tid == 3:
        _neg_slice(a3, o3, numel, blk, BLOCK, EVEN)
    elif tid == 4:
        _neg_slice(a4, o4, numel, blk, BLOCK, EVEN)
    elif tid == 5:
        _neg_slice(a5, o5, numel, blk, BLOCK, EVEN)
    elif tid == 6:
        _neg_slice(a6, o6, numel, blk, BLOCK, EVEN)
    elif tid == 7:
        _neg_slice(a7, o7, numel, blk, BLOCK, EVEN)
    elif tid == 8:
        _neg_slice(a8, o8, numel, blk, BLOCK, EVEN)
    elif tid == 9:
        _neg_slice(a9, o9, numel, blk, BLOCK, EVEN)
    elif tid == 10:
        _neg_slice(a10, o10, numel, blk, BLOCK, EVEN)
    elif tid == 11:
        _neg_slice(a11, o11, numel, blk, BLOCK, EVEN)
    elif tid == 12:
        _neg_slice(a12, o12, numel, blk, BLOCK, EVEN)
    elif tid == 13:
        _neg_slice(a13, o13, numel, blk, BLOCK, EVEN)
    elif tid == 14:
        _neg_slice(a14, o14, numel, blk, BLOCK, EVEN)
    else:
        _neg_slice(a15, o15, numel, blk, BLOCK, EVEN)


@triton.jit
def _neg_strided_kernel(
    in_ptr,
    out_ptr,
    inner,
    nb,
    bi,
    bo,
    istride,
    ostride,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_o = pid // nb
    pid_i = pid % nb
    in_base = bi
    out_base = bo
    c = pid_o % s0
    pid_o = pid_o // s0
    in_base += c * i0
    out_base += c * o0
    c = pid_o % s1
    pid_o = pid_o // s1
    in_base += c * i1
    out_base += c * o1
    c = pid_o % s2
    pid_o = pid_o // s2
    in_base += c * i2
    out_base += c * o2
    c = pid_o % s3
    pid_o = pid_o // s3
    in_base += c * i3
    out_base += c * o3
    c = pid_o % s4
    pid_o = pid_o // s4
    in_base += c * i4
    out_base += c * o4
    c = pid_o % s5
    pid_o = pid_o // s5
    in_base += c * i5
    out_base += c * o5
    c = pid_o % s6
    pid_o = pid_o // s6
    in_base += c * i6
    out_base += c * o6

    offs = pid_i * BLOCK + tl.arange(0, BLOCK)
    in_off = in_base + offs * istride
    out_off = out_base + offs * ostride
    if EVEN:
        x = tl.load(in_ptr + in_off)
        tl.store(out_ptr + out_off, -x)
    else:
        mask = offs < inner
        x = tl.load(in_ptr + in_off, mask=mask)
        tl.store(out_ptr + out_off, -x, mask=mask)


def _plan(src, dst):
    """Collapse the layout into one innermost dim plus zero or more outer dims.

    Returns (inner[n, istride, ostride], outer[[n, istride, ostride], ...])
    ordered so that the innermost enumeration step has the smallest input
    stride (unit stride whenever the layout offers one).
    """
    dims = []
    for d in range(src.dim()):
        n = src.size(d)
        if n == 1:
            continue
        dims.append([n, src.stride(d), dst.stride(d)])
    dims = sorted(dims, key=lambda x: (x[1], x[2]))
    merged = []
    for d in dims:
        if merged:
            p = merged[-1]
            if p[1] == d[1] * d[0] and p[2] == d[2] * d[0]:
                p[0] *= d[0]
                continue
        merged.append(d)
    return merged[0], merged[1:]


def _launch_strided_block(
    src, dst, inner_n, nb, bi, bo, is_in, os_in, outer, even, nprog
):
    ss = [1] * _MAX_OUTER
    ii = [0] * _MAX_OUTER
    oo = [0] * _MAX_OUTER
    for k, d in enumerate(outer):
        ss[k] = d[0]
        ii[k] = d[1]
        oo[k] = d[2]
    _neg_strided_kernel[(nprog,)](
        src,
        dst,
        inner_n,
        nb,
        bi,
        bo,
        is_in,
        os_in,
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ii[0],
        ii[1],
        ii[2],
        ii[3],
        ii[4],
        ii[5],
        ii[6],
        oo[0],
        oo[1],
        oo[2],
        oo[3],
        oo[4],
        oo[5],
        oo[6],
        BLOCK=_BLOCK,
        EVEN=even,
    )


_FAST = {}


def _dispatch(jit_fn, grid, args, const, num_warps, key):
    """Launch ``jit_fn``, replaying the compiled variant once it exists.

    Triton's Python-level launch costs ~10 us on this target, more than the
    kernel itself for short lists, so the CompiledKernel is replayed directly.
    ``key`` carries everything the compile specialisation depends on (dtype,
    int specialisation, pointer alignment); a stale handle falls back to the
    ordinary launch path.
    """
    entry = _FAST.get(key)
    if entry is not None:
        compiled, stream = entry
        try:
            compiled.run(
                grid[0],
                grid[1] if len(grid) > 1 else 1,
                1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                *const.values(),
            )
            return
        except Exception:
            _FAST.pop(key, None)
    compiled = jit_fn[grid](*args, num_warps=num_warps, **const)
    try:
        drv = _driver.active
        _FAST[key] = (compiled, drv.get_current_stream(drv.get_current_device()))
    except Exception:
        _FAST.pop(key, None)


def _spec_int(v):
    return (v == 1, v % 16 == 0)


def _neg_contig(src, dst):
    numel = src.numel()
    nb = (numel + _BLOCK - 1) // _BLOCK
    even = numel % _BLOCK == 0
    key = (
        "flat",
        src.dtype,
        _spec_int(numel),
        src.data_ptr() & 15,
        dst.data_ptr() & 15,
        even,
    )
    _dispatch(
        _neg_flat_kernel,
        (nb,),
        (src, dst, numel),
        {"BLOCK": _BLOCK, "EVEN": even},
        _WARPS,
        key,
    )


def _neg_strided(src, dst):
    inner, outer = _plan(src, dst)
    inner_n, is_in, os_in = inner
    nb = (inner_n + _BLOCK - 1) // _BLOCK
    even = inner_n % _BLOCK == 0
    if len(outer) <= _MAX_OUTER:
        nprog = nb
        for d in outer:
            nprog *= d[0]
        _launch_strided_block(
            src, dst, inner_n, nb, 0, 0, is_in, os_in, outer, even, nprog
        )
        return

    rest = outer[:_MAX_OUTER]
    lead = outer[_MAX_OUTER:]
    nprog = nb
    for d in rest:
        nprog *= d[0]
    for combo in itertools.product(*[range(d[0]) for d in lead]):
        io = 0
        oo = 0
        for c, d in zip(combo, lead):
            io += c * d[1]
            oo += c * d[2]
        _launch_strided_block(
            src, dst, inner_n, nb, io, oo, is_in, os_in, rest, even, nprog
        )


def _neg_group_uni(entries):
    """One 33-argument launch for a batch of equal-length contiguous tensors."""
    n = entries[0][0].numel()
    nb = (n + _BLOCK - 1) // _BLOCK
    args = []
    fill = entries[0]
    for k in range(_KMAX):
        if k < len(entries):
            s, d = entries[k]
            args.append(s)
            args.append(d)
        else:
            args.append(fill[0])
            args.append(fill[1])
    args.append(n)
    even = n % _BLOCK == 0
    aligns = tuple(p & 15 for s, d in entries for p in (s.data_ptr(), d.data_ptr()))
    key = (
        "uni",
        entries[0][0].dtype,
        len(entries),
        _spec_int(n),
        aligns,
        even,
    )
    _dispatch(
        _neg_multi_uni_kernel,
        (nb * len(entries),),
        tuple(args),
        {"BLOCK": _BLOCK, "EVEN": even},
        _WARPS,
        key,
    )


def _neg_group(entries):
    """Negate a batch of contiguous, same-dtype tensors with one launch."""
    total = 0
    even = True
    first = entries[0][0].numel()
    uniform = True
    for src, _ in entries:
        n = src.numel()
        total += (n + _BLOCK - 1) // _BLOCK
        if n % _BLOCK:
            even = False
        if n != first:
            uniform = False
    if uniform:
        _neg_group_uni(entries)
        return
    args = []
    fill = entries[0]
    for k in range(_KMAX):
        if k < len(entries):
            s, d = entries[k]
            args.append(s)
            args.append(d)
            args.append(s.numel())
        else:
            args.append(fill[0])
            args.append(fill[1])
            args.append(0)
    key = (
        "multi",
        entries[0][0].dtype,
        tuple(_spec_int(s.numel()) for s, _ in entries),
        tuple(p & 15 for s, d in entries for p in (s.data_ptr(), d.data_ptr())),
        even,
    )
    _dispatch(
        _neg_multi_kernel,
        (total,),
        tuple(args),
        {"BLOCK": _BLOCK, "EVEN": even},
        _WARPS,
        key,
    )


def _run_neg(seq):
    out = []
    groups = {}
    for t in seq:
        dt = t.dtype
        if dt == torch.bool:
            raise RuntimeError(_BOOL_MSG)
        dst = torch.empty_like(t)
        out.append(dst)
        if t.numel() == 0:
            continue
        src = t
        if dt.is_complex:
            fdt = torch.float32 if dt is torch.complex64 else torch.float64
            try:
                src = t.view(fdt)
                dst_v = dst.view(fdt)
            except RuntimeError:
                _neg_contig(t.real, dst.real)
                _neg_contig(t.imag, dst.imag)
                continue
        else:
            dst_v = dst
        if src.is_contiguous():
            groups.setdefault(src.dtype, []).append((src, dst_v))
        elif dt.is_complex:
            _neg_strided(t.real, dst.real)
            _neg_strided(t.imag, dst.imag)
        else:
            _neg_strided(src, dst_v)
    for entries in groups.values():
        # The fused launch has _KMAX pointer slots; a longer list must be
        # chunked or the tail tensors are never written.
        for start in range(0, len(entries), _KMAX):
            part = entries[start : start + _KMAX]
            if len(part) == 1:
                _neg_contig(part[0][0], part[0][1])
            else:
                _neg_group(part)
    return out


def _foreach_neg(self=None, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_NEG")
    if "tensors" in kwargs and self is None and not args:
        self = kwargs.pop("tensors")
    if args:
        seq = [] if self is None else [self]
        seq.extend(args)
    elif isinstance(self, (list, tuple)):
        seq = self
    elif self is None:
        seq = []
    else:
        seq = [self]
    if len(seq) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in seq):
        return default__foreach_neg(seq)
    return _run_neg(seq)
