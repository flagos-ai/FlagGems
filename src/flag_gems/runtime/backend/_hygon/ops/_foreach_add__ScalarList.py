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
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared in-program tile helper
# ---------------------------------------------------------------------------


@triton.jit
def _tile_add(p, base, n, s, BLOCK: tl.constexpr):
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(p + offs, tl.load(p + offs, mask=mask) + s, mask=mask)


@triton.jit
def _tile_add64(p, base, n, sbits, BLOCK: tl.constexpr):
    s = sbits.to(tl.float64, bitcast=True)
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(p + offs, tl.load(p + offs, mask=mask) + s, mask=mask)


# ---------------------------------------------------------------------------
# Single-tensor contiguous kernels (used when a group holds only one tensor)
# ---------------------------------------------------------------------------


@triton.jit
def _add_c_f32(x_ptr, n, s, BLOCK: tl.constexpr):
    _tile_add(x_ptr, tl.program_id(0) * BLOCK, n, s, BLOCK)


@triton.jit
def _add_c_i64(x_ptr, n, s, BLOCK: tl.constexpr):
    _tile_add(x_ptr, tl.program_id(0) * BLOCK, n, s, BLOCK)


@triton.jit
def _add_c_f64(x_ptr, n, sbits, BLOCK: tl.constexpr):
    _tile_add64(x_ptr, tl.program_id(0) * BLOCK, n, sbits, BLOCK)


# ---------------------------------------------------------------------------
# Fused four-tensor contiguous kernels: one launch covers up to four tensors
# of the same dtype.  Unused slots are passed with n == 0 and are unreachable.
# ---------------------------------------------------------------------------


@triton.jit
def _fadd4_f32(
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = b0 + tl.cdiv(n1, BLOCK)
    b2 = b1 + tl.cdiv(n2, BLOCK)
    if pid < b0:
        _tile_add(x0, pid * BLOCK, n0, s0, BLOCK)
    elif pid < b1:
        _tile_add(x1, (pid - b0) * BLOCK, n1, s1, BLOCK)
    elif pid < b2:
        _tile_add(x2, (pid - b1) * BLOCK, n2, s2, BLOCK)
    else:
        _tile_add(x3, (pid - b2) * BLOCK, n3, s3, BLOCK)


@triton.jit
def _fadd4u_f32(x0, x1, x2, x3, n, s0, s1, s2, s3, BLOCK: tl.constexpr):
    """All four tensors share one element count, so one count argument suffices."""
    pid = tl.program_id(0)
    b = tl.cdiv(n, BLOCK)
    if pid < b:
        _tile_add(x0, pid * BLOCK, n, s0, BLOCK)
    elif pid < 2 * b:
        _tile_add(x1, (pid - b) * BLOCK, n, s1, BLOCK)
    elif pid < 3 * b:
        _tile_add(x2, (pid - 2 * b) * BLOCK, n, s2, BLOCK)
    else:
        _tile_add(x3, (pid - 3 * b) * BLOCK, n, s3, BLOCK)


@triton.jit
def _fadd4_i64(
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = b0 + tl.cdiv(n1, BLOCK)
    b2 = b1 + tl.cdiv(n2, BLOCK)
    if pid < b0:
        _tile_add(x0, pid * BLOCK, n0, s0, BLOCK)
    elif pid < b1:
        _tile_add(x1, (pid - b0) * BLOCK, n1, s1, BLOCK)
    elif pid < b2:
        _tile_add(x2, (pid - b1) * BLOCK, n2, s2, BLOCK)
    else:
        _tile_add(x3, (pid - b2) * BLOCK, n3, s3, BLOCK)


@triton.jit
def _fadd4_f64(
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b0 = tl.cdiv(n0, BLOCK)
    b1 = b0 + tl.cdiv(n1, BLOCK)
    b2 = b1 + tl.cdiv(n2, BLOCK)
    if pid < b0:
        _tile_add64(x0, pid * BLOCK, n0, s0, BLOCK)
    elif pid < b1:
        _tile_add64(x1, (pid - b0) * BLOCK, n1, s1, BLOCK)
    elif pid < b2:
        _tile_add64(x2, (pid - b1) * BLOCK, n2, s2, BLOCK)
    else:
        _tile_add64(x3, (pid - b2) * BLOCK, n3, s3, BLOCK)


# ---------------------------------------------------------------------------
# General strided kernels.  Each tensor view is collapsed host-side into an
# innermost run (inner_len, inner_stride) plus an outer index space whose dims
# are passed as constexpr tuples and decomposed once per program.
# ---------------------------------------------------------------------------


@triton.jit
def _add_s_f32(
    x_ptr,
    inner_len,
    inner_stride,
    s,
    OSHAPE: tl.constexpr,
    OSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nb = tl.cdiv(inner_len, BLOCK)
    outer = pid // nb
    ib = pid - outer * nb
    o = outer
    off = 0
    for d in tl.static_range(len(OSHAPE)):
        idx = o % OSHAPE[d]
        o = o // OSHAPE[d]
        off += idx * OSTRIDE[d]
    inner = ib * BLOCK + tl.arange(0, BLOCK)
    mask = inner < inner_len
    ptrs = x_ptr + off + inner * inner_stride
    v = tl.load(ptrs, mask=mask)
    tl.store(ptrs, v + s, mask=mask)


@triton.jit
def _add_s_i64(
    x_ptr,
    inner_len,
    inner_stride,
    s,
    OSHAPE: tl.constexpr,
    OSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nb = tl.cdiv(inner_len, BLOCK)
    outer = pid // nb
    ib = pid - outer * nb
    o = outer
    off = 0
    for d in tl.static_range(len(OSHAPE)):
        idx = o % OSHAPE[d]
        o = o // OSHAPE[d]
        off += idx * OSTRIDE[d]
    inner = ib * BLOCK + tl.arange(0, BLOCK)
    mask = inner < inner_len
    ptrs = x_ptr + off + inner * inner_stride
    v = tl.load(ptrs, mask=mask)
    tl.store(ptrs, v + s, mask=mask)


@triton.jit
def _add_s_f64(
    x_ptr,
    inner_len,
    inner_stride,
    sbits,
    OSHAPE: tl.constexpr,
    OSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nb = tl.cdiv(inner_len, BLOCK)
    outer = pid // nb
    ib = pid - outer * nb
    o = outer
    off = 0
    for d in tl.static_range(len(OSHAPE)):
        idx = o % OSHAPE[d]
        o = o // OSHAPE[d]
        off += idx * OSTRIDE[d]
    s = sbits.to(tl.float64, bitcast=True)
    inner = ib * BLOCK + tl.arange(0, BLOCK)
    mask = inner < inner_len
    ptrs = x_ptr + off + inner * inner_stride
    v = tl.load(ptrs, mask=mask)
    tl.store(ptrs, v + s, mask=mask)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_F32_FAMILY = {torch.float16: 0, torch.bfloat16: 0, torch.float32: 0}
_F64_FAMILY = {torch.float64: 2}
_INT_FAMILY = {
    torch.int8: 1,
    torch.int16: 1,
    torch.int32: 1,
    torch.int64: 1,
    torch.uint8: 1,
    torch.bool: 1,
}

_FAMILY = {}
_FAMILY.update(_F32_FAMILY)
_FAMILY.update(_F64_FAMILY)
_FAMILY.update(_INT_FAMILY)

_BLOCK = {torch.float16: 2048, torch.bfloat16: 2048, torch.float32: 1024}


def _block_for(dtype):
    b = _BLOCK.get(dtype)
    if b is not None:
        return b
    return 1024 if dtype.itemsize >= 4 else 2048


def _double_bits(value):
    return struct.unpack("<q", struct.pack("<d", value))[0]


def _strided_plan(t):
    """Collapse a tensor view into (inner_len, inner_stride, oshape, ostride)."""
    dims = []
    for sz, st in zip(t.shape, t.stride()):
        if sz > 1:
            if st == 0:
                raise RuntimeError(
                    "unsupported operation: more than one element of the written-to "
                    "tensor refers to a single memory location"
                )
            dims.append((sz, st))
    if not dims:
        return None
    dims.sort(key=lambda p: abs(p[1]))
    inner_len, inner_stride = dims[0]
    rest = dims[1:]
    if inner_stride == 1:
        while rest and rest[0][1] == inner_len:
            inner_len *= rest[0][0]
            rest.pop(0)
    oshape = tuple(p[0] for p in rest)
    ostride = tuple(p[1] for p in rest)
    return inner_len, inner_stride, oshape, ostride


_FAST = {}
_HOOKS = None


def _hooks():
    global _HOOKS
    if _HOOKS is None:
        from triton import knobs

        _HOOKS = (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    return _HOOKS


def _build_fast(ck):
    """Collect the launch pieces of an already compiled kernel.

    Re-entering Triton's JIT binder on every call (arg binding, specialization
    key construction, cache lookup, launch-metadata build) is the dominant cost
    for small tensors on this backend.  The compiled kernel handle can be
    launched directly with the same argument layout the binder uses.
    """
    ent, ext = _hooks()
    ps = getattr(ck.metadata, "profile_scratch_size", 0) or 0
    scratch = torch.empty(ps, dtype=torch.int8, device="cuda") if ps else 0
    return (ck.run, ck.function, ck.packed_metadata, scratch, ent, ext)


def _fast_call(ent, grid0, stream, *args):
    launcher, fn, pm, scratch, enth, exth = ent
    launcher(grid0, 1, 1, stream, fn, pm, None, enth, exth, *args, scratch)


def _spec(v):
    # Triton specialization class of an integer argument: 1 -> value is 1,
    # 2 -> divisible by 16, 0 -> generic.  The cache key must never merge two
    # distinct classes, otherwise a specialized kernel could be reused for a
    # value it was not compiled for.
    if v == 1:
        return 1
    if v % 16 == 0:
        return 2
    return 0


try:
    _RAW_STREAM = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - fall back to the public API
    _RAW_STREAM = None

_current_device = torch.cuda.current_device


def _cur():
    dev = torch.cuda.current_device()
    if _RAW_STREAM is not None:
        return dev, _RAW_STREAM(dev)
    return dev, torch.cuda.current_stream(dev).cuda_stream


def _launch_one(t, sval, fam):
    n = t.numel()
    block = _block_for(t.dtype)
    grid0 = (n + block - 1) // block
    if fam == 1:
        a = int(sval)
    elif fam == 2:
        a = _double_bits(sval)
    else:
        a = sval
    dev, stream = _cur()
    key = (
        dev,
        1,
        fam,
        t.dtype,
        block,
        _spec(n),
        t.data_ptr() & 15 == 0,
        n < 2147483648,
        _spec(a) if fam == 1 else 0,
    )
    fast = _FAST.get(key)
    if fast is None:
        grid = (grid0,)
        if fam == 0:
            ck = _add_c_f32[grid](t, n, a, BLOCK=block, num_warps=4)
        elif fam == 1:
            ck = _add_c_i64[grid](t, n, a, BLOCK=block, num_warps=4)
        else:
            ck = _add_c_f64[grid](t, n, a, BLOCK=block, num_warps=4)
        _FAST[key] = _build_fast(ck)
    else:
        _fast_call(fast, grid0, stream, t, n, a)


def _launch_strided(t, sval, fam, plan):
    inner_len, inner_stride, oshape, ostride = plan
    outer_count = 1
    for sz in oshape:
        outer_count *= sz
    block = min(_block_for(t.dtype), triton.next_power_of_2(inner_len))
    grid0 = outer_count * ((inner_len + block - 1) // block)
    if fam == 1:
        a = int(sval)
    elif fam == 2:
        a = _double_bits(sval)
    else:
        a = sval
    dev, stream = _cur()
    key = (
        dev,
        2,
        fam,
        t.dtype,
        block,
        oshape,
        ostride,
        _spec(inner_len),
        _spec(inner_stride),
        t.data_ptr() & 15 == 0,
        _spec(a) if fam == 1 else 0,
    )
    fast = _FAST.get(key)
    if fast is None:
        grid = (grid0,)
        if fam == 0:
            ck = _add_s_f32[grid](
                t,
                inner_len,
                inner_stride,
                a,
                OSHAPE=oshape,
                OSTRIDE=ostride,
                BLOCK=block,
                num_warps=4,
            )
        elif fam == 1:
            ck = _add_s_i64[grid](
                t,
                inner_len,
                inner_stride,
                a,
                OSHAPE=oshape,
                OSTRIDE=ostride,
                BLOCK=block,
                num_warps=4,
            )
        else:
            ck = _add_s_f64[grid](
                t,
                inner_len,
                inner_stride,
                a,
                OSHAPE=oshape,
                OSTRIDE=ostride,
                BLOCK=block,
                num_warps=4,
            )
        _FAST[key] = _build_fast(ck)
    else:
        _fast_call(fast, grid0, stream, t, inner_len, inner_stride, a)


def _launch_fused4(items, fam):
    # items: four (tensor, scalar, element_count) triples; count 0 pads a slot.
    t0, v0, n0 = items[0]
    t1, v1, n1 = items[1]
    t2, v2, n2 = items[2]
    t3, v3, n3 = items[3]
    if fam == 1:
        a0, a1, a2, a3 = int(v0), int(v1), int(v2), int(v3)
    elif fam == 2:
        a0, a1, a2, a3 = (
            _double_bits(v0),
            _double_bits(v1),
            _double_bits(v2),
            _double_bits(v3),
        )
    else:
        a0, a1, a2, a3 = float(v0), float(v1), float(v2), float(v3)
    _launch4(t0, t1, t2, t3, n0, n1, n2, n3, a0, a1, a2, a3, fam)


def _launch4(t0, t1, t2, t3, n0, n1, n2, n3, a0, a1, a2, a3, fam):
    block = _block_for(t0.dtype)
    dev, stream = _cur()
    if fam == 0 and n0 == n1 == n2 == n3:
        # Uniform element count: one count argument instead of four.
        nb = 4 * ((n0 + block - 1) // block)
        key = (
            dev,
            4,
            t0.dtype,
            block,
            _spec(n0),
            n0 < 2147483648,
            t0.data_ptr() & 15 == 0,
            t1.data_ptr() & 15 == 0,
            t2.data_ptr() & 15 == 0,
            t3.data_ptr() & 15 == 0,
        )
        ent = _FAST.get(key)
        if ent is None:
            grid = (nb,)
            ck = _fadd4u_f32[grid](
                t0,
                t1,
                t2,
                t3,
                n0,
                a0,
                a1,
                a2,
                a3,
                BLOCK=block,
                num_warps=4,
            )
            _FAST[key] = _build_fast(ck)
        else:
            launcher, fn, pm, scratch, enth, exth = ent
            launcher(
                nb,
                1,
                1,
                stream,
                fn,
                pm,
                None,
                enth,
                exth,
                t0,
                t1,
                t2,
                t3,
                n0,
                a0,
                a1,
                a2,
                a3,
                scratch,
            )
        return
    nb = (
        (n0 + block - 1) // block
        + (n1 + block - 1) // block
        + (n2 + block - 1) // block
        + (n3 + block - 1) // block
    )
    key = (
        dev,
        3,
        fam,
        t0.dtype,
        block,
        _spec(n0),
        _spec(n1),
        _spec(n2),
        _spec(n3),
        t0.data_ptr() & 15 == 0,
        t1.data_ptr() & 15 == 0,
        t2.data_ptr() & 15 == 0,
        t3.data_ptr() & 15 == 0,
        n0 < 2147483648 and n1 < 2147483648 and n2 < 2147483648 and n3 < 2147483648,
        (_spec(a0), _spec(a1), _spec(a2), _spec(a3)) if fam == 1 else (),
    )
    fast = _FAST.get(key)
    if fast is None:
        grid = (nb,)
        if fam == 0:
            ck = _fadd4_f32[grid](
                t0,
                t1,
                t2,
                t3,
                n0,
                n1,
                n2,
                n3,
                a0,
                a1,
                a2,
                a3,
                BLOCK=block,
                num_warps=4,
            )
        elif fam == 1:
            ck = _fadd4_i64[grid](
                t0,
                t1,
                t2,
                t3,
                n0,
                n1,
                n2,
                n3,
                a0,
                a1,
                a2,
                a3,
                BLOCK=block,
                num_warps=4,
            )
        else:
            ck = _fadd4_f64[grid](
                t0,
                t1,
                t2,
                t3,
                n0,
                n1,
                n2,
                n3,
                a0,
                a1,
                a2,
                a3,
                BLOCK=block,
                num_warps=4,
            )
        _FAST[key] = _build_fast(ck)
    else:
        _fast_call(fast, nb, stream, t0, t1, t2, t3, n0, n1, n2, n3, a0, a1, a2, a3)


def _foreach_add__ScalarList(self, scalars):
    logger.debug("GEMS_HYGON _FOREACH_ADD__SCALARLIST")
    if isinstance(self, torch.Tensor):
        tensors = (self,)
    else:
        tensors = self

    if isinstance(scalars, torch.Tensor) and scalars.dim() > 0:
        scalars = scalars.tolist()

    per_tensor = isinstance(scalars, (list, tuple))
    nt = len(tensors)
    if per_tensor and len(scalars) != nt:
        raise RuntimeError(
            "Tensor list and scalar list must have the same length, "
            f"but got {nt} tensors and {len(scalars)} scalars"
        )

    # Straight-line hot path: four contiguous tensors of one dtype (the common
    # foreach shape).  Avoids the grouping machinery entirely.
    if per_tensor and nt == 4:
        t0 = tensors[0]
        t1 = tensors[1]
        t2 = tensors[2]
        t3 = tensors[3]
        dt = t0.dtype
        if (
            dt == t1.dtype
            and dt == t2.dtype
            and dt == t3.dtype
            and t0.is_contiguous()
            and t1.is_contiguous()
            and t2.is_contiguous()
            and t3.is_contiguous()
        ):
            fam = _FAMILY.get(dt)
            n0 = t0.numel()
            n1 = t1.numel()
            n2 = t2.numel()
            n3 = t3.numel()
            if fam is not None and n0 and n1 and n2 and n3:
                v0 = scalars[0]
                v1 = scalars[1]
                v2 = scalars[2]
                v3 = scalars[3]
                if fam == 1:
                    a0 = int(v0)
                    a1 = int(v1)
                    a2 = int(v2)
                    a3 = int(v3)
                    if a0 != v0 or a1 != v1 or a2 != v2 or a3 != v3:
                        raise RuntimeError(
                            "result type Float can't be cast to the desired "
                            "output type"
                        )
                elif fam == 2:
                    a0 = _double_bits(v0)
                    a1 = _double_bits(v1)
                    a2 = _double_bits(v2)
                    a3 = _double_bits(v3)
                else:
                    a0 = float(v0)
                    a1 = float(v1)
                    a2 = float(v2)
                    a3 = float(v3)
                if fam == 0 and n0 == n1 == n2 == n3:
                    # Inlined uniform-count launch: no helper frames, no _spec
                    # calls and the same cache key the generic path uses.
                    block = _BLOCK[dt]
                    nb = 4 * ((n0 + block - 1) // block)
                    dev = _current_device()
                    nspec = 1 if n0 == 1 else (2 if n0 % 16 == 0 else 0)
                    key = (
                        dev,
                        4,
                        dt,
                        block,
                        nspec,
                        n0 < 2147483648,
                        t0.data_ptr() & 15 == 0,
                        t1.data_ptr() & 15 == 0,
                        t2.data_ptr() & 15 == 0,
                        t3.data_ptr() & 15 == 0,
                    )
                    ent = _FAST.get(key)
                    if ent is None:
                        grid = (nb,)
                        ck = _fadd4u_f32[grid](
                            t0,
                            t1,
                            t2,
                            t3,
                            n0,
                            a0,
                            a1,
                            a2,
                            a3,
                            BLOCK=block,
                            num_warps=4,
                        )
                        _FAST[key] = _build_fast(ck)
                    else:
                        launcher, fn, pm, scratch, enth, exth = ent
                        if _RAW_STREAM is not None:
                            stream = _RAW_STREAM(dev)
                        else:
                            stream = torch.cuda.current_stream(dev).cuda_stream
                        launcher(
                            nb,
                            1,
                            1,
                            stream,
                            fn,
                            pm,
                            None,
                            enth,
                            exth,
                            t0,
                            t1,
                            t2,
                            t3,
                            n0,
                            a0,
                            a1,
                            a2,
                            a3,
                            scratch,
                        )
                else:
                    _launch4(t0, t1, t2, t3, n0, n1, n2, n3, a0, a1, a2, a3, fam)
                return None

    groups = {}
    strided = []
    k = 0
    for t in tensors:
        raw = scalars[k] if per_tensor else scalars
        k += 1
        sv = float(raw)
        n = t.numel()
        if n == 0:
            continue
        dt = t.dtype
        fam = _FAMILY.get(dt)
        if fam is None:
            raise RuntimeError(f"unsupported dtype {dt}")
        if fam == 1 and sv != int(sv):
            raise RuntimeError(
                "result type Float can't be cast to the desired output type"
            )
        if t.is_contiguous():
            g = groups.get(dt)
            if g is None:
                groups[dt] = [(t, sv, n)]
            else:
                g.append((t, sv, n))
        else:
            strided.append((t, sv, fam, _strided_plan(t)))

    for t, sv, fam, plan in strided:
        if plan is None:
            _launch_one(t, sv, fam)
        else:
            _launch_strided(t, sv, fam, plan)

    for dt, items in groups.items():
        fam = _FAMILY[dt]
        L = len(items)
        if L == 1:
            _launch_one(items[0][0], items[0][1], fam)
        elif L == 2:
            pad = items[-1]
            _launch_fused4(items + [(pad[0], 0.0, 0), (pad[0], 0.0, 0)], fam)
        elif L == 3:
            pad = items[-1]
            _launch_fused4(items + [(pad[0], 0.0, 0)], fam)
        elif L == 4:
            _launch_fused4(items, fam)
        else:
            i = 0
            while i < L:
                chunk = items[i : i + 4]
                i += 4
                c = len(chunk)
                if c == 1:
                    _launch_one(chunk[0][0], chunk[0][1], fam)
                elif c == 2:
                    pad = chunk[-1]
                    _launch_fused4(chunk + [(pad[0], 0.0, 0), (pad[0], 0.0, 0)], fam)
                elif c == 3:
                    pad = chunk[-1]
                    _launch_fused4(chunk + [(pad[0], 0.0, 0)], fam)
                else:
                    _launch_fused4(chunk, fam)
    return None
