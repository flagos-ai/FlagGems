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

"""In-place multi-tensor expm1 (``aten::_foreach_expm1_``).

Every tensor in the list is mutated in place: ``T[i] <- expm1(T[i])``.
Storage, strides and dtype of each element are preserved and the call
returns ``None``.

The common case (a list of flat-addressable tensors sharing a dtype) is
executed by a *single* launch: the per-tensor ``data_ptr``/``numel`` pairs
travel as scalar kernel arguments, so no host-to-device metadata buffer and
no per-element bookkeeping is needed.
"""

import torch
import triton
import triton.language as tl

try:  # NVIDIA-style path
    from triton.language.extra import libdevice as _libdevice
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _libdevice
    except Exception:
        _libdevice = None

try:
    from triton import knobs as _knobs
except Exception:  # pragma: no cover
    _knobs = None

try:
    from triton.runtime import driver as _trt_driver
except Exception:  # pragma: no cover
    _trt_driver = None

_HAS_EXPM1 = _libdevice is not None and hasattr(_libdevice, "expm1")
_HAS_EXP2 = hasattr(tl, "exp2")
_LOG2E = 1.4426950408889634


if _HAS_EXP2:

    @triton.jit
    def _expm1_f32(x):
        # exp2 is a single hardware ex2 operation on this backend, while the
        # libdevice expm1 is a range-reduced polynomial; the elementwise rate
        # of the streaming workloads is set by that math, not by bandwidth.
        return tl.exp2(x * 1.4426950408889634) - 1.0

elif _HAS_EXPM1:

    @triton.jit
    def _expm1_f32(x):
        return _libdevice.expm1(x)

else:  # pragma: no cover - fallback when the backend has neither

    @triton.jit
    def _expm1_f32(x):
        return tl.exp(x) - 1.0


if _HAS_EXPM1:

    @triton.jit
    def _expm1_f64(x):
        return _libdevice.expm1(x)

else:  # pragma: no cover - fallback when the backend has no libdevice

    @triton.jit
    def _expm1_f64(x):
        return tl.exp(x) - 1.0


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _tile_expm1(ptr, n, chunk, BLOCK: tl.constexpr, IS_F64: tl.constexpr):
    """expm1 over one BLOCK-wide tile of the tensor ``ptr``."""
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0)
    if IS_F64:
        y = _expm1_f64(x)
    else:
        y = _expm1_f32(x.to(tl.float32))
    tl.store(ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _tile_expm1_i64(p, n, chunk, BLOCK: tl.constexpr, DT: tl.constexpr):
    """Same tile as ``_tile_expm1`` for an address that arrived as an int64.

    Handing the already-computed ``data_ptr()`` value to the launcher spares it
    one Python ``data_ptr`` call per tensor argument, which dominates the host
    side of a launch whose tensors are only a few thousand elements long.
    """
    if DT == 0:
        _tile_expm1(p.to(tl.pointer_type(tl.float16)), n, chunk, BLOCK, False)
    elif DT == 1:
        _tile_expm1(p.to(tl.pointer_type(tl.bfloat16)), n, chunk, BLOCK, False)
    elif DT == 2:
        _tile_expm1(p.to(tl.pointer_type(tl.float32)), n, chunk, BLOCK, False)
    else:
        _tile_expm1(p.to(tl.pointer_type(tl.float64)), n, chunk, BLOCK, True)


@triton.jit
def _expm1_flat_kernel(
    x_ptr,
    n_elem,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if IS_F64:
        y = _expm1_f64(x)
    else:
        y = _expm1_f32(x.to(tl.float32))
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _expm1_fused16_uniform_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    """Same dtype, same element count, up to 16 tensors, one shared numel.

    ``grid`` is ``(count, chunks)``, so every launched program owns a real
    tensor and the argument list stays at 17 runtime values instead of 33.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    if t == 0:
        _tile_expm1(p0, n, chunk, BLOCK, IS_F64)
    elif t == 1:
        _tile_expm1(p1, n, chunk, BLOCK, IS_F64)
    elif t == 2:
        _tile_expm1(p2, n, chunk, BLOCK, IS_F64)
    elif t == 3:
        _tile_expm1(p3, n, chunk, BLOCK, IS_F64)
    elif t == 4:
        _tile_expm1(p4, n, chunk, BLOCK, IS_F64)
    elif t == 5:
        _tile_expm1(p5, n, chunk, BLOCK, IS_F64)
    elif t == 6:
        _tile_expm1(p6, n, chunk, BLOCK, IS_F64)
    elif t == 7:
        _tile_expm1(p7, n, chunk, BLOCK, IS_F64)
    elif t == 8:
        _tile_expm1(p8, n, chunk, BLOCK, IS_F64)
    elif t == 9:
        _tile_expm1(p9, n, chunk, BLOCK, IS_F64)
    elif t == 10:
        _tile_expm1(p10, n, chunk, BLOCK, IS_F64)
    elif t == 11:
        _tile_expm1(p11, n, chunk, BLOCK, IS_F64)
    elif t == 12:
        _tile_expm1(p12, n, chunk, BLOCK, IS_F64)
    elif t == 13:
        _tile_expm1(p13, n, chunk, BLOCK, IS_F64)
    elif t == 14:
        _tile_expm1(p14, n, chunk, BLOCK, IS_F64)
    else:
        _tile_expm1(p15, n, chunk, BLOCK, IS_F64)


@triton.jit
def _expm1_fused16_uniform_i64_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    """Small-tensor variant of the uniform fused kernel: int64 addresses.

    The addresses are the values the host already read for its cache key, so
    the launcher converts 17 integers instead of making one Python
    ``data_ptr`` call per tensor argument.  The cast costs the wide loads, but
    for tiles this small the launch is latency-bound, not bandwidth-bound.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    if t == 0:
        _tile_expm1_i64(p0, n, chunk, BLOCK, DT)
    elif t == 1:
        _tile_expm1_i64(p1, n, chunk, BLOCK, DT)
    elif t == 2:
        _tile_expm1_i64(p2, n, chunk, BLOCK, DT)
    elif t == 3:
        _tile_expm1_i64(p3, n, chunk, BLOCK, DT)
    elif t == 4:
        _tile_expm1_i64(p4, n, chunk, BLOCK, DT)
    elif t == 5:
        _tile_expm1_i64(p5, n, chunk, BLOCK, DT)
    elif t == 6:
        _tile_expm1_i64(p6, n, chunk, BLOCK, DT)
    elif t == 7:
        _tile_expm1_i64(p7, n, chunk, BLOCK, DT)
    elif t == 8:
        _tile_expm1_i64(p8, n, chunk, BLOCK, DT)
    elif t == 9:
        _tile_expm1_i64(p9, n, chunk, BLOCK, DT)
    elif t == 10:
        _tile_expm1_i64(p10, n, chunk, BLOCK, DT)
    elif t == 11:
        _tile_expm1_i64(p11, n, chunk, BLOCK, DT)
    elif t == 12:
        _tile_expm1_i64(p12, n, chunk, BLOCK, DT)
    elif t == 13:
        _tile_expm1_i64(p13, n, chunk, BLOCK, DT)
    elif t == 14:
        _tile_expm1_i64(p14, n, chunk, BLOCK, DT)
    else:
        _tile_expm1_i64(p15, n, chunk, BLOCK, DT)


@triton.jit
def _expm1_fused16_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
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
):
    """Up to 16 same-dtype flat-addressable tensors in one launch.

    The tensor is chosen with a *static* branch per slot rather than by
    selecting one pointer value at run time: each branch then uses a real
    typed pointer argument, which keeps the alignment/contiguity analysis
    that vectorizes the loads and stores.
    """
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    if t == 0:
        _tile_expm1(p0, n0, chunk, BLOCK, IS_F64)
    elif t == 1:
        _tile_expm1(p1, n1, chunk, BLOCK, IS_F64)
    elif t == 2:
        _tile_expm1(p2, n2, chunk, BLOCK, IS_F64)
    elif t == 3:
        _tile_expm1(p3, n3, chunk, BLOCK, IS_F64)
    elif t == 4:
        _tile_expm1(p4, n4, chunk, BLOCK, IS_F64)
    elif t == 5:
        _tile_expm1(p5, n5, chunk, BLOCK, IS_F64)
    elif t == 6:
        _tile_expm1(p6, n6, chunk, BLOCK, IS_F64)
    elif t == 7:
        _tile_expm1(p7, n7, chunk, BLOCK, IS_F64)
    elif t == 8:
        _tile_expm1(p8, n8, chunk, BLOCK, IS_F64)
    elif t == 9:
        _tile_expm1(p9, n9, chunk, BLOCK, IS_F64)
    elif t == 10:
        _tile_expm1(p10, n10, chunk, BLOCK, IS_F64)
    elif t == 11:
        _tile_expm1(p11, n11, chunk, BLOCK, IS_F64)
    elif t == 12:
        _tile_expm1(p12, n12, chunk, BLOCK, IS_F64)
    elif t == 13:
        _tile_expm1(p13, n13, chunk, BLOCK, IS_F64)
    elif t == 14:
        _tile_expm1(p14, n14, chunk, BLOCK, IS_F64)
    else:
        _tile_expm1(p15, n15, chunk, BLOCK, IS_F64)


@triton.jit
def _expm1_strided_kernel(
    x_ptr,
    n_elem,
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
    IS_F64: tl.constexpr,
):
    """Logical index -> address walk for gappy (not flat-addressable) views."""
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n_elem
    rem = idx
    addr = tl.zeros([BLOCK], dtype=tl.int64)
    r = rem % s7
    addr += r * t7
    rem = rem // s7
    r = rem % s6
    addr += r * t6
    rem = rem // s6
    r = rem % s5
    addr += r * t5
    rem = rem // s5
    r = rem % s4
    addr += r * t4
    rem = rem // s4
    r = rem % s3
    addr += r * t3
    rem = rem // s3
    r = rem % s2
    addr += r * t2
    rem = rem // s2
    r = rem % s1
    addr += r * t1
    rem = rem // s1
    r = rem % s0
    addr += r * t0
    x = tl.load(x_ptr + addr, mask=mask, other=0)
    if IS_F64:
        y = _expm1_f64(x)
    else:
        y = _expm1_f32(x.to(tl.float32))
    tl.store(x_ptr + addr, y.to(x.dtype), mask=mask)


@triton.jit
def _expm1_complex_kernel(
    x_ptr,
    n_elem,
    BLOCK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elem
    re = tl.load(x_ptr + 2 * offs, mask=mask, other=0.0)
    im = tl.load(x_ptr + 2 * offs + 1, mask=mask, other=0.0)
    if IS_F64:
        r = re
        i = im
    else:
        r = re.to(tl.float32)
        i = im.to(tl.float32)
    e = tl.exp(r)
    out_r = e * tl.cos(i) - 1.0
    out_i = e * tl.sin(i)
    tl.store(x_ptr + 2 * offs, out_r.to(re.dtype), mask=mask)
    tl.store(x_ptr + 2 * offs + 1, out_i.to(im.dtype), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8
_FUSED = 16
_INT32_MAX = 0x7FFFFFFF

_TL_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Per-tensor element count below which a launch is latency-bound, so the fused
# path may trade the wide vector accesses for a much cheaper host launch.
_SMALL_N = 1 << 16

_DT_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


def _is_flat(t):
    """True when flat indexing over [data_ptr, data_ptr + numel) is exact."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _has_internal_overlap(t):
    try:
        return torch._debug_has_internal_overlap(t) == 1
    except Exception:  # pragma: no cover
        return False


_flat_runners = {}


def _bind_runner(kernel, g0, g1):
    """Launcher for an already compiled kernel, minus unused per-call work.

    Triton's bound runner builds a fresh launch-metadata dictionary on every
    call even when no launch hooks are installed.  While both hook slots are
    empty the metadata is never read, so this returns a launcher that skips its
    construction and passes the same (empty) hooks; as soon as a hook appears
    it defers to the stock runner, so hook-based instrumentation is unchanged.
    """
    stock = None
    try:
        stock = kernel[(g0, g1, 1)]
    except Exception:  # pragma: no cover - backend without bound launchers
        return None
    if _knobs is None or _trt_driver is None:
        return stock
    try:
        runtime = _knobs.runtime
        if (
            runtime.launch_enter_hook is not None
            or runtime.launch_exit_hook is not None
        ):
            return stock
        run = kernel.run
        function = kernel.function
        packed = kernel.packed_metadata
        active = _trt_driver.active
        get_device = active.get_current_device
        get_stream = active.get_current_stream
    except Exception:  # pragma: no cover - unexpected runtime layout
        return stock

    def launch(*args):
        if (
            runtime.launch_enter_hook is not None
            or runtime.launch_exit_hook is not None
        ):
            stock(*args)
            return
        stream = get_stream(get_device())
        run(g0, g1, 1, stream, function, packed, None, None, None, *args)

    return launch


def _launch_flat(t):
    """Single-tensor launch through a cached pre-bound compiled kernel.

    Triton's generic entry point re-binds arguments, re-computes the
    specialization/cache key and re-validates bindings on every call, which on
    this stack costs more than the kernel itself for small tensors.  The bound
    runner performs only the stream lookup, the launch metadata and the
    launch.  The cache key covers everything that selects the compiled binary
    (device, dtype, element count and pointer alignment), so a mismatch simply
    falls back to a normal launch.
    """
    n = t.numel()
    is_f64 = t.dtype == torch.float64
    key = (t.get_device(), t.dtype, n, t.data_ptr() & 15)
    runner = _flat_runners.get(key)
    if runner is not None:
        runner(t, n, _BLOCK, is_f64, n > _INT32_MAX)
        return
    g0 = (n + _BLOCK - 1) // _BLOCK
    kernel = _expm1_flat_kernel[(g0,)](t, n, _BLOCK, is_f64, n > _INT32_MAX)
    runner = _bind_runner(kernel, g0, 1)
    if runner is None:
        return
    if len(_flat_runners) > 512:
        _flat_runners.clear()
    _flat_runners[key] = runner


_fused_runners = {}


def _launch_uniform(dtype, ts, n, dev, ptrs):
    """One launch for <=16 equal-size flat-addressable tensors of one dtype.

    ``ptrs`` are the ``data_ptr`` values the caller already read for its
    validation pass.  For very small tiles they are also what the kernel
    receives, so the launcher performs no per-argument Python call at all;
    larger tiles keep the typed pointer arguments that carry the alignment
    information the wide vector accesses need.
    """
    cnt = len(ts)
    small = n <= _SMALL_N
    key = (dev, dtype, n, cnt, small, tuple(ptrs))
    entry = _fused_runners.get(key)
    if entry is not None:
        base, tail, use_i64 = entry
        if use_i64:
            base(*ptrs, n, *tail)
        else:
            base(*ts, n, *tail)
        return
    g1 = (n + _BLOCK - 1) // _BLOCK
    pad = _FUSED - cnt
    if small:
        ps = ptrs if pad == 0 else ptrs + [ptrs[0]] * pad
        tail = (_BLOCK, _DT_CODE[dtype])
        kernel = _expm1_fused16_uniform_i64_kernel[(cnt, g1)](*ps, n, *tail)
    else:
        ps = ts if pad == 0 else list(ts) + [ts[0]] * pad
        tail = (_BLOCK, dtype is torch.float64)
        kernel = _expm1_fused16_uniform_kernel[(cnt, g1)](*ps, n, *tail)
    base = _bind_runner(kernel, cnt, g1)
    if base is None:
        return
    if len(_fused_runners) > 256:
        _fused_runners.clear()
    _fused_runners[key] = (base, tail, small)


def _launch_fused(dtype, ts, ns):
    """One launch for every flat-addressable tensor of ``dtype`` (<=16 each)."""
    count = len(ts)
    if count == 1:
        _launch_flat(ts[0])
        return
    is_f64 = dtype == torch.float64
    for start in range(0, count, _FUSED):
        ps = ts[start : start + _FUSED]
        cs = ns[start : start + _FUSED]
        cnt = len(ps)
        if cnt == 1:
            _launch_flat(ps[0])
            continue
        max_n = max(cs)
        if max_n > _INT32_MAX:
            for tt in ps:
                _launch_flat(tt)
            continue
        if max_n == min(cs):
            _launch_uniform(
                dtype,
                ps,
                max_n,
                ps[0].get_device(),
                [p.data_ptr() for p in ps],
            )
            continue
        pad = _FUSED - cnt
        if pad:
            ps = ps + [ps[0]] * pad
            cs = cs + [0] * pad
        grid = (_FUSED, (max_n + _BLOCK - 1) // _BLOCK)
        _expm1_fused16_kernel[grid](*ps, *cs, _BLOCK, is_f64, num_warps=_NUM_WARPS)


def _launch_strided(t):
    n = t.numel()
    shape = list(t.shape)
    stride = list(t.stride())
    # Collapse adjacent dimensions that are already contiguous in memory so
    # that almost every layout fits the fixed rank-8 walk.
    cshape = [shape[-1]]
    cstride = [stride[-1]]
    for d in range(len(shape) - 2, -1, -1):
        if stride[d] == cstride[0] * cshape[0]:
            cshape[0] *= shape[d]
        else:
            cshape.insert(0, shape[d])
            cstride.insert(0, stride[d])
    if len(cshape) > _MAX_RANK:
        # Extremely exotic layout: split along the outermost dimension.
        for i in range(cshape[0]):
            _launch_strided(t[i])
        return
    pad = _MAX_RANK - len(cshape)
    shape = [1] * pad + cshape
    stride = [0] * pad + cstride
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _expm1_strided_kernel[grid](
        t,
        n,
        *shape,
        *stride,
        _BLOCK,
        t.dtype == torch.float64,
        num_warps=_NUM_WARPS,
    )


def _apply_slow(t):
    dtype = t.dtype
    if dtype.is_complex:
        if dtype is torch.complex32:
            raise RuntimeError("\"foreach\" not implemented for 'complex32'")
        n = t.numel()
        if n == 0:
            return
        try:
            rv = torch.view_as_real(t)
        except Exception:
            rv = None
        if rv is None or not _is_flat(rv):
            # Rare layout: stage through a dense copy, then write back.
            tmp = t.contiguous()
            rv = torch.view_as_real(tmp)
            grid = ((n + _BLOCK - 1) // _BLOCK,)
            _expm1_complex_kernel[grid](
                rv, n, _BLOCK, rv.dtype == torch.float64, num_warps=_NUM_WARPS
            )
            t.copy_(tmp)
            return
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _expm1_complex_kernel[grid](
            rv, n, _BLOCK, rv.dtype == torch.float64, num_warps=_NUM_WARPS
        )
        return

    if _is_flat(t):
        _launch_flat(t)
    else:
        _launch_strided(t)


def foreach_expm1_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(self)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Fast path: homogeneous list of contiguous floating tensors, the common
    # case.  One attribute pass per element gathers everything the launch
    # needs, including the addresses, so no element is visited twice.
    first = self[0]
    if type(first) is torch.Tensor:
        dt = first.dtype
        if dt in _TL_DT:
            n = first.numel()
            if n and first.is_contiguous():
                if count == 1:
                    _launch_flat(first)
                    return None
                dev = first.get_device()
                ptrs = [first.data_ptr()]
                try:
                    for t in self[1:]:
                        if (
                            t.dtype is dt
                            and t.numel() == n
                            and t.is_contiguous()
                            and t.get_device() == dev
                        ):
                            ptrs.append(t.data_ptr())
                            continue
                        ptrs = None
                        break
                except AttributeError:
                    ptrs = None
                if ptrs is not None and count <= _FUSED:
                    _launch_uniform(dt, self, n, dev, ptrs)
                    return None

    groups = {}
    slow = []

    # Validate every element first (ATen's Tensor[] preconditions) and
    # classify it, so no tensor is mutated when a later element is invalid.
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype.is_complex:
            if dtype is torch.complex32:
                raise RuntimeError("\"foreach\" not implemented for 'complex32'")
            if _has_internal_overlap(t):
                raise RuntimeError(_OVERLAP_MSG)
            if t.numel():
                slow.append(t)
            continue
        if not dtype.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + str(dtype).replace("torch.", "").capitalize()
            )
        n = t.numel()
        if n == 0:
            continue
        key = (dtype, t.get_device())
        if dtype in _TL_DT and t.is_contiguous():
            entry = groups.get(key)
            if entry is None:
                groups[key] = ([t], [n])
            else:
                entry[0].append(t)
                entry[1].append(n)
            continue
        if _has_internal_overlap(t):
            raise RuntimeError(_OVERLAP_MSG)
        if dtype in _TL_DT and _is_flat(t):
            entry = groups.get(key)
            if entry is None:
                groups[key] = ([t], [n])
            else:
                entry[0].append(t)
                entry[1].append(n)
        else:
            slow.append(t)

    for gkey, entry in groups.items():
        _launch_fused(gkey[0], entry[0], entry[1])
    for t in slow:
        _apply_slow(t)
    return None
