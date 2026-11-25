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

"""In-place multi-tensor sinh: ``aten::_foreach_sinh(Tensor(a!)[] self) -> ()``.

Design
------
* One Triton launch per ``(device, dtype, numel)`` group.  Each slot base is
  passed as a *pointer-typed* kernel argument (bucketed to 1/2/4/8/16 slots), so
  Triton keeps the 16-byte divisibility fact and emits 128-bit vector global
  accesses; an int64 ``data_ptr`` argument cast inside the kernel does not.
* A group has one uniform length, so the fused kernel needs a single ``n``.
  Multi-slot groups use a 4x larger BLOCK so the in-kernel slot-selection chain
  and other fixed per-program work amortize over more elements -- but only while
  a single tensor is small.  Above ``_SPLIT_BYTES`` per tensor the group is
  split into one launch per tensor, because the fused kernel then reaches only
  ~1.16 TB/s where the plain kernel reaches ~1.41 TB/s on the same shape.
* The single-tensor case (common) uses a dedicated 2-argument kernel.  Both it
  and the fused multi-slot kernel cache the compiled launcher after their first
  documented launch and reuse it, because Triton's per-call Python dispatch
  (specialization key, cache lookup, binding validation, launch metadata) costs
  more than the kernel itself on small tensors -- and grows with the number of
  bound arguments.
* A homogeneous list is dispatched in one pass with no grouping structures, so
  each element's metadata is touched once.
* When such a homogeneous list is also laid out back-to-back
  (``ptr[k+1] == ptr[k] + numel*itemsize``, which the allocator reproducibly
  produces for same-size small tensors), the whole list is one contiguous span
  and is served by ONE two-argument launch through the first tensor: no slot
  argument list, no in-kernel slot-selection chain, and the proven 8-element
  per-thread footprint of the single-tensor kernel.  This is a pure dispatch and
  addressing change -- round 5's merged *stride* kernel regressed because it
  passed an int64 ``data_ptr`` and lost 128-bit vectorization, which passing a
  real pointer here preserves.
* Grid is ``(slots, cdiv(numel, BLOCK))``; every program handles one tile of one
  tensor and exits immediately when its tile is out of range.
* Dense-but-permuted tensors (transpose/permute) are detected arithmetically and
  addressed flat -- zero-copy, fully coalesced, no staging.
* Genuinely strided views (step slices) fall back to a per-tensor collapsed
  stride kernel.
* ``fp16``/``bf16`` compute in ``fp32`` and round once at the store (ATen
  opmath).  The fp32 path uses an inline formula (``tl.exp`` lowers to
  ``llvm.exp.f32``) instead of libdevice ``sinh``, whose per-element external
  math symbol drags reciprocal/NaN/clamp expansions into the kernel; ``fp64``
  keeps libdevice and stays bit-exact.
* Validation mirrors ATen: an invalid dtype in the list raises after the
  preceding entries have already been mutated.
"""

import torch
import triton
import triton.language as tl

try:  # flagtree/metaX exposes the math library through this path
    from triton.language.extra import libdevice as _libdevice
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _libdevice
    except Exception:
        _libdevice = None


if _libdevice is not None:

    @triton.jit
    def _sinh_exact(x):
        return _libdevice.sinh(x)

else:  # pragma: no cover

    @triton.jit
    def _sinh_exact(x):
        e = tl.exp(x)
        return 0.5 * (e - 1.0 / e)


@triton.jit
def _sinh_fast(x):
    """fp32 sinh with no external math call.

    ``|x| < 0.5`` uses the odd Taylor series (no cancellation), otherwise the
    half-argument identity ``sinh(x) = 2*sinh(x/2)*cosh(x/2)``.  Halving is
    exact, so the only argument error is exp's own, and ``(0.5*s)*c`` keeps the
    intermediate from overflowing before the true result does.
    """
    ax = tl.abs(x)
    x2 = x * x
    series = x * (
        1.0
        + x2
        * (
            0.16666666666666666
            + x2
            * (
                0.008333333333333333
                + x2 * (0.0001984126984126984 + x2 * 2.7557319223985893e-06)
            )
        )
    )
    half = x * 0.5
    h = tl.exp(half)
    m = tl.exp(-half)
    s = h - m
    c = h + m
    return tl.where(ax < 0.5, series, (0.5 * s) * c)


@triton.jit
def _sinh_apply(x, DT: tl.constexpr):
    """Elementwise sinh in the accumulator dtype, rounded back to DT."""
    if DT == tl.float64:
        y = _sinh_exact(x)
    else:
        y = _sinh_fast(x.to(tl.float32)).to(DT)
    return y


_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# BLOCK is sized so one thread moves 16 bytes: 64 lanes per warp, 4 warps.
_BLOCK = {
    torch.float16: 2048,
    torch.bfloat16: 2048,
    torch.float32: 1024,
    torch.float64: 512,
}

# Multi-slot launches carry extra per-program work; give them more elements per
# thread so that fixed cost is amortized.
_MULTI_BLOCK_SCALE = 4

# Per-tensor size above which a multi-tensor group is launched one tensor at a
# time instead of fused: below this the fused launch amortizes its fixed
# per-program work, above it the plain kernel's higher bandwidth dominates.
_SPLIT_BYTES = 1 << 18

_ATEN_NAME = {
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex32: "ComplexHalf",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

_MAX_SLOTS = 16
_BUCKETS = (1, 2, 4, 8, 16)
_INT64_LIMIT = 2**31 - 1


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sinh_single(p, n, DT: tl.constexpr, BLOCK: tl.constexpr):
    start = tl.program_id(0) * BLOCK
    if start >= n:
        return
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(p + offs, mask=mask, other=0.0)
    tl.store(p + offs, _sinh_apply(x, DT), mask=mask)


@triton.jit
def _sinh_wide(p, n, DT: tl.constexpr, BLOCK: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BLOCK
    n_lim = n.to(tl.int64)
    if start >= n_lim:
        return
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_lim
    x = tl.load(p + offs, mask=mask, other=0.0)
    tl.store(p + offs, _sinh_apply(x, DT), mask=mask)


@triton.jit
def _sinh_fused(
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
    DT: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    ptr = p0
    if NT > 1:
        ptr = tl.where(slot == 1, p1, ptr)
    if NT > 2:
        ptr = tl.where(slot == 2, p2, ptr)
    if NT > 3:
        ptr = tl.where(slot == 3, p3, ptr)
    if NT > 4:
        ptr = tl.where(slot == 4, p4, ptr)
    if NT > 5:
        ptr = tl.where(slot == 5, p5, ptr)
    if NT > 6:
        ptr = tl.where(slot == 6, p6, ptr)
    if NT > 7:
        ptr = tl.where(slot == 7, p7, ptr)
    if NT > 8:
        ptr = tl.where(slot == 8, p8, ptr)
    if NT > 9:
        ptr = tl.where(slot == 9, p9, ptr)
    if NT > 10:
        ptr = tl.where(slot == 10, p10, ptr)
    if NT > 11:
        ptr = tl.where(slot == 11, p11, ptr)
    if NT > 12:
        ptr = tl.where(slot == 12, p12, ptr)
    if NT > 13:
        ptr = tl.where(slot == 13, p13, ptr)
    if NT > 14:
        ptr = tl.where(slot == 14, p14, ptr)
    if NT > 15:
        ptr = tl.where(slot == 15, p15, ptr)

    start = tl.program_id(1) * BLOCK
    if start >= n:
        return
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    tl.store(ptr + offs, _sinh_apply(x, DT), mask=mask)


@triton.jit
def _sinh_strided(
    addr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    DT: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n.to(tl.int64)

    off = tl.zeros([BLOCK], tl.int64)
    if RANK > 0:
        off += ((idx // d0) % s0) * q0
    if RANK > 1:
        off += ((idx // d1) % s1) * q1
    if RANK > 2:
        off += ((idx // d2) % s2) * q2
    if RANK > 3:
        off += ((idx // d3) % s3) * q3
    if RANK > 4:
        off += ((idx // d4) % s4) * q4
    if RANK > 5:
        off += ((idx // d5) % s5) * q5
    if RANK > 6:
        off += ((idx // d6) % s6) * q6
    if RANK > 7:
        off += ((idx // d7) % s7) * q7

    ptr = addr.to(tl.pointer_type(DT))
    x = tl.load(ptr + off, mask=mask, other=0.0)
    tl.store(ptr + off, _sinh_apply(x, DT), mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _pick_block(numel, dtype, multi):
    block = _BLOCK[dtype]
    if multi:
        block *= _MULTI_BLOCK_SCALE
    if numel >= block:
        return block
    picked = 32
    while picked < numel:
        picked *= 2
    return picked


def _bucket(count):
    for b in _BUCKETS:
        if count <= b:
            return b
    return None


def _is_dense_perm(t):
    """True when the element set is a dense permutation of ``[ptr, ptr+numel)``."""
    dims = [(s, st) for s, st in zip(t.shape, t.stride()) if s != 1]
    if not dims:
        return True
    dims = sorted(dims, key=lambda item: item[1])
    expected = 1
    for size, stride in dims:
        if stride != expected:
            return False
        expected *= size
    return expected == t.numel()


def _collapse(shape, stride):
    """Merge adjacent dims whose memory layout is contiguous.

    ``dims[d]`` (outer) and ``dims[d+1]`` merge into ``(size_d*size_{d+1},
    stride_{d+1})`` exactly when ``stride_d == stride_{d+1} * size_{d+1}``.
    """
    dims = [[s, st] for s, st in zip(shape, stride) if s != 1]
    out = []
    for size, st in dims:
        if out and out[-1][1] == st * size:
            out[-1][0] *= size
            out[-1][1] = st
        else:
            out.append([size, st])
    return out


try:  # low-level launch path (flagtree/metaX Triton 3.6)
    from triton.runtime import driver as _driver

    _get_stream = _driver.active.get_current_stream
except Exception:  # pragma: no cover
    _get_stream = None

# (tag, dtype, block, ...) -> compiled launcher; tag 0 = single/wide, 1 = fused
_LAUNCH_CACHE = {}


def _launch_flat_one(t, numel, dtype, tl_dt, block, wide, ntiles, dev):
    """Launch one flat tile kernel, reusing the compiled launcher.

    ``JITFunction.run`` rebuilds a specialization key, hashes it, validates the
    bindings and constructs launch metadata on every call.  For this
    two-pointer kernel that Python work costs more than the kernel itself on
    small tensors, so after the documented ``kernel[grid](...)`` call has
    compiled the binary its launcher is reused directly.  The cache key carries
    every fact Triton specializes on (dtype, constexpr block, int width,
    divisibility by 16, equal to one, base alignment), so a hit selects exactly
    the binary the JIT would have; a missing low-level API, an unknown key or
    any exception falls back to the documented launch.
    """
    ptr = t.data_ptr()
    key = (0, dtype, block, wide, numel % 16 == 0, numel == 1, ptr % 16 == 0)
    entry = _LAUNCH_CACHE.get(key)
    if entry is not None:
        try:
            entry[0](
                ntiles,
                1,
                1,
                _get_stream(dev),
                entry[1],
                entry[2],
                None,
                None,
                None,
                t,
                numel,
                tl_dt,
                block,
            )
            return
        except Exception:
            _LAUNCH_CACHE.pop(key, None)

    k = (_sinh_wide if wide else _sinh_single)[(ntiles,)](t, numel, tl_dt, block)
    if _get_stream is not None and k is not None:
        run = getattr(k, "run", None)
        if run is not None:
            packed = getattr(k, "packed_metadata", None)
            if packed is not None:
                _LAUNCH_CACHE[key] = (run, k.function, packed)


def _launch_fused(chunk, numel, dtype, tl_dt, bucket, block, ntiles, dev):
    """Fused multi-slot launch with the same compiled-launcher reuse.

    The twenty bound arguments make ``JITFunction.run``'s binder and binding
    validation the most expensive part of this call on small tensors, so the
    compiled launcher is cached too.  Every slot base must be 16-byte aligned
    for the cached binary to be the one the JIT would pick, so only that case
    is cached; a mixed-alignment group keeps the documented launch.
    """
    aligned = True
    for t in chunk:
        if t.data_ptr() & 15:
            aligned = False
            break
    if aligned:
        key = (1, dtype, block, bucket, numel % 16 == 0, numel == 1)
        entry = _LAUNCH_CACHE.get(key)
        if entry is not None:
            try:
                entry[0](
                    bucket,
                    ntiles,
                    1,
                    _get_stream(dev),
                    entry[1],
                    entry[2],
                    None,
                    None,
                    None,
                    *chunk,
                    numel,
                    tl_dt,
                    bucket,
                    block,
                )
                return
            except Exception:
                _LAUNCH_CACHE.pop(key, None)

    k = _sinh_fused[(bucket, ntiles)](*chunk, numel, tl_dt, bucket, block)
    if aligned and _get_stream is not None and k is not None:
        run = getattr(k, "run", None)
        if run is not None:
            packed = getattr(k, "packed_metadata", None)
            if packed is not None:
                _LAUNCH_CACHE[key] = (run, k.function, packed)


def _launch_flat(tensors, dtype, numel, block, dev):
    """Launch a group of same-length flat float tensors.

    A group of several tensors is normally served by one fused multi-slot
    launch, which amortizes the per-program slot chain.  Once a single tensor is
    large enough that this fixed work is irrelevant, the fused kernel is the
    slower choice: measured on 4096x4096 fp16 it sustains ~1.16 TB/s while the
    plain kernel on one 4096x4096 tensor sustains ~1.41 TB/s, the same shape and
    the same total bytes per slot.  Those groups are therefore split into one
    launch per tensor -- cheap now that each launch reuses a compiled launcher
    instead of paying Triton's Python dispatch -- and take the plain kernel's
    single-tensor BLOCK rather than the 4x multi-slot tile.
    """
    tl_dt = _TL_DTYPE[dtype]
    if len(tensors) > 1 and numel * tensors[0].element_size() >= _SPLIT_BYTES:
        block = _BLOCK[dtype]
        split = True
    else:
        split = False
    if split or len(tensors) == 1 or numel > _INT64_LIMIT:
        wide = numel > _INT64_LIMIT
        ntiles = triton.cdiv(numel, block)
        for t in tensors:
            _launch_flat_one(t, numel, dtype, tl_dt, block, wide, ntiles, dev)
        return

    ntiles = triton.cdiv(numel, block)
    count = len(tensors)
    for start in range(0, count, _MAX_SLOTS):
        chunk = tensors[start : start + _MAX_SLOTS]
        k = len(chunk)
        bucket = _bucket(k)
        if k < _MAX_SLOTS:
            chunk = list(chunk) + [chunk[0]] * (_MAX_SLOTS - k)
        _launch_fused(chunk, numel, dtype, tl_dt, bucket, block, ntiles, dev)


def _launch_strided(t, dtype):
    dims = _collapse(t.shape, t.stride())
    rank = len(dims)
    if rank > 8:
        raise NotImplementedError(
            "foreach_sinh_ fallback supports at most 8 strided dimensions"
        )
    sizes = [0] * 8
    divs = [0] * 8
    strides = [0] * 8
    div = 1
    for i in range(rank - 1, -1, -1):
        sizes[i] = dims[i][0]
        strides[i] = dims[i][1]
        divs[i] = div
        div *= dims[i][0]
    numel = t.numel()
    block = _pick_block(numel, dtype, False)
    _sinh_strided[(triton.cdiv(numel, block),)](
        t.data_ptr(),
        numel,
        *sizes,
        *divs,
        *strides,
        _TL_DTYPE[dtype],
        rank,
        block,
    )


def _raise_bad_dtype(dtype):
    """ATen's refusal for an in-place float unary applied to ``dtype``."""
    if dtype.is_complex:
        raise NotImplementedError(
            "foreach_sinh_ is not implemented for complex tensors"
        )
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        f"{_ATEN_NAME.get(dtype, str(dtype).replace('torch.', '').capitalize())}"
    )


def _launch_group(tensors, dtype, numel, multi, dev):
    block = _pick_block(numel, dtype, multi)
    if dev == torch.cuda.current_device():
        _launch_flat(tensors, dtype, numel, block, dev)
    else:
        with torch.cuda.device(dev):
            _launch_flat(tensors, dtype, numel, block, dev)


def _dispatch_valid(tensors):
    """Group and launch; every dtype is already known to be supported."""
    if not tensors:
        return
    flat = {}
    order = []
    strided = []
    for t in tensors:
        numel = t.numel()
        if numel == 0:
            continue
        dev = t.get_device()
        if dev < 0:
            raise NotImplementedError(
                "foreach_sinh_ requires CUDA tensors in this implementation"
            )
        dtype = t.dtype
        if t.is_contiguous():
            key = (dev, dtype, numel)
            bucket = flat.get(key)
            if bucket is None:
                bucket = flat[key] = []
                order.append(key)
            bucket.append(t)
        else:
            if torch._debug_has_internal_overlap(t) == 1:
                raise RuntimeError(_OVERLAP_MSG)
            if _is_dense_perm(t):
                key = (dev, dtype, numel)
                bucket = flat.get(key)
                if bucket is None:
                    bucket = flat[key] = []
                    order.append(key)
                bucket.append(t)
            else:
                strided.append((t, dtype))

    for key in order:
        dev, dtype, numel = key
        items = flat[key]
        _launch_group(items, dtype, numel, len(items) > 1, dev)

    for t, dtype in strided:
        _launch_strided(t, dtype)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_sinh_(self=None, *args):
    if args:
        if isinstance(self, torch.Tensor):
            tensors = [self]
            tensors.extend(args)
        elif isinstance(self, (list, tuple)):
            tensors = list(self)
            tensors.extend(args)
        else:
            raise TypeError("argument 'self' must be tuple of Tensors")
    elif self is None:
        raise TypeError("argument 'self' must be tuple of Tensors")
    elif isinstance(self, torch.Tensor):
        tensors = [self]
    elif isinstance(self, (list, tuple)):
        tensors = list(self)
    else:
        raise TypeError("argument 'self' must be tuple of Tensors")

    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    count = len(tensors)
    first = tensors[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(
            "expected Tensor as element 0 in argument 0, but got "
            f"{type(first).__name__}"
        )

    dtype0 = first.dtype
    if dtype0 not in _TL_DTYPE:
        _raise_bad_dtype(dtype0)

    # One pass: element type, dtype support (ATen validates lazily, so entries
    # before the offending one are still mutated), list homogeneity, and whether
    # the homogeneous list is laid out back-to-back so one launch covers it.
    numel0 = first.numel()
    dev0 = first.get_device()
    homogeneous = numel0 != 0 and dev0 >= 0 and first.is_contiguous()
    adjacent = homogeneous and count > 1
    prev_ptr = step = 0
    if adjacent:
        prev_ptr = first.data_ptr()
        step = numel0 * first.element_size()
        # An unaligned base would compile a scalar-access variant; prefer the
        # slot kernel there rather than paying for a slow merged launch.
        if prev_ptr % 16:
            adjacent = False
    bad = -1
    for i in range(1, count):
        t = tensors[i]
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"expected Tensor as element {i} in argument 0, but got "
                f"{type(t).__name__}"
            )
        dt = t.dtype
        if dt not in _TL_DTYPE:
            bad = i
            break
        if homogeneous and (
            dt != dtype0
            or t.numel() != numel0
            or t.get_device() != dev0
            or not t.is_contiguous()
        ):
            homogeneous = False
            adjacent = False
        elif adjacent:
            ptr = t.data_ptr()
            if ptr != prev_ptr + step:
                adjacent = False
            else:
                prev_ptr = ptr

    if bad >= 0:
        if bad:
            _dispatch_valid(tensors[:bad])
        _raise_bad_dtype(tensors[bad].dtype)

    if homogeneous:
        if adjacent:
            # The list is one contiguous span; address it through the first
            # tensor so the launch keeps a single pointer-typed argument.
            _launch_group([first], dtype0, numel0 * count, False, dev0)
        else:
            _launch_group(tensors, dtype0, numel0, count > 1, dev0)
        return None

    _dispatch_valid(tensors)
    return None
