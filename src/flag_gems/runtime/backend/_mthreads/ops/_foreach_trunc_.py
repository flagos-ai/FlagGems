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
"""In-place foreach trunc: ``aten::_foreach_trunc_(Tensor(a!)[] self) -> ()``.

Applies ``trunc`` toward zero to every element of every tensor in the list,
mutating the tensors in place (same storage, same strides) and returning
``None``.

Dispatch policy (mirrors the measured ATen behaviour of the pinned build):

* floating dtypes -> Triton kernel, computed as
  ``where(x >= 0, floor(x), ceil(x))`` in fp32 (exact for fp16/bf16 because
  every representable value is representable in fp32 and trunc of a
  representable value is representable in the source dtype).  ``floor`` /
  ``ceil`` are used instead of an int round trip so NaN, +-inf and the sign of
  -0.0 are preserved.
* integral dtypes -> identity, no device work.
* bool and complex -> ``RuntimeError`` (ATen rejects them).

Layout handling never copies: flat traversal is used for contiguous and
non-overlapping-and-dense tensors (memory order visits exactly the tensor's
element set, so in-place elementwise math is exact), and a stride-aware kernel
is used for gappy views.

Launch policy: the common foreach list (one dtype, uniform ``numel``, all
flat-addressable) is served with *no* metadata staging copy.  Small groups are
fused into a single launch with one pointer argument per slot (grid
``(tiles, slots)``), because there the launch count is what costs; large groups
use one lean single-pointer launch per tensor, because the device work hides
those launches and the lean kernel keeps the statically known pointer
alignment.  On top of that, repeated launches of the same signature go through
a pre-bound ``CompiledKernel`` so the per-call Python dispatch of the JIT
wrapper is not paid on the launch-bound small workloads.  Anything else falls
back to per-tensor flat launches plus the stride-aware kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_trunc_ as default__foreach_trunc_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_MAX_SLOTS = 16
_MAXD = 8
_META_BLOCK = 3 + 2 * _MAXD
_BLOCK = 1024
# A same-dtype group is fused into one launch only while a single tensor's
# tile is small enough that kernel launches cannot be hidden behind device
# work; above this the lean one-pointer kernel is used per tensor.
_FUSE_MAX_BYTES = 1 << 22

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _trunc_val(x):
    """Exact trunc toward zero, dtype preserving."""
    if x.dtype == tl.float64:
        return tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    y = x.to(tl.float32)
    return tl.where(y >= 0, tl.floor(y), tl.ceil(y)).to(x.dtype)


@triton.jit
def _trunc_flat_kernel(
    x_ptr,
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, _trunc_val(x))
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        tl.store(x_ptr + offs, _trunc_val(x), mask=mask)


@triton.jit
def _trunc_multi_kernel(
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
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch for a whole same-numel dtype group.

    Grid is ``(tiles, slots)`` so the tile axis (which can be huge) stays on
    axis 0 and the slot axis stays inside the 65535 limit of axis 1.  The
    pointers are real tensor arguments, which keeps the compiler's pointer
    alignment (divisibility) information and therefore the wide vectorized
    accesses.
    """
    slot = tl.program_id(1)
    x_ptr = x0
    if slot == 1:
        x_ptr = x1
    elif slot == 2:
        x_ptr = x2
    elif slot == 3:
        x_ptr = x3
    elif slot == 4:
        x_ptr = x4
    elif slot == 5:
        x_ptr = x5
    elif slot == 6:
        x_ptr = x6
    elif slot == 7:
        x_ptr = x7
    elif slot == 8:
        x_ptr = x8
    elif slot == 9:
        x_ptr = x9
    elif slot == 10:
        x_ptr = x10
    elif slot == 11:
        x_ptr = x11
    elif slot == 12:
        x_ptr = x12
    elif slot == 13:
        x_ptr = x13
    elif slot == 14:
        x_ptr = x14
    elif slot == 15:
        x_ptr = x15
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, _trunc_val(x))
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        tl.store(x_ptr + offs, _trunc_val(x), mask=mask)


@triton.jit
def _trunc_strided_kernel(
    meta_ptr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
    MAXD: tl.constexpr,
    META_BLOCK: tl.constexpr,
):
    """Stride-aware in-place trunc for gappy views.

    ``meta_ptr`` is int64, ``META_BLOCK`` entries per tensor:
    ``[ptr, numel, ndim, shape[MAXD], stride[MAXD]]``.
    """
    t = tl.program_id(0)
    base = t * META_BLOCK
    ptr = tl.load(meta_ptr + base).to(tl.pointer_type(DT))
    n = tl.load(meta_ptr + base + 1)
    ndim = tl.load(meta_ptr + base + 2)
    offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    off = tl.zeros([BLOCK], dtype=tl.int64)
    k = 0
    while k < ndim:
        d = ndim - 1 - k
        shp = tl.load(meta_ptr + base + 3 + d)
        strd = tl.load(meta_ptr + base + 3 + MAXD + d)
        off = off + (rem % shp) * strd
        rem = rem // shp
        k = k + 1
    x = tl.load(ptr + off, mask=mask, other=0)
    tl.store(ptr + off, _trunc_val(x), mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _is_flat(t):
    """True when ``[data_ptr, data_ptr + numel)`` covers exactly the tensor."""
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


def _collapse(t):
    """Drop size-1 dims and merge adjacent dims that are contiguous together.

    Merging dim ``i`` (outer) with dim ``i+1`` is legal when
    ``stride[i] == shape[i+1] * stride[i+1]``; the merged dim then advances
    with ``stride[i+1]`` (the inner one), because a flat index walks the
    tensor in memory order.
    """
    shape = []
    strides = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if shape and strides[-1] == st * sz:
            shape[-1] *= sz
            strides[-1] = st
        else:
            shape.append(sz)
            strides.append(st)
    if not shape:
        shape = [1]
        strides = [0]
    return shape, strides


def _expand_entries(t):
    """Yield ``(tensor, shape, strides)`` rows with at most ``_MAXD`` dims.

    Over-long layouts are split along the outermost dim into zero-copy views.
    """
    out = []
    stack = [t]
    while stack:
        cur = stack.pop()
        shape, strides = _collapse(cur)
        if len(shape) <= _MAXD:
            out.append((cur, shape, strides))
        else:
            for i in range(cur.shape[0] - 1, -1, -1):
                stack.append(cur[i])
    return out


def _launch_strided(rows, dtype, device):
    meta = []
    max_numel = 0
    for t, shape, strides in rows:
        numel = t.numel()
        if numel > max_numel:
            max_numel = numel
        meta.append(t.data_ptr())
        meta.append(numel)
        meta.append(len(shape))
        meta.extend(shape)
        meta.extend([1] * (_MAXD - len(shape)))
        meta.extend(strides)
        meta.extend([0] * (_MAXD - len(strides)))
    if max_numel == 0:
        return
    meta_t = torch.tensor(meta, dtype=torch.int64, device=device)
    grid = (len(rows), triton.cdiv(max_numel, _BLOCK))
    _trunc_strided_kernel[grid](
        meta_t, _TL_DTYPE[dtype], BLOCK=_BLOCK, MAXD=_MAXD, META_BLOCK=_META_BLOCK
    )


# ---------------------------------------------------------------------------
# low-overhead launch path
# ---------------------------------------------------------------------------
#
# A plain ``kernel[grid](...)`` pays Triton's Python dispatch on every call
# (binding, specialization, cache lookup).  For the tiny foreach workloads that
# host cost is the whole measured latency, so the first call for a given
# signature compiles the kernel once with ``warmup`` and later calls invoke the
# resulting ``CompiledKernel`` launcher directly.  Any failure permanently
# disables the shortcut, and the ordinary path is used instead.
#
# The shortcut is only taken when the specialization class is unambiguous:
# ``n`` divisible by 16 and greater than 1 (so it is neither folded to a
# constant nor value-specialized) and every pointer 16-byte aligned.

_DRIVER = None
_DEVICE = None
_FAST_OK = True
_FAST_CACHE = {}


def _fast_launch(jit_fn, key, args, even, grid0, grid1):
    """Launch ``args`` through a pre-bound CompiledKernel; False if unusable."""
    global _DRIVER, _DEVICE, _FAST_OK
    if not _FAST_OK:
        return False
    entry = _FAST_CACHE.get(key)
    if entry is None:
        try:
            ck = jit_fn.warmup(*args, grid=(1, 1, 1), BLOCK=_BLOCK, EVEN=even)
            # entry[3] is the driver wrapper's raw C launcher, tried first so
            # the per-call scratch-bookkeeping of the Python wrapper is skipped;
            # it is dropped on the first failure and the wrapper is used.
            entry = [
                ck.run,
                ck.function,
                ck.packed_metadata,
                getattr(ck.run, "launch", None),
            ]
            _FAST_CACHE[key] = entry
        except Exception:
            _FAST_OK = False
            return False
    try:
        if _DRIVER is None:
            _DRIVER = triton.runtime.driver.active
            _DEVICE = _DRIVER.get_current_device()
        stream = _DRIVER.get_current_stream(_DEVICE)
        # launcher signature: grid x3, stream, function, packed metadata,
        # launch metadata, enter/exit hooks, kernel args, global scratch,
        # profile scratch.
        raw = entry[3]
        if raw is not None:
            try:
                raw(
                    grid0,
                    grid1,
                    1,
                    stream,
                    entry[1],
                    entry[2],
                    None,
                    None,
                    None,
                    *args,
                    None,
                    None,
                )
                return True
            except Exception:
                entry[3] = None
        entry[0](
            grid0,
            grid1,
            1,
            stream,
            entry[1],
            entry[2],
            None,
            None,
            None,
            *args,
            None,
            None,
        )
        return True
    except Exception:
        _FAST_OK = False
        return False


def _launch_uniform(tensors, ns, n, esize):
    # Reached only after run() verified one dtype/device, uniform numel,
    # contiguity and 16-byte pointer alignment for the whole list.
    grid0 = (n + _BLOCK - 1) // _BLOCK
    even = (n % _BLOCK) == 0
    fastable = n > 1 and (n & 15) == 0
    if ns == 1:
        t = tensors[0]
        if fastable and _fast_launch(
            _trunc_flat_kernel, (1, t.dtype, even), (t, n), even, grid0, 1
        ):
            return
        _trunc_flat_kernel[(grid0,)](t, n, BLOCK=_BLOCK, EVEN=even)
        return
    if n * esize <= _FUSE_MAX_BYTES:
        xs = list(tensors)
        if ns < _MAX_SLOTS:
            xs.extend([xs[0]] * (_MAX_SLOTS - ns))
        xs.append(n)
        if fastable and _fast_launch(
            _trunc_multi_kernel,
            (ns, tensors[0].dtype, even),
            xs,
            even,
            grid0,
            ns,
        ):
            return
        _trunc_multi_kernel[(grid0, ns)](*xs, BLOCK=_BLOCK, EVEN=even)
        return
    # Large tiles: device work hides the launch cost, so keep the lean
    # single-pointer kernel.
    if fastable:
        key = (1, tensors[0].dtype, even)
        for t in tensors:
            if not _fast_launch(_trunc_flat_kernel, key, (t, n), even, grid0, 1):
                _trunc_flat_kernel[(grid0,)](t, n, BLOCK=_BLOCK, EVEN=even)
        return
    for t in tensors:
        _trunc_flat_kernel[(grid0,)](t, n, BLOCK=_BLOCK, EVEN=even)


def _general(tensors):
    """Validate everything, then launch per (device, dtype, layout) bucket."""
    flat = {}
    strided = {}
    for t in tensors:
        dtype = t.dtype
        if dtype is torch.bool or dtype.is_complex:
            raise RuntimeError(
                "\"foreach\" not implemented for '%s'"
                % str(dtype).replace("torch.", "")
            )
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        if not dtype.is_floating_point or t.numel() == 0:
            continue
        key = (t.device, dtype)
        if _is_flat(t):
            bucket = flat.get(key)
            if bucket is None:
                flat[key] = [t]
            else:
                bucket.append(t)
        else:
            rows = _expand_entries(t)
            bucket = strided.get(key)
            if bucket is None:
                strided[key] = rows
            else:
                bucket.extend(rows)

    for (device, dtype), ts in flat.items():
        for t in ts:
            numel = t.numel()
            _trunc_flat_kernel[((numel + _BLOCK - 1) // _BLOCK,)](
                t, numel, BLOCK=_BLOCK, EVEN=(numel % _BLOCK) == 0
            )

    for (device, dtype), rows in strided.items():
        _launch_strided(rows, dtype, device)


def _foreach_trunc_(self=None, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_TRUNC_")
    if self is None and "self" in kwargs:
        self = kwargs.pop("self")
    if (
        isinstance(self, torch.Tensor)
        and args
        and all(isinstance(a, torch.Tensor) for a in args)
    ):
        self = (self,) + tuple(args)
    tensors = self
    if not isinstance(tensors, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in tensors
    ):
        return default__foreach_trunc_(tensors)
    ns = len(tensors)
    if ns == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    t0 = tensors[0]
    if not isinstance(t0, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")

    dtype = t0.dtype
    n = t0.numel()
    # Fast path: one dtype, uniform numel, every tensor flat-addressable.
    # Contiguous tensors cannot have internal overlap, so the overlap guard is
    # not needed here.
    if ns <= _MAX_SLOTS and n > 0 and dtype in _TL_DTYPE and t0.is_contiguous():
        device = t0.get_device()
        ok = (t0.data_ptr() & 15) == 0
        for i in range(1, ns):
            t = tensors[i]
            if not isinstance(t, torch.Tensor):
                raise TypeError("argument 'self' must be tuple of Tensors")
            if (
                t.dtype is not dtype
                or t.numel() != n
                or not t.is_contiguous()
                or t.get_device() != device
                or (t.data_ptr() & 15)
            ):
                ok = False
                break
        if ok:
            _launch_uniform(tensors, ns, n, t0.element_size())
            return None

    _general(tensors)
    return None
