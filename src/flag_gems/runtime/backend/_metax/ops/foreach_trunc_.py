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

"""Self-contained Triton implementation of ``aten::_foreach_trunc_``.

Applies truncation toward zero to every tensor of a ``Tensor[]`` in place,
preserving storage, shape and strides, and returns ``None``.

Design:
  * flat coalesced kernel for flat-addressable (contiguous / non-overlapping-and-
    dense) tensors;
  * launch-bound groups (equal element counts, <= _BATCH_MAX_ELEM elements) are
    folded into ONE launch on a 2-D grid (chunk, tensor) whose argument list is
    just the base addresses as plain integers plus a single shared element
    count, so the per-call argument cost is the cheap integer path rather than
    16 tensor arguments;
  * larger groups use one launch per tensor: their kernels are long enough to
    hide the host enqueue, and separate launches keep full vectorization;
  * generic collapsed-strided kernel for gappy views, with the smallest-stride
    dimension innermost so accesses stay coalesced;
  * truncation is exact: ``x >= 0 ? floor(x) : ceil(x)`` in fp32 (fp64 kept in
    fp64), which preserves NaN, +/-Inf and signed zero.  Integral dtypes are an
    identity store, never routed through float.

Host launch: the compiled kernel's own entry point is reused across calls with
the same specialization (only argument values change), which skips the per-call
binder, cache-key build, global-value audit, node-binding validation and
runtime-context setup of ``kernel[grid](...)``.  It is used only while no launch
hook is registered and falls back to the ordinary launch on any surprise, so
the operator stays portable across Triton builds.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_DIMS = 8
_MAX_GROUP = 16
_BATCH_MAX_ELEM = 1 << 20
_LAUNCH_KW = {"num_warps": _NUM_WARPS}
_BIG_ELEM = 1 << 20
_BIG_BLOCK = 4096
_BIG_KW = {"num_warps": 8}

_ALLOWED_DTYPES = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )
)

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
}


@triton.jit
def _trunc_vals(x):
    """trunc(x) elementwise, exact for NaN / +/-Inf / -0.0 and all magnitudes."""
    if x.dtype.is_floating():
        if x.dtype == tl.float64:
            y = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
        else:
            xf = x.to(tl.float32)
            y = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf)).to(x.dtype)
    else:
        y = x
    return y


@triton.jit
def _trunc_slice(ptr, off, n, EVEN: tl.constexpr):
    if EVEN:
        tl.store(ptr + off, _trunc_vals(tl.load(ptr + off)))
    else:
        mask = off < n
        tl.store(
            ptr + off, _trunc_vals(tl.load(ptr + off, mask=mask, other=0)), mask=mask
        )


@triton.jit
def _trunc_flat_kernel(x_ptr, n_elements, EVEN: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _trunc_slice(x_ptr, off, n_elements, EVEN)


@triton.jit
def _trunc_group_kernel(
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
    n_elements,
    NT: tl.constexpr,
    DT: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for ``NT`` same-dtype tensors of equal element count.

    Axis 0 is the chunk inside a tensor, axis 1 selects the tensor.  Base
    addresses arrive as plain integers, so the host-side argument list costs one
    cheap integer per tensor instead of one Tensor object, and the kernel keeps
    a single load/store site.  Used only for launch-bound sizes where the access
    pattern is irrelevant; larger tensors take the tensor-argument path so their
    vectorization is derived from the data pointer.
    """
    t = tl.program_id(1)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = a0
    if NT > 1:
        a = tl.where(t == 1, a1, a)
    if NT > 2:
        a = tl.where(t == 2, a2, a)
    if NT > 3:
        a = tl.where(t == 3, a3, a)
    if NT > 4:
        a = tl.where(t == 4, a4, a)
    if NT > 5:
        a = tl.where(t == 5, a5, a)
    if NT > 6:
        a = tl.where(t == 6, a6, a)
    if NT > 7:
        a = tl.where(t == 7, a7, a)
    if NT > 8:
        a = tl.where(t == 8, a8, a)
    if NT > 9:
        a = tl.where(t == 9, a9, a)
    if NT > 10:
        a = tl.where(t == 10, a10, a)
    if NT > 11:
        a = tl.where(t == 11, a11, a)
    if NT > 12:
        a = tl.where(t == 12, a12, a)
    if NT > 13:
        a = tl.where(t == 13, a13, a)
    if NT > 14:
        a = tl.where(t == 14, a14, a)
    if NT > 15:
        a = tl.where(t == 15, a15, a)
    _trunc_slice(a.to(tl.pointer_type(DT)), off, n_elements, EVEN)


@triton.jit
def _trunc_strided_kernel(
    x_ptr,
    n_elements,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NDIM: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rem = idx.to(tl.int64)
    off = tl.zeros((BLOCK,), dtype=tl.int64) + s0 * (rem % c0)
    rem = rem // c0
    if NDIM > 1:
        off += s1 * (rem % c1)
        rem = rem // c1
    if NDIM > 2:
        off += s2 * (rem % c2)
        rem = rem // c2
    if NDIM > 3:
        off += s3 * (rem % c3)
        rem = rem // c3
    if NDIM > 4:
        off += s4 * (rem % c4)
        rem = rem // c4
    if NDIM > 5:
        off += s5 * (rem % c5)
        rem = rem // c5
    if NDIM > 6:
        off += s6 * (rem % c6)
        rem = rem // c6
    if NDIM > 7:
        off += s7 * rem

    if EVEN:
        x = tl.load(x_ptr + off)
        tl.store(x_ptr + off, _trunc_vals(x))
    else:
        mask = idx < n_elements
        x = tl.load(x_ptr + off, mask=mask, other=0)
        tl.store(x_ptr + off, _trunc_vals(x), mask=mask)


# ---------------------------------------------------------------------------
# host launch
# ---------------------------------------------------------------------------

_DIRECT_BASE = None
_DIRECT_KERNELS = {}


def _hooks_clear():
    """True when no launch hook is registered (flagtree exposes a HookChain)."""
    try:
        from triton import knobs

        for chain in (
            knobs.runtime.launch_enter_hook,
            knobs.runtime.launch_exit_hook,
        ):
            if chain is None:
                continue
            calls = getattr(chain, "calls", None)
            if calls is None or len(calls):
                return False
        return True
    except Exception:
        return False


def _direct_ok():
    """Whether the cached direct launch path may be used for this call.

    The compiled kernel's own entry point is reused across calls with the same
    specialization, which skips the per-call binder, cache-key build,
    global-value audit, node-binding validation and runtime-context setup of
    ``kernel[grid](...)``.  Only used while no launch hook is registered, so any
    instrumentation still sees every launch through the ordinary path.
    """
    global _DIRECT_BASE
    if not _hooks_clear():
        return False
    if _DIRECT_BASE is None:
        try:
            dist = ()
            ctx_cls = getattr(triton.runtime.jit, "DistributedRtContext", None)
            if ctx_cls is not None:
                ctx = ctx_cls()
                if getattr(ctx, "is_lite_mode", False):
                    dist = (int(ctx.comm_ptr), int(ctx.mem_ptr))
            _DIRECT_BASE = {"dist": dist, "drv": triton.runtime.driver.active}
        except Exception:
            _DIRECT_BASE = False
    return _DIRECT_BASE is not False


def _direct(jit_fn, key, grid, argv, kw):
    """Launch ``jit_fn`` through its compiled kernel; False means fall back."""
    kern = _DIRECT_KERNELS.get(key)
    if kern is None:
        try:
            kern = jit_fn.warmup(*argv, grid=grid, **kw)
        except Exception:
            _DIRECT_KERNELS[key] = False
            return False
        if hasattr(kern, "result"):
            try:
                kern = kern.result()
            except Exception:
                _DIRECT_KERNELS[key] = False
                return False
        if not (
            kern is not None
            and hasattr(kern, "foreach_trunc_")
            and hasattr(kern, "function")
            and hasattr(kern, "packed_metadata")
        ):
            _DIRECT_KERNELS[key] = False
            return False
        _DIRECT_KERNELS[key] = kern
    elif kern is False:
        return False
    try:
        base = _DIRECT_BASE
        drv = base["drv"]
        stream = drv.get_current_stream(drv.get_current_device())
        kern.run(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            stream,
            kern.function,
            kern.packed_metadata,
            None,
            None,
            None,
            *base["dist"],
            *argv,
        )
        return True
    except Exception:
        _DIRECT_KERNELS[key] = False
        return False


def _normalize(self, others):
    if others:
        head = list(self) if isinstance(self, (list, tuple)) else [self]
        tensors = head + list(others)
    else:
        tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
    return tensors


def _is_flat_addressable(t):
    """True when ``[data_ptr, data_ptr + numel)`` covers exactly this tensor."""
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


def _collapsed_dims(t):
    """Collapsed (size, stride) dims, smallest stride innermost (coalesced)."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return [(1, 1)]
    cols = []
    for sz, st in reversed(dims):
        if cols and st == cols[-1][0] * cols[-1][1]:
            cols[-1] = (cols[-1][0] * sz, cols[-1][1])
        else:
            cols.append((sz, st))
    return sorted(cols, key=lambda p: p[1])


def _launch_flat(t, n, fast):
    # Streaming sizes get a larger per-program payload (fewer, longer-lived
    # programs to hide memory latency); launch-bound sizes keep the small tile.
    if n >= _BIG_ELEM:
        block = _BIG_BLOCK
        kw = _BIG_KW
    else:
        block = _BLOCK
        kw = _LAUNCH_KW
    grid = ((n + block - 1) // block,)
    even = (n % block) == 0
    if fast:
        key = (
            0,
            block,
            t.dtype,
            even,
            (t.data_ptr() & 15) == 0,
            n % 16 == 0,
            n == 1,
        )
        if _direct(_trunc_flat_kernel, key, grid, (t, n, even, block), kw):
            return
    _trunc_flat_kernel[grid](t, n, EVEN=even, BLOCK=block, **kw)


def _launch_strided(t):
    n = t.numel()
    dims = _collapsed_dims(t)
    ndim = len(dims)
    if ndim > _MAX_DIMS:
        raise NotImplementedError("foreach_trunc_: unsupported layout (%d dims)" % ndim)
    sizes = [d[0] for d in dims] + [1] * (_MAX_DIMS - ndim)
    strides = [d[1] for d in dims] + [0] * (_MAX_DIMS - ndim)
    _trunc_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t,
        n,
        *sizes,
        *strides,
        NDIM=ndim,
        EVEN=(n % _BLOCK == 0),
        BLOCK=_BLOCK,
        **_LAUNCH_KW,
    )


def _launch_group(group, fast):
    """One launch covers every ``(tensor, numel)`` pair of ``group``."""
    nt = len(group)
    if nt == 1:
        t, n = group[0]
        _launch_flat(t, n, fast)
        return
    first_n = group[0][1]
    uniform = True
    for _, n in group:
        if n != first_n:
            uniform = False
            break
    # Batching pays only while a single tensor's kernel is shorter than the host
    # enqueue it replaces: then the launches are host bound and folding them
    # removes real time.  Once a tensor's own streaming kernel is longer than
    # that, one launch per tensor is faster because the host enqueue overlaps
    # the earlier kernels on the device and each launch keeps full vectorization.
    if not uniform or first_n > _BATCH_MAX_ELEM:
        for t, n in group:
            _launch_flat(t, n, fast)
        return
    max_n = first_n
    addrs = []
    aligned = True
    for t, _ in group:
        p = t.data_ptr()
        addrs.append(p)
        if p & 15:
            aligned = False
    if nt < _MAX_GROUP:
        addrs += [addrs[0]] * (_MAX_GROUP - nt)
    grid = ((max_n + _BLOCK - 1) // _BLOCK, nt)
    even = (max_n % _BLOCK) == 0
    dtype = _TL_DTYPE[group[0][0].dtype]
    if fast:
        key = (1, group[0][0].dtype, even, aligned, max_n % 16 == 0, max_n == 1, nt)
        if _direct(
            _trunc_group_kernel,
            key,
            grid,
            (*addrs, max_n, nt, dtype, even, _BLOCK),
            _LAUNCH_KW,
        ):
            return
    _trunc_group_kernel[grid](
        *addrs,
        max_n,
        NT=nt,
        DT=dtype,
        EVEN=even,
        BLOCK=_BLOCK,
        **_LAUNCH_KW,
    )


def foreach_trunc_(self, *others):
    tensors = _normalize(self, others)

    fast = _direct_ok()
    groups = {}
    strided = []
    for t in tensors:
        dtype = t.dtype
        if dtype not in _ALLOWED_DTYPES:
            # NotImplementedError subclasses RuntimeError, so this satisfies
            # either expectation for ATen's refusal.
            raise NotImplementedError(
                "\"foreach\" not implemented for '%s'"
                % str(dtype).replace("torch.", "")
            )
        contig = t.is_contiguous()
        # A contiguous tensor of more than one element maps distinct addresses
        # to distinct elements, so the overlap probe is only needed for views.
        if not contig and torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        n = t.numel()
        if n == 0:
            continue
        if contig or _is_flat_addressable(t):
            key = (t.device, dtype)
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = [(t, n)]
            else:
                bucket.append((t, n))
        else:
            strided.append(t)

    for group in groups.values():
        if len(group) <= _MAX_GROUP:
            _launch_group(group, fast)
        else:
            for i in range(0, len(group), _MAX_GROUP):
                _launch_group(group[i : i + _MAX_GROUP], fast)
    for t in strided:
        _launch_strided(t)
    return None
