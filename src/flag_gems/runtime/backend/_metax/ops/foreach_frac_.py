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

"""In-place ``_foreach_frac_``: for every tensor x in the list, x <- frac(x) = x - trunc(x).

Reference semantics (ATen ``_foreach_frac_``):
* floating dtypes only; integer/bool raise ``RuntimeError`` ("not implemented for ..."),
* the mutation is strictly in place (storage pointer, offset and strides preserved),
* ``returns None``,
* ``frac(+-inf) = NaN``, ``frac(NaN) = NaN``, ``frac(-0.0) = +0.0``, ``frac(x) = 0`` for integral x,
* an empty tensor list raises ``RuntimeError``.

Launch architecture (measured on this target: ~13us fixed + ~15us per 3-argument
Triton launch and ~1us per extra kernel argument, all inside the event-timed
window):

* small tensors: the whole list is fused into one launch (2D grid
  ``(chunk, tensor)``, every tensor passed as a real pointer argument, so no
  host->device metadata tensor is needed);
* large tensors: one launch per tensor, because their device work hides the
  launch cost and the branchy fused kernel loses ~14% of memory throughput on
  2-byte element types;
* both paths replay a cached ``CompiledKernel`` through its launcher whenever
  the argument specialization is exactly the compiled one (16-byte aligned
  pointers, sizes a multiple of 16 and not 1, no profiler hook installed).
  Otherwise the ordinary ``kernel[grid](...)`` path runs, so the fast path can
  never change which binary or which values are produced.
"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

_BLOCK = 1024
_NDIM = 8
_MAX_T = 16
_FUSE_MAX_N = 65536

_ERR_EMPTY = "Tensor list must have at least one tensor."
_ERR_LIST = "argument 'self' must be tuple of Tensors"

_Tensor = torch.Tensor
_LAUNCH = {}
_DIST = None


@triton.jit
def _frac_body(x_ptr, idx, n, BLOCK: tl.constexpr):
    """In-place frac of the BLOCK elements at ``idx`` of one dense buffer."""
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = x - tl.where(x >= 0, tl.math.floor(x), tl.math.ceil(x))
    else:
        xf = x.to(tl.float32)
        y = (xf - tl.where(xf >= 0, tl.math.floor(xf), tl.math.ceil(xf))).to(x.dtype)
    tl.store(x_ptr + idx, y, mask=mask)


@triton.jit
def _frac_flat_kernel(x_ptr, n, BLOCK: tl.constexpr):
    """Single-tensor in-place frac over a dense buffer."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _frac_body(x_ptr, idx, n, BLOCK)


@triton.jit
def _frac_multi_kernel(
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
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Whole small-tensor list in one launch: axis 1 selects the tensor.

    ``NT`` is the number of live slots; the constexpr guards remove the unused
    branches.  Slots past each tensor's own element count are fully masked.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t = tl.program_id(1)
    if t == 0:
        _frac_body(x0, idx, n0, BLOCK)
    if NT > 1:
        if t == 1:
            _frac_body(x1, idx, n1, BLOCK)
    if NT > 2:
        if t == 2:
            _frac_body(x2, idx, n2, BLOCK)
    if NT > 3:
        if t == 3:
            _frac_body(x3, idx, n3, BLOCK)
    if NT > 4:
        if t == 4:
            _frac_body(x4, idx, n4, BLOCK)
    if NT > 5:
        if t == 5:
            _frac_body(x5, idx, n5, BLOCK)
    if NT > 6:
        if t == 6:
            _frac_body(x6, idx, n6, BLOCK)
    if NT > 7:
        if t == 7:
            _frac_body(x7, idx, n7, BLOCK)
    if NT > 8:
        if t == 8:
            _frac_body(x8, idx, n8, BLOCK)
    if NT > 9:
        if t == 9:
            _frac_body(x9, idx, n9, BLOCK)
    if NT > 10:
        if t == 10:
            _frac_body(x10, idx, n10, BLOCK)
    if NT > 11:
        if t == 11:
            _frac_body(x11, idx, n11, BLOCK)
    if NT > 12:
        if t == 12:
            _frac_body(x12, idx, n12, BLOCK)
    if NT > 13:
        if t == 13:
            _frac_body(x13, idx, n13, BLOCK)
    if NT > 14:
        if t == 14:
            _frac_body(x14, idx, n14, BLOCK)
    if NT > 15:
        if t == 15:
            _frac_body(x15, idx, n15, BLOCK)


@triton.jit
def _frac_strided_kernel(
    x_ptr,
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
):
    """In-place frac over a logical index space mapped through real strides.

    ``s0..s7`` / ``t0..t7`` describe the collapsed shape in *fastest-varying first*
    order; unused slots use ``s = 1`` (so the coordinate is 0) and ``t = 0``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx.to(tl.int64)
    off = tl.zeros((BLOCK,), tl.int64)
    c = rem % s0
    rem = rem // s0
    off += c * t0
    c = rem % s1
    rem = rem // s1
    off += c * t1
    c = rem % s2
    rem = rem // s2
    off += c * t2
    c = rem % s3
    rem = rem // s3
    off += c * t3
    c = rem % s4
    rem = rem // s4
    off += c * t4
    c = rem % s5
    rem = rem // s5
    off += c * t5
    c = rem % s6
    rem = rem // s6
    off += c * t6
    c = rem % s7
    rem = rem // s7
    off += c * t7

    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = x - tl.where(x >= 0, tl.math.floor(x), tl.math.ceil(x))
    else:
        xf = x.to(tl.float32)
        y = (xf - tl.where(xf >= 0, tl.math.floor(xf), tl.math.ceil(xf))).to(x.dtype)
    tl.store(x_ptr + off, y, mask=mask)


def _dist_params():
    """flagtree distributed-launch extras, mirroring JITFunction.run."""
    global _DIST
    if _DIST is None:
        _DIST = []
        try:
            from triton.runtime.jit import DistributedRtContext

            ctx = DistributedRtContext()
            if ctx.is_lite_mode:
                _DIST = [ctx.comm_ptr, ctx.mem_ptr]
        except Exception:  # backend without the extension
            _DIST = []
    return _DIST


def _hooks_clear():
    """True when no launch hook is registered (flagtree exposes HookChains).

    ``knobs.runtime.launch_*_hook`` are always-present ``HookChain`` objects
    whose ``calls`` list holds the registered hooks.  When anything is
    registered (or the shape is unknown) the ordinary launch path runs, so a
    profiler/tracer can never miss a launch because of the fast path.
    """
    knobs = getattr(triton, "knobs", None)
    if knobs is None:
        return True
    runtime = knobs.runtime
    for name in ("launch_enter_hook", "launch_exit_hook"):
        chain = getattr(runtime, name, None)
        if chain is None:
            continue
        calls = getattr(chain, "calls", None)
        if calls is None or calls:
            return False
    return True


def _host_args():
    drv = _driver.active
    return drv.get_current_stream(drv.get_current_device())


def _bind(kern, key):
    try:
        kern._init_handles()
        entry = (kern.run, kern.function, kern.packed_metadata, _dist_params())
        _LAUNCH[key] = entry
    except Exception:
        pass


def _replay(entry, grid_0, grid_1, args):
    run_fn, fn, meta, dist = entry
    run_fn(
        grid_0,
        grid_1,
        1,
        _host_args(),
        fn,
        meta,
        None,
        None,
        None,
        *dist,
        *args,
    )


def _flat_addressable(t):
    """True when ``[data_ptr, data_ptr + numel)`` covers exactly this tensor.

    Equivalent to ATen's ``is_non_overlapping_and_dense`` (that method is not
    exposed on this torch build): a transposed-but-dense view qualifies, a gappy
    slice such as ``x[:, ::2]`` does not.  Flat memory-order traversal is exact
    for the qualified set because every element is read and written at its own
    address.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for st, sz in sorted((s, z) for s, z in zip(t.stride(), t.shape) if z != 1):
        if st != expected:
            return False
        expected *= sz
    return True


def _collapse(shape, stride):
    """Collapse a strided layout to (fastest-first) shape/stride pairs, up to _NDIM."""
    pairs = [(sz, st) for sz, st in zip(shape, stride) if sz != 1]
    merged = []
    for sz, st in reversed(pairs):
        if merged:
            inner_sz, inner_st = merged[-1]
            if st == inner_st * inner_sz:
                merged[-1] = (inner_sz * sz, inner_st)
                continue
        merged.append((sz, st))
    return merged


def _launch_strided(t, n):
    dims = _collapse(t.shape, t.stride())
    if len(dims) > _NDIM:
        raise RuntimeError("frac: tensor rank too large for the strided kernel")
    sizes = [1] * _NDIM
    strides = [0] * _NDIM
    for i, (sz, st) in enumerate(dims):
        sizes[i] = sz
        strides[i] = st
    _frac_strided_kernel[(triton.cdiv(n, _BLOCK),)](t, n, *sizes, *strides, _BLOCK)


def _launch_flat(t, n):
    """One launch for one dense tensor, replayed from the compiled-kernel cache."""
    grid = (n + _BLOCK - 1) // _BLOCK
    if n != 1 and not (n & 15) and not (t.data_ptr() & 15) and _hooks_clear():
        key = (0, t.dtype)
        entry = _LAUNCH.get(key)
        if entry is not None:
            _replay(entry, grid, 1, (t, n, _BLOCK))
            return
        _bind(_frac_flat_kernel[(grid,)](t, n, _BLOCK), key)
        return
    _frac_flat_kernel[(grid,)](t, n, _BLOCK)


def _launch_multi(dense):
    """One launch per up-to-_MAX_T small dense tensors."""
    for start in range(0, len(dense), _MAX_T):
        chunk = dense[start : start + _MAX_T]
        nt = len(chunk)
        ptrs = [item[0] for item in chunk]
        sizes = [item[1] for item in chunk]
        if nt < _MAX_T:
            pad = _MAX_T - nt
            ptrs += [ptrs[0]] * pad
            sizes += [0] * pad
        grid = ((max(sizes) + _BLOCK - 1) // _BLOCK, nt)
        fast = _hooks_clear()
        if fast:
            for size in sizes:
                if size == 1 or (size & 15):
                    fast = False
                    break
        if fast:
            for t, _ in chunk:
                if t.data_ptr() & 15:
                    fast = False
                    break
        if fast:
            args = tuple(ptrs) + tuple(sizes) + (nt, _BLOCK)
            key = (1, tuple(t.dtype for t in ptrs), nt)
            entry = _LAUNCH.get(key)
            if entry is not None:
                _replay(entry, grid[0], grid[1], args)
                return
            _bind(_frac_multi_kernel[grid](*args), key)
            return
        _frac_multi_kernel[grid](*ptrs, *sizes, nt, _BLOCK)


def foreach_frac_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError(_ERR_LIST)
    if len(self) == 0:
        raise RuntimeError(_ERR_EMPTY)

    dense = []
    largest = 0
    for t in self:
        if not isinstance(t, _Tensor):
            raise TypeError(_ERR_LIST)
        if not t.dtype.is_floating_point:
            raise RuntimeError(
                "\"frac\" not implemented for '%s'" % str(t.dtype).replace("torch.", "")
            )
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous() or _flat_addressable(t):
            dense.append((t, n))
            if n > largest:
                largest = n
        else:
            _launch_strided(t, n)

    count = len(dense)
    if count == 0:
        return None
    if count > 1 and largest <= _FUSE_MAX_N:
        _launch_multi(dense)
    else:
        for t, n in dense:
            _launch_flat(t, n)
    return None
