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

"""Triton implementation of aten::_foreach_add_.Scalar (in-place).

For every tensor in the input sequence, in place:  T[i] <- T[i] + scalar

Storage identity, shape, stride, dtype are preserved and the call returns None.

Design
------
* Homogeneous contiguous lists (the common case and the whole benchmark) are
  served by ONE fused multi-tensor launch; every program block maps to exactly
  one tensor through in-kernel cumulative block counts.
* Repeat launches reuse the compiled kernel handle: the runner is keyed by
  device, dtype, grid, per-slot size and per-slot 16-byte pointer alignment
  (the full Triton specialization state), so it is always the variant the JIT
  would have selected, and it forwards the same argument list the JIT uses
  (runtime args followed by the constexpr block size).  When no launch hook is
  registered (HookChain.calls empty) the per-call launch-metadata build is
  skipped entirely.  A one-time self check per dtype validates the direct path
  numerically; any problem permanently falls back to ``kernel[grid](...)``.
  Every call performs exactly one launch: on a runner-cache miss the JIT launch
  itself performs the update and only hands back the compiled kernel to cache.
* Strided views are updated in place through a rank-collapsed stride-aware
  kernel (no materialization, storage identity preserved).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_NUM_WARPS = 4
_SBLOCK = 1024

try:
    from triton import knobs as _knobs
    from triton.runtime import driver as _driver

    _DIRECT_IMPORTED = True
except Exception:  # pragma: no cover - older/odd builds
    _DIRECT_IMPORTED = False

_RUNNERS = {}
_CHECKED = set()
_FAST = True


@triton.jit
def _add_flat(x_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(x_ptr + offs, v + s, mask=mask)


@triton.jit
def _add4(x0, n0, x1, n1, x2, n2, x3, n3, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c0 = tl.cdiv(n0, BLOCK)
    c1 = c0 + tl.cdiv(n1, BLOCK)
    c2 = c1 + tl.cdiv(n2, BLOCK)
    if pid < c0:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n0
        v = tl.load(x0 + offs, mask=mask, other=0, eviction_policy="evict_first")
        tl.store(x0 + offs, v + s, mask=mask, cache_modifier=".cs")
    elif pid < c1:
        offs = (pid - c0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n1
        v = tl.load(x1 + offs, mask=mask, other=0, eviction_policy="evict_first")
        tl.store(x1 + offs, v + s, mask=mask, cache_modifier=".cs")
    elif pid < c2:
        offs = (pid - c1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n2
        v = tl.load(x2 + offs, mask=mask, other=0, eviction_policy="evict_first")
        tl.store(x2 + offs, v + s, mask=mask, cache_modifier=".cs")
    else:
        offs = (pid - c2) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n3
        v = tl.load(x3 + offs, mask=mask, other=0, eviction_policy="evict_first")
        tl.store(x3 + offs, v + s, mask=mask, cache_modifier=".cs")


@triton.jit
def _add4_nomask(x0, n0, x1, n1, x2, n2, x3, n3, s, BLOCK: tl.constexpr):
    """Fused add for slots whose sizes are all multiples of BLOCK: no masking."""
    pid = tl.program_id(0)
    c0 = n0 // BLOCK
    c1 = c0 + n1 // BLOCK
    c2 = c1 + n2 // BLOCK
    if pid < c0:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x0 + offs, eviction_policy="evict_first")
        tl.store(x0 + offs, v + s, cache_modifier=".cs")
    elif pid < c1:
        offs = (pid - c0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x1 + offs, eviction_policy="evict_first")
        tl.store(x1 + offs, v + s, cache_modifier=".cs")
    elif pid < c2:
        offs = (pid - c1) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x2 + offs, eviction_policy="evict_first")
        tl.store(x2 + offs, v + s, cache_modifier=".cs")
    else:
        offs = (pid - c2) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x3 + offs, eviction_policy="evict_first")
        tl.store(x3 + offs, v + s, cache_modifier=".cs")


@triton.jit
def _add_strided(
    x_ptr,
    n,
    s,
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
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    r = offs
    i7 = r % s7
    r = r // s7
    i6 = r % s6
    r = r // s6
    i5 = r % s5
    r = r // s5
    i4 = r % s4
    r = r // s4
    i3 = r % s3
    r = r // s3
    i2 = r % s2
    r = r // s2
    i1 = r % s1
    i0 = r // s1

    off = i0 * t0 + i1 * t1 + i2 * t2 + i3 * t3 + i4 * t4 + i5 * t5 + i6 * t6 + i7 * t7

    v = tl.load(x_ptr + off, mask=mask, other=0)
    tl.store(x_ptr + off, v + s, mask=mask)


def _collapse(shape, strides):
    """Collapse to (size, stride) pairs, outer -> inner. Size-1 dims dropped."""
    dims = []
    for d in range(len(shape) - 1, -1, -1):
        sz = shape[d]
        if sz == 1:
            continue
        st = strides[d]
        if dims and st == dims[-1][1] * dims[-1][0]:
            dims[-1] = (dims[-1][0] * sz, st)
        else:
            dims.append((sz, st))
    dims.reverse()
    return dims


def _slow(tensors, scalar):
    """Heterogeneous / noncontiguous fallback: one launch per tensor."""
    f_scalar = None
    i_scalar = None
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        if t.dtype.is_floating_point:
            if f_scalar is None:
                f_scalar = float(scalar)
            s = f_scalar
        else:
            if i_scalar is None:
                i_scalar = int(scalar)
            s = i_scalar

        if t.is_contiguous():
            _add_flat[((n + _BLOCK - 1) // _BLOCK,)](
                t, n, s, BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
        else:
            dims = _collapse(t.shape, t.stride())
            if not dims:
                dims = [(1, 0)]
            if len(dims) > 8:
                raise NotImplementedError("rank after collapse exceeds 8")
            sizes = [1] * 8
            strides_ = [0] * 8
            k = 8 - len(dims)
            for j, (sz, st) in enumerate(dims):
                sizes[k + j] = sz
                strides_[k + j] = st
            _add_strided[((n + _SBLOCK - 1) // _SBLOCK,)](
                t,
                n,
                s,
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3],
                sizes[4],
                sizes[5],
                sizes[6],
                sizes[7],
                strides_[0],
                strides_[1],
                strides_[2],
                strides_[3],
                strides_[4],
                strides_[5],
                strides_[6],
                strides_[7],
                BLOCK=_SBLOCK,
                num_warps=4,
            )


def _make_runner(ck, g0, block, dev):
    """Bind every per-launch constant outside the hot path."""
    g3 = (g0, 1, 1)
    run = ck.run
    fn = ck.function
    pm = ck.packed_metadata
    get_stream = _driver.active.get_current_stream
    enter = _knobs.runtime.launch_enter_hook
    exit_ = _knobs.runtime.launch_exit_hook
    try:
        hooks_off = not enter.calls and not exit_.calls
    except Exception:
        hooks_off = False

    if hooks_off:
        # No launch hook is attached: skip building launch metadata per call.
        def runner(x0, n0, x1, n1, x2, n2, x3, n3, s):
            run(
                g0,
                1,
                1,
                get_stream(dev),
                fn,
                pm,
                None,
                enter,
                exit_,
                x0,
                n0,
                x1,
                n1,
                x2,
                n2,
                x3,
                n3,
                s,
                block,
            )

        return runner

    lm_fn = ck.launch_metadata

    def runner(x0, n0, x1, n1, x2, n2, x3, n3, s):
        stream = get_stream(dev)
        lm = lm_fn(g3, stream, x0, n0, x1, n1, x2, n2, x3, n3, s)
        run(
            g0,
            1,
            1,
            stream,
            fn,
            pm,
            lm,
            enter,
            exit_,
            x0,
            n0,
            x1,
            n1,
            x2,
            n2,
            x3,
            n3,
            s,
            block,
        )

    return runner


def _selfcheck(ck, dtype, s, dev, nomask):
    """One-time proof that the direct launcher executes this kernel."""
    try:
        n = _BLOCK if nomask else 16
        scratch = torch.zeros(n, dtype=dtype, device="cuda")
        runner = _make_runner(ck, 1, _BLOCK, dev)
        runner(scratch, n, scratch, 0, scratch, 0, scratch, 0, s)
        torch.cuda.synchronize()
        got = float(scratch[0].item())
        want = float(torch.tensor(s, dtype=dtype).item())
        tol = 1e-4 * max(1.0, abs(want))
        if abs(got - want) > tol:
            return False
        if nomask:
            # The maskless variant must cover the whole block.
            if abs(float(scratch.min().item()) - want) > tol:
                return False
            if abs(float(scratch.max().item()) - want) > tol:
                return False
        return True
    except Exception:
        return False


def _launch4(x0, n0, x1, n1, x2, n2, x3, n3, s, g0, dtype):
    """One fused launch for up to four tensors; exactly one launch per call."""
    global _FAST
    if not _FAST:
        _add4[(g0,)](
            x0,
            n0,
            x1,
            n1,
            x2,
            n2,
            x3,
            n3,
            s,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        return
    dev = x0.get_device()
    nomask = ((n0 | n1 | n2 | n3) & (_BLOCK - 1)) == 0
    key = (
        dev,
        dtype,
        g0,
        nomask,
        n0,
        n1,
        n2,
        n3,
        x0.data_ptr() & 15,
        x1.data_ptr() & 15,
        x2.data_ptr() & 15,
        x3.data_ptr() & 15,
    )
    runner = _RUNNERS.get(key)
    if runner is not None:
        runner(x0, n0, x1, n1, x2, n2, x3, n3, s)
        return
    # Cold entry for this specialization: this JIT call performs the group's
    # update, and only its returned compiled kernel is cached for later calls.
    kfn = _add4_nomask if nomask else _add4
    kern = kfn.run(
        x0,
        n0,
        x1,
        n1,
        x2,
        n2,
        x3,
        n3,
        s,
        grid=(g0,),
        warmup=False,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    if kern is None or not _FAST:
        return
    try:
        if dtype not in _CHECKED:
            _CHECKED.add(dtype)
            if not _selfcheck(kern, dtype, s, dev, nomask):
                _FAST = False
                return
        runner = _make_runner(kern, g0, _BLOCK, dev)
    except Exception:
        _FAST = False
        return
    if len(_RUNNERS) > 512:
        _RUNNERS.clear()
    _RUNNERS[key] = runner


def foreach_add_scalar_(*args, **kwargs):
    if args:
        tensors = args[0]
        scalar = args[1] if len(args) > 1 else kwargs.get("scalar", 0.0)
    else:
        tensors = kwargs.get("self", kwargs.get("tensors"))
        scalar = kwargs.get("scalar")

    if tensors is None:
        return None
    m = len(tensors)
    if m == 0:
        return None
    if type(scalar) is not float:
        # Scalar overload or a 0-dim tensor / numpy scalar.
        if torch.is_tensor(scalar):
            scalar = scalar.item()
        elif type(scalar) is not int:
            scalar = float(scalar)

    t0 = tensors[0]
    dt = t0.dtype
    fl = dt.is_floating_point
    if fl:
        s = scalar if type(scalar) is float else float(scalar)
    else:
        s = scalar if type(scalar) is int else int(scalar)

    if not _DIRECT_IMPORTED:
        _slow(tensors, scalar)
        return None

    bm = _BLOCK - 1

    # Common case: four homogeneous contiguous tensors -> one fused launch.
    if m == 4 and fl:
        x0 = tensors[0]
        x1 = tensors[1]
        x2 = tensors[2]
        x3 = tensors[3]
        if (
            x1.dtype is dt
            and x2.dtype is dt
            and x3.dtype is dt
            and x0.is_contiguous()
            and x1.is_contiguous()
            and x2.is_contiguous()
            and x3.is_contiguous()
        ):
            n0 = x0.numel()
            n1 = x1.numel()
            n2 = x2.numel()
            n3 = x3.numel()
            g0 = (
                (n0 + bm) // _BLOCK
                + (n1 + bm) // _BLOCK
                + (n2 + bm) // _BLOCK
                + (n3 + bm) // _BLOCK
            )
            if g0:
                _launch4(x0, n0, x1, n1, x2, n2, x3, n3, s, g0, dt)
            return None
        _slow(tensors, scalar)
        return None

    # Fast path: every tensor contiguous with the same dtype -> fused launches.
    ns = []
    for t in tensors:
        if not t.is_contiguous() or t.dtype is not dt:
            _slow(tensors, scalar)
            return None
        ns.append(t.numel())

    j = 0
    while j < m:
        g = m - j
        x0 = tensors[j]
        n0 = ns[j]
        if g >= 4:
            x1 = tensors[j + 1]
            x2 = tensors[j + 2]
            x3 = tensors[j + 3]
            n1 = ns[j + 1]
            n2 = ns[j + 2]
            n3 = ns[j + 3]
            j += 4
        else:
            x1 = tensors[j + 1] if g > 1 else x0
            x2 = tensors[j + 2] if g > 2 else x0
            x3 = x0
            n1 = ns[j + 1] if g > 1 else 0
            n2 = ns[j + 2] if g > 2 else 0
            n3 = 0
            j += 3
        g0 = (
            (n0 + bm) // _BLOCK
            + (n1 + bm) // _BLOCK
            + (n2 + bm) // _BLOCK
            + (n3 + bm) // _BLOCK
        )
        if g0:
            _launch4(x0, n0, x1, n1, x2, n2, x3, n3, s, g0, dt)
    return None
