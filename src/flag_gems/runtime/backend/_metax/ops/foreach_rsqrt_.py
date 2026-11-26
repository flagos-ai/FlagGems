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

"""aten::_foreach_rsqrt_(Tensor(a!)[] self) -> () in Triton (in place).

Architecture:
  * flat 1D traversal whenever a tensor is non-overlapping-and-dense, so
    transposes / permutes / offset views need no stride math at all
  * a rank-bounded div/mod kernel for genuinely gappy views
  * same-dtype, same-numel groups are packed into ONE launch with up to sixteen
    pointer arguments (no host->device metadata copy), which removes per-launch
    host cost for lists of tiny tensors and the inter-kernel gaps for lists of
    large ones
  * launches go through a cached compiled-kernel handle instead of re-running
    Triton's per-call argument binding/specialization. The framework's launch
    hooks are forwarded exactly as the standard path forwards them, and any
    unsupported case (single element, misaligned pointer, foreign device, kernel
    scratch, or a toolchain without that handle) falls back to kernel[grid](...)
  * integer / bool inputs are rejected on the host, exactly like ATen's
    in-place promotion refusal, before any launch
  * fp16 / bf16 use fp32 opmath with the hardware rsqrt, fp32 the same,
    fp64 keeps the precise 1/sqrt
"""

import torch
import triton
import triton.language as tl

try:  # runtime knobs (launch hooks) live here on Triton 3.x
    _RT = triton.knobs.runtime
except Exception:  # noqa: BLE001
    _RT = None

_MAX_RANK = 4
_BLOCK = 1024
_PACK = 16


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _rsqrt_tile(x_ptr, offs, mask):
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        y = tl.math.rsqrt(x.to(tl.float32))
    elif x.dtype == tl.float64:
        y = 1.0 / tl.sqrt(x)
    else:
        y = tl.math.rsqrt(x)
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _rsqrt_flat(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _rsqrt_tile(x_ptr, offs, offs < n)


@triton.jit
def _rsqrt_pack16(
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
):
    """One launch for up to sixteen same-dtype tensors of equal numel.

    Axis 0 selects the tensor, axis 1 the chunk inside it. ``n`` is shared
    because the pack is built from tensors with identical element counts.
    """
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if t == 0:
        _rsqrt_tile(x0, offs, mask)
    elif t == 1:
        _rsqrt_tile(x1, offs, mask)
    elif t == 2:
        _rsqrt_tile(x2, offs, mask)
    elif t == 3:
        _rsqrt_tile(x3, offs, mask)
    elif t == 4:
        _rsqrt_tile(x4, offs, mask)
    elif t == 5:
        _rsqrt_tile(x5, offs, mask)
    elif t == 6:
        _rsqrt_tile(x6, offs, mask)
    elif t == 7:
        _rsqrt_tile(x7, offs, mask)
    elif t == 8:
        _rsqrt_tile(x8, offs, mask)
    elif t == 9:
        _rsqrt_tile(x9, offs, mask)
    elif t == 10:
        _rsqrt_tile(x10, offs, mask)
    elif t == 11:
        _rsqrt_tile(x11, offs, mask)
    elif t == 12:
        _rsqrt_tile(x12, offs, mask)
    elif t == 13:
        _rsqrt_tile(x13, offs, mask)
    elif t == 14:
        _rsqrt_tile(x14, offs, mask)
    else:
        _rsqrt_tile(x15, offs, mask)


@triton.jit
def _rsqrt_strided(
    x_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    s0,
    s1,
    s2,
    s3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    i0 = idx % d0
    r = idx // d0
    i1 = r % d1
    r = r // d1
    i2 = r % d2
    r = r // d2
    i3 = r % d3
    _rsqrt_tile(x_ptr, i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3, mask)


# ---------------------------------------------------------------------------
# low-overhead launch path
# ---------------------------------------------------------------------------
#
# Triton's JITFunction.run rebinds and re-specializes every argument on every
# call, which costs more than the device work once the tensors are small. The
# compiled kernel already exposes its launcher, so the same launch can be issued
# from a cached handle. The handle is built through the public warmup() entry
# point and is only used when every specialization assumption it was compiled
# with still holds; anything else uses the ordinary launch.

_LAUNCH_CACHE = {}
_FAST_ENABLED = True
_STREAM_FNS = None


def _cur_dev_stream():
    global _STREAM_FNS
    fns = _STREAM_FNS
    if fns is None:
        from triton.runtime import driver as _d

        active = _d.active
        fns = (active.get_current_device, active.get_current_stream)
        _STREAM_FNS = fns
    dev = fns[0]()
    return dev, fns[1](dev)


def _build_entry(kernel, warm_args):
    """Cached (launcher, function, packed_metadata, kernel) quadruple, or None."""
    global _FAST_ENABLED
    if not _FAST_ENABLED:
        return None
    try:
        ck = kernel.warmup(*warm_args, BLOCK=_BLOCK, grid=(1,), num_warps=4)
        ck._init_handles()
        launcher = ck.run
        if launcher.global_scratch_size or launcher.profile_scratch_size:
            _FAST_ENABLED = False
            return None
        return (launcher.launch, ck.function, ck.packed_metadata, ck)
    except Exception:  # noqa: BLE001
        _FAST_ENABLED = False
        return None


def _fast_launch(ent, g0, g1, stream, args):
    enter = _RT.launch_enter_hook if _RT is not None else None
    leave = _RT.launch_exit_hook if _RT is not None else None
    metadata = None
    if enter is not None or leave is not None:
        metadata = ent[3].launch_metadata((g0, g1, 1), stream, *args)
    ent[0](g0, g1, 1, stream, ent[1], None, None, ent[2], metadata, enter, leave, *args)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _ordered_by_stride(t):
    """(size, stride) pairs, size-1 dims dropped, ascending stride, no sort()."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    ordered = []
    for d in dims:
        pos = len(ordered)
        for i, o in enumerate(ordered):
            if d[1] < o[1]:
                pos = i
                break
        ordered.insert(pos, d)
    return ordered


def _is_flat_addressable(t):
    """True when [data_ptr, data_ptr+numel) covers exactly this tensor."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for sz, st in _ordered_by_stride(t):
        if st != expected:
            return False
        expected *= sz
    return True


def _canonical_dims(t):
    """Ordered (size, stride) pairs with adjacent mergeable dims collapsed."""
    dims = _ordered_by_stride(t)
    if not dims:
        return [(1, 0)]
    out = [[dims[0][0], dims[0][1]]]
    for sz, st in dims[1:]:
        if out[-1][1] == st * sz:
            out[-1][0] *= sz
        else:
            out.append([sz, st])
    return out


def _launch_flat(t, n):
    grid0 = triton.cdiv(n, _BLOCK)
    if n > 1 and (t.data_ptr() & 15) == 0:
        dev, stream = _cur_dev_stream()
        if t.get_device() == dev:
            key = ("f", t.dtype, n & 15 == 0, dev)
            ent = _LAUNCH_CACHE.get(key)
            if ent is None:
                ent = _build_entry(_rsqrt_flat, (t, n))
                _LAUNCH_CACHE[key] = ent
            if ent is not None:
                _fast_launch(ent, grid0, 1, stream, (t, n, _BLOCK))
                return
    _rsqrt_flat[(grid0,)](t, n, BLOCK=_BLOCK, num_warps=4)


def _launch_gappy(t):
    if t.dim() > _MAX_RANK:
        for i in range(t.shape[0]):
            _launch_gappy(t.select(0, i))
        return
    n = t.numel()
    if n == 0:
        return
    dims = _canonical_dims(t)
    dims = dims + [(1, 0)] * (_MAX_RANK - len(dims))
    _rsqrt_strided[(triton.cdiv(n, _BLOCK),)](
        t,
        n,
        dims[0][0],
        dims[1][0],
        dims[2][0],
        dims[3][0],
        dims[0][1],
        dims[1][1],
        dims[2][1],
        dims[3][1],
        BLOCK=_BLOCK,
        num_warps=4,
    )


def _launch_pack(ts, n):
    # A packed launch writes the slots in parallel, so duplicate list entries
    # (which ATen applies once per position, in order) must stay sequential.
    # Axis 1 of the packed grid is the chunk count, capped by the grid limit.
    chunks = triton.cdiv(n, _BLOCK)
    if chunks > 65535:
        for t in ts:
            _launch_flat(t, n)
        return
    ptrs = []
    usable = n > 1
    dev = stream = None
    if usable:
        dev, stream = _cur_dev_stream()
    if usable:
        for t in ts:
            p = t.data_ptr()
            if (p & 15) or t.get_device() != dev:
                usable = False
                break
            ptrs.append(p)
    if not usable or len(set(ptrs)) != len(ptrs):
        for t in ts:
            _launch_flat(t, n)
        return

    for start in range(0, len(ts), _PACK):
        part = ts[start : start + _PACK]
        k = len(part)
        if k == 1:
            _launch_flat(part[0], n)
            continue
        args = part + [part[0]] * (_PACK - k)
        key = ("p", part[0].dtype, n & 15 == 0, dev)
        ent = _LAUNCH_CACHE.get(key)
        if ent is None:
            ent = _build_entry(_rsqrt_pack16, list(args) + [n])
            _LAUNCH_CACHE[key] = ent
        if ent is None:
            for t in part:
                _launch_flat(t, n)
            continue
        _fast_launch(ent, k, chunks, stream, list(args) + [n, _BLOCK])


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_rsqrt_(self):
    tensors = self
    if tensors is None:
        tensors = []
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    n_tensors = len(tensors)
    if n_tensors == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt.is_complex:
            raise NotImplementedError("foreach rsqrt_: complex not supported")
        if not dt.is_floating_point:
            raise RuntimeError(
                "result type Float32 can't be cast to the desired output type "
                + str(dt).replace("torch.", "").capitalize()
            )

    groups = {}
    order = []
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        if _is_flat_addressable(t):
            if n_tensors > 1:
                key = (t.dtype, n)
                g = groups.get(key)
                if g is None:
                    groups[key] = [t]
                    order.append(key)
                else:
                    g.append(t)
                continue
            _launch_flat(t, n)
            continue
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to "
                "tensor refers to a single memory location. Please clone() the "
                "tensor before performing the operation."
            )
        _launch_gappy(t)

    for key in order:
        g = groups[key]
        if len(g) == 1:
            _launch_flat(g[0], key[1])
        else:
            _launch_pack(g, key[1])
    return None
