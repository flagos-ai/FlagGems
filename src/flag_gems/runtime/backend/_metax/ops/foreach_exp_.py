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

"""Triton implementation of the in-place foreach exponential.

ATen schema: ``aten::_foreach_exp_(Tensor(a!)[] self) -> ()``

Every tensor in ``self`` is exponentiated element-wise, in place, keeping its
storage, dtype and strides.  No value is returned.

Host cost is the binding constraint for short lists, so the hot path is kept
minimal: one pass over the list, no device metadata buffer, no host->device
copy, no per-tensor temporary objects, and a cached pre-bound launcher for the
compiled kernel.  Tensors sharing (device, dtype, numel) are fused into a single
2-D-grid launch.  Everything else (non-contiguous, complex, mixed numel,
zero-numel, aliasing) takes the general path.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 4
_FUSE_MAX = 16
_I32_MAX = 2**31 - 1

try:  # tri-state internal-overlap probe (0 no, 1 yes, 2 unknown)
    _has_internal_overlap = torch._debug_has_internal_overlap
except AttributeError:  # pragma: no cover - very old torch
    _has_internal_overlap = None

# ``JITFunction.run`` prepends two distributed-runtime arguments to the launch
# in flagtree's lite mode; the pre-bound ``CompiledKernel`` runner does not, so
# the cached-launcher fast path is disabled when that mode is active.
_FAST_LAUNCH = True
try:
    from triton.runtime.jit import DistributedRtContext as _DistCtx
except Exception:  # pragma: no cover - build without distributed support
    _DistCtx = None
if _DistCtx is not None:  # pragma: no cover - depends on the target build
    try:
        _FAST_LAUNCH = not bool(_DistCtx().is_lite_mode)
    except Exception:
        _FAST_LAUNCH = False

_DTYPE_NAME = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
}

# Pre-bound CompiledKernel runners, keyed by everything that shapes the binary:
# kind, device, dtype, pointer alignment classes, element count, tensor count.
_LAUNCHERS = {}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _exp_flat_kernel(
    x_ptr, n, BLOCK: tl.constexpr, IDX64: tl.constexpr, ACC32: tl.constexpr
):
    pid = tl.program_id(0)
    if IDX64:
        idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    if ACC32:
        y = tl.exp(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.exp(x)
    tl.store(x_ptr + idx, y, mask=mask)


@triton.jit
def _exp_fuse_flat_kernel(
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
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
    ACC32: tl.constexpr,
):
    """One launch for ``K`` tensors that share dtype and element count.

    Axis 0 selects the tile (it can exceed the 65535 limit of the inner grid
    dimensions) and axis 1 selects the tensor, so consecutive CTAs stream
    through consecutive tiles of the same tensor.
    """
    tile = tl.program_id(0)
    slot = tl.program_id(1)
    if IDX64:
        idx = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        idx = tile * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    ptr = p0
    if K > 1:
        if slot == 1:
            ptr = p1
        elif slot == 2:
            ptr = p2
        elif slot == 3:
            ptr = p3
        elif slot == 4:
            ptr = p4
        elif slot == 5:
            ptr = p5
        elif slot == 6:
            ptr = p6
        elif slot == 7:
            ptr = p7
        elif slot == 8:
            ptr = p8
        elif slot == 9:
            ptr = p9
        elif slot == 10:
            ptr = p10
        elif slot == 11:
            ptr = p11
        elif slot == 12:
            ptr = p12
        elif slot == 13:
            ptr = p13
        elif slot == 14:
            ptr = p14
        elif slot == 15:
            ptr = p15
    x = tl.load(ptr + idx, mask=mask, other=0.0)
    if ACC32:
        y = tl.exp(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.exp(x)
    tl.store(ptr + idx, y, mask=mask)


@triton.jit
def _exp_strided_kernel(
    x_ptr,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    st0,
    st1,
    st2,
    st3,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
    ACC32: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = idx * 0
    if RANK == 1:
        off = rem * st0
    if RANK == 2:
        i1 = rem % sh1
        rem = rem // sh1
        off = i1 * st1 + rem * st0
    if RANK == 3:
        i2 = rem % sh2
        rem = rem // sh2
        i1 = rem % sh1
        rem = rem // sh1
        off = i2 * st2 + i1 * st1 + rem * st0
    if RANK == 4:
        i3 = rem % sh3
        rem = rem // sh3
        i2 = rem % sh2
        rem = rem // sh2
        i1 = rem % sh1
        rem = rem // sh1
        off = i3 * st3 + i2 * st2 + i1 * st1 + rem * st0
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    if ACC32:
        y = tl.exp(x.to(tl.float32)).to(x.dtype)
    else:
        y = tl.exp(x)
    tl.store(x_ptr + off, y, mask=mask)


@triton.jit
def _exp_cplx_flat_kernel(
    x_ptr, n, BLOCK: tl.constexpr, IDX64: tl.constexpr, ACC32: tl.constexpr
):
    pid = tl.program_id(0)
    if IDX64:
        idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    re = tl.load(x_ptr + 2 * idx, mask=mask, other=0.0)
    im = tl.load(x_ptr + 2 * idx + 1, mask=mask, other=0.0)
    if ACC32:
        e = tl.exp(re.to(tl.float32))
        out_re = (e * tl.cos(im.to(tl.float32))).to(x_ptr.dtype.element_ty)
        out_im = (e * tl.sin(im.to(tl.float32))).to(x_ptr.dtype.element_ty)
    else:
        e = tl.exp(re)
        out_re = e * tl.cos(im)
        out_im = e * tl.sin(im)
    tl.store(x_ptr + 2 * idx, out_re, mask=mask)
    tl.store(x_ptr + 2 * idx + 1, out_im, mask=mask)


@triton.jit
def _exp_cplx_strided_kernel(
    x_ptr,
    n,
    sh0,
    sh1,
    sh2,
    sh3,
    st0,
    st1,
    st2,
    st3,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
    ACC32: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = idx * 0
    if RANK == 1:
        off = rem * st0
    if RANK == 2:
        i1 = rem % sh1
        rem = rem // sh1
        off = i1 * st1 + rem * st0
    if RANK == 3:
        i2 = rem % sh2
        rem = rem // sh2
        i1 = rem % sh1
        rem = rem // sh1
        off = i2 * st2 + i1 * st1 + rem * st0
    if RANK == 4:
        i3 = rem % sh3
        rem = rem // sh3
        i2 = rem % sh2
        rem = rem // sh2
        i1 = rem % sh1
        rem = rem // sh1
        off = i3 * st3 + i2 * st2 + i1 * st1 + rem * st0
    re = tl.load(x_ptr + off, mask=mask, other=0.0)
    im = tl.load(x_ptr + off + 1, mask=mask, other=0.0)
    if ACC32:
        e = tl.exp(re.to(tl.float32))
        out_re = (e * tl.cos(im.to(tl.float32))).to(x_ptr.dtype.element_ty)
        out_im = (e * tl.sin(im.to(tl.float32))).to(x_ptr.dtype.element_ty)
    else:
        e = tl.exp(re)
        out_re = e * tl.cos(im)
        out_im = e * tl.sin(im)
    tl.store(x_ptr + off, out_re, mask=mask)
    tl.store(x_ptr + off + 1, out_im, mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _is_flat_addressable(t):
    """True when flat ``[0, numel)`` indexing visits every element exactly once."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted((s, z) for s, z in zip(t.stride(), t.shape) if z != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _collapse(t):
    """Drop size-1 dims and merge adjacent dims that stay contiguous.

    Dims ``(P, T)`` (outer) and ``(n, s)`` (inner) describe the same element set
    as ``(P*n, s)`` exactly when ``T == n*s``; the surviving stride is the inner
    one, not the outer one.
    """
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for sz, st in dims:
        if out and out[-1][1] == st * sz:
            psz, _pst = out[-1]
            out[-1] = (psz * sz, st)
        else:
            out.append((sz, st))
    return out


def _launch_flat(t, n):
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    idx64 = n > _I32_MAX
    acc32 = t.dtype is torch.float16 or t.dtype is torch.bfloat16
    args = (t, n, _BLOCK, idx64, acc32)
    if _FAST_LAUNCH:
        key = (0, t.device, t.dtype, t.data_ptr() & 15, n, 1)
        runner = _LAUNCHERS.get(key)
        if runner is not None:
            runner(*args)
            return
        kernel = _exp_flat_kernel[grid](*args, num_warps=_NUM_WARPS)
        if kernel is not None:
            if len(_LAUNCHERS) > 1024:
                _LAUNCHERS.clear()
            # CompiledKernel.__getitem__ needs the full 3-D grid tuple.
            _LAUNCHERS[key] = kernel[(grid[0], 1, 1)]
        return
    _exp_flat_kernel[grid](*args, num_warps=_NUM_WARPS)


def _launch_fused_flat(tensors, n, ptrs=None):
    m = len(tensors)
    dtype = tensors[0].dtype
    if ptrs is None:
        ptrs = [t.data_ptr() for t in tensors]
    grid = ((n + _BLOCK - 1) // _BLOCK, m)
    idx64 = n > _I32_MAX
    acc32 = dtype is torch.float16 or dtype is torch.bfloat16
    args = (
        tuple(tensors) + (tensors[0],) * (_FUSE_MAX - m) + (n, m, _BLOCK, idx64, acc32)
    )
    if _FAST_LAUNCH:
        key = (1, tensors[0].device, dtype, tuple(p & 15 for p in ptrs), n, m)
        runner = _LAUNCHERS.get(key)
        if runner is not None:
            runner(*args)
            return
        kernel = _exp_fuse_flat_kernel[grid](*args, num_warps=_NUM_WARPS)
        if kernel is not None:
            if len(_LAUNCHERS) > 1024:
                _LAUNCHERS.clear()
            # CompiledKernel.__getitem__ needs the full 3-D grid tuple.
            _LAUNCHERS[key] = kernel[(grid[0], m, 1)]
        return
    _exp_fuse_flat_kernel[grid](*args, num_warps=_NUM_WARPS)


def _launch_strided(t, dims, n):
    rank = len(dims)
    padded = dims + [(1, 0)] * (_MAX_RANK - rank)
    _exp_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t,
        n,
        padded[0][0],
        padded[1][0],
        padded[2][0],
        padded[3][0],
        padded[0][1],
        padded[1][1],
        padded[2][1],
        padded[3][1],
        rank,
        _BLOCK,
        n > _I32_MAX,
        t.dtype is torch.float16 or t.dtype is torch.bfloat16,
        num_warps=_NUM_WARPS,
    )


def _launch_cplx_flat(real_view, n):
    _exp_cplx_flat_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        real_view,
        n,
        _BLOCK,
        n > _I32_MAX,
        real_view.dtype is torch.float16,
        num_warps=_NUM_WARPS,
    )


def _launch_cplx_strided(real_view, dims, n):
    rank = len(dims)
    padded = dims + [(1, 0)] * (_MAX_RANK - rank)
    _exp_cplx_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        real_view,
        n,
        padded[0][0],
        padded[1][0],
        padded[2][0],
        padded[3][0],
        padded[0][1],
        padded[1][1],
        padded[2][1],
        padded[3][1],
        rank,
        _BLOCK,
        n > _I32_MAX,
        real_view.dtype is torch.float16,
        num_warps=_NUM_WARPS,
    )


# kind tags returned by _prepare
_FLAT = 0
_STRIDED = 1
_CPLX_FLAT = 2
_CPLX_STRIDED = 3


def _prepare(t):
    """Validate one entry and classify it.  ``None`` means nothing to launch."""
    dtype = t.dtype
    if dtype.is_complex:
        if dtype is torch.complex32:
            raise RuntimeError("\"foreach\" not implemented for 'complex32'")
    elif not dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            + _DTYPE_NAME.get(dtype, str(dtype).replace("torch.", "").capitalize())
        )

    n = t.numel()
    if n == 0:
        return None

    if _has_internal_overlap is not None and _has_internal_overlap(t) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )

    flat = _is_flat_addressable(t)
    if dtype.is_complex:
        return (t.real, n, _CPLX_FLAT if flat else _CPLX_STRIDED)
    return (t, n, _FLAT if flat else _STRIDED)


def _launch_one(t, n, kind):
    if kind == _FLAT:
        _launch_flat(t, n)
    elif kind == _STRIDED:
        dims = _collapse(t)
        if len(dims) > _MAX_RANK:
            for i in range(t.shape[0]):
                _exp_inplace(t[i])
            return
        _launch_strided(t, dims, n)
    elif kind == _CPLX_FLAT:
        _launch_cplx_flat(t, n)
    else:
        dims = _collapse(t)
        if len(dims) > _MAX_RANK:
            for i in range(t.shape[0]):
                _exp_inplace(t[i])
            return
        _launch_cplx_strided(t, dims, n)


def _exp_inplace(t):
    item = _prepare(t)
    if item is not None:
        _launch_one(*item)


def _disjoint(items):
    """Whether no two entries of the list share storage (byte-range check)."""
    spans = []
    for t, n, _kind in items:
        base = t.data_ptr()
        spans.append((base, base + n * t.element_size()))
    spans = sorted(spans)
    for lo, hi in zip(spans, spans[1:]):
        if hi[0] < lo[1]:
            return False
    return True


def _run_fast(tensors, n, dtype):
    """All entries are same-device, same-dtype, contiguous, non-empty, equal numel."""
    m = len(tensors)
    if m == 1:
        _launch_flat(tensors[0], n)
        return
    ptrs = [t.data_ptr() for t in tensors]
    span = n * tensors[0].element_size()
    ordered = sorted(ptrs)
    for a, b in zip(ordered, ordered[1:]):
        if b - a < span:
            # Aliasing entries must be applied one after another, like ATen.
            _run_general(tensors)
            return
    for i in range(0, m, _FUSE_MAX):
        _launch_fused_flat(tensors[i : i + _FUSE_MAX], n, ptrs[i : i + _FUSE_MAX])


def _run_general(self):
    items = []
    for t in self:
        item = _prepare(t)
        if item is not None:
            items.append(item)

    if len(items) > 1 and _disjoint(items):
        buckets = {}
        order = []
        for t, n, kind in items:
            if kind == _FLAT:
                key = (t.device, t.dtype, n)
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [t]
                    order.append(key)
                else:
                    bucket.append(t)
            else:
                _launch_one(t, n, kind)
        for key in order:
            tensors = buckets[key]
            n = key[2]
            if len(tensors) == 1:
                _launch_flat(tensors[0], n)
            else:
                for i in range(0, len(tensors), _FUSE_MAX):
                    _launch_fused_flat(tensors[i : i + _FUSE_MAX], n)
    else:
        for item in items:
            _launch_one(*item)


def foreach_exp_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if not self:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Single pass over the list: validate, and detect the homogeneous
    # contiguous case that can be fused with no per-tensor bookkeeping.
    dtype0 = None
    n0 = -1
    dev0 = -1
    homogenous = True
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        d = t.dtype
        if not (
            d is torch.float16
            or d is torch.float32
            or d is torch.bfloat16
            or d is torch.float64
        ):
            homogenous = False
            continue
        if not t.is_contiguous():
            homogenous = False
            continue
        n = t.numel()
        if n == 0:
            homogenous = False
            continue
        if dtype0 is None:
            dtype0 = d
            n0 = n
            dev0 = t.get_device()
        elif d is not dtype0 or n != n0 or t.get_device() != dev0:
            homogenous = False

    if homogenous:
        _run_fast(self, n0, dtype0)
    else:
        _run_general(self)
    return None
