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
"""In-place foreach sign: ``aten::_foreach_sign_(Tensor(a!)[] self) -> ()``.

Execution plan (chosen per call from metadata only; no input-derived state is
reused across calls):

* one contiguous tensor -> one flat masked launch;
* several contiguous tensors of one dtype, each **small** -> one fused launch
  with a 2-D grid ``(nt, cdiv(max_numel, BLOCK))`` where ``program_id(0)``
  picks the tensor from packed address arguments.  Short tensors are
  launch-bound, so collapsing ``nt`` launches into one is the whole win there
  (16x64x64 measured 0.322 ms -> 0.073 ms).  A 17-argument variant is used
  when the group shares one numel, because host argument processing costs
  about 1 us per argument on this target;
* several contiguous tensors of one dtype, any of them **large** -> one launch
  per tensor.  The fused kernel reaches its base pointer through a runtime
  int64 cast, so Triton cannot prove alignment and the accesses are not
  vectorized; measured at ~0.44 TB/s against ~1.3 TB/s for the plain kernel,
  far more than the launch count saves;
* non-contiguous tensors -> stride-decomposed kernel with the dimension list
  collapsed on the host (size-1 dims dropped, adjacent dims merged,
  fastest-varying first).

Launch path: the compiled kernel is cached per specialization key and invoked
through ``CompiledKernel.__getitem__``, which skips Triton's per-call argument
binding and cache-key construction.  The key covers dtype, pointer 16-byte
divisibility, numel divisibility / equal-to-one and int32-vs-int64 width, so a
cached variant is never invoked with arguments it was not compiled for.  Any
failure raises out of the fast path and falls back to the ordinary launch.

The element rule reproduces ATen's ``sign`` CUDA kernel exactly:
``(0 < a) - (a < 0)`` evaluated in the input dtype (probed on this target:
NaN -> 0, -0.0 -> +0.0, bool identity, integers stay integral).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_sign_ as default__foreach_sign_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_BLOCK = 1024
_MAX_DIMS = 8
_MAX_T = 16
# Below this many bytes per tensor the launch count dominates the ~3x
# vectorization penalty of the runtime-addressed fused kernel.
_FUSE_MAX_BYTES = 1 << 20
_I32 = 2147483648

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
    torch.bool: tl.int1,
}

_COMPLEX_NAME = {
    torch.complex32: "ComplexHalf",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

# torch dtype -> compiled kernel, keyed by the specialization bits.
_FLAT_CACHE = {}
_MULTI_CACHE = {}
_FAST_OK = True

_MULTI_ARGS = [
    "p0",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "p10",
    "p11",
    "p12",
    "p13",
    "p14",
    "p15",
]


@triton.jit
def _sign_flat_kernel(x_ptr, n_elem, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    x = tl.load(x_ptr + idx, mask=mask, other=0)
    if x.dtype == tl.int1:
        # sign(bool) is the identity
        y = x
    else:
        y = (x > 0).to(x.dtype) - (x < 0).to(x.dtype)
    tl.store(x_ptr + idx, y, mask=mask)


@triton.jit(do_not_specialize=_MULTI_ARGS)
def _sign_multi_u_kernel(
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
    n_elem,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to 16 equally sized contiguous tensors of dtype ``DT``.

    Axis 0 selects the tensor, axis 1 the block inside it.  Only one element
    count crosses the host boundary and no argument is specialized, so a
    single compiled variant serves every call of this dtype.
    """
    tid = tl.program_id(0)
    c = (tid == 0).to(tl.int64)
    addr = c * p0
    c = (tid == 1).to(tl.int64)
    addr += c * p1
    c = (tid == 2).to(tl.int64)
    addr += c * p2
    c = (tid == 3).to(tl.int64)
    addr += c * p3
    c = (tid == 4).to(tl.int64)
    addr += c * p4
    c = (tid == 5).to(tl.int64)
    addr += c * p5
    c = (tid == 6).to(tl.int64)
    addr += c * p6
    c = (tid == 7).to(tl.int64)
    addr += c * p7
    c = (tid == 8).to(tl.int64)
    addr += c * p8
    c = (tid == 9).to(tl.int64)
    addr += c * p9
    c = (tid == 10).to(tl.int64)
    addr += c * p10
    c = (tid == 11).to(tl.int64)
    addr += c * p11
    c = (tid == 12).to(tl.int64)
    addr += c * p12
    c = (tid == 13).to(tl.int64)
    addr += c * p13
    c = (tid == 14).to(tl.int64)
    addr += c * p14
    c = (tid == 15).to(tl.int64)
    addr += c * p15

    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elem
    ptr = addr.to(tl.pointer_type(DT))
    x = tl.load(ptr + off, mask=mask, other=0)
    if x.dtype == tl.int1:
        y = x
    else:
        y = (x > 0).to(x.dtype) - (x < 0).to(x.dtype)
    tl.store(ptr + off, y, mask=mask)


@triton.jit(
    do_not_specialize=_MULTI_ARGS
    + [
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
    ]
)
def _sign_multi_kernel(
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
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused launch for a small group whose members have different sizes."""
    tid = tl.program_id(0)
    c = (tid == 0).to(tl.int64)
    addr = c * p0
    nn = c * n0
    c = (tid == 1).to(tl.int64)
    addr += c * p1
    nn += c * n1
    c = (tid == 2).to(tl.int64)
    addr += c * p2
    nn += c * n2
    c = (tid == 3).to(tl.int64)
    addr += c * p3
    nn += c * n3
    c = (tid == 4).to(tl.int64)
    addr += c * p4
    nn += c * n4
    c = (tid == 5).to(tl.int64)
    addr += c * p5
    nn += c * n5
    c = (tid == 6).to(tl.int64)
    addr += c * p6
    nn += c * n6
    c = (tid == 7).to(tl.int64)
    addr += c * p7
    nn += c * n7
    c = (tid == 8).to(tl.int64)
    addr += c * p8
    nn += c * n8
    c = (tid == 9).to(tl.int64)
    addr += c * p9
    nn += c * n9
    c = (tid == 10).to(tl.int64)
    addr += c * p10
    nn += c * n10
    c = (tid == 11).to(tl.int64)
    addr += c * p11
    nn += c * n11
    c = (tid == 12).to(tl.int64)
    addr += c * p12
    nn += c * n12
    c = (tid == 13).to(tl.int64)
    addr += c * p13
    nn += c * n13
    c = (tid == 14).to(tl.int64)
    addr += c * p14
    nn += c * n14
    c = (tid == 15).to(tl.int64)
    addr += c * p15
    nn += c * n15

    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < nn.to(tl.int32)
    ptr = addr.to(tl.pointer_type(DT))
    x = tl.load(ptr + off, mask=mask, other=0)
    if x.dtype == tl.int1:
        y = x
    else:
        y = (x > 0).to(x.dtype) - (x < 0).to(x.dtype)
    tl.store(ptr + off, y, mask=mask)


@triton.jit
def _sign_strided_kernel(
    x_ptr,
    n_elem,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = idx < n_elem
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        off += (idx % d0) * s0
        rem = idx // d0
        if RANK > 1:
            off += (rem % d1) * s1
            rem = rem // d1
            if RANK > 2:
                off += (rem % d2) * s2
                rem = rem // d2
                if RANK > 3:
                    off += (rem % d3) * s3
                    rem = rem // d3
                    if RANK > 4:
                        off += (rem % d4) * s4
                        rem = rem // d4
                        if RANK > 5:
                            off += (rem % d5) * s5
                            rem = rem // d5
                            if RANK > 6:
                                off += (rem % d6) * s6
                                rem = rem // d6
                                if RANK > 7:
                                    off += (rem % d7) * s7
    x = tl.load(x_ptr + off, mask=mask, other=0)
    if x.dtype == tl.int1:
        y = x
    else:
        y = (x > 0).to(x.dtype) - (x < 0).to(x.dtype)
    tl.store(x_ptr + off, y, mask=mask)


def _collapse(t):
    """Fastest-first (size, stride) pairs, size-1 dims dropped, merges applied."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return [1], [0]
    merged = []
    for sz, st in sorted(dims, key=lambda p: p[1]):
        if merged and st == merged[-1][1] * merged[-1][0]:
            merged[-1] = (merged[-1][0] * sz, merged[-1][1])
        else:
            merged.append((sz, st))
    return [m[0] for m in merged], [m[1] for m in merged]


def _launch_flat(t, n):
    """Flat masked launch of one contiguous tensor; uses the cached variant."""
    global _FAST_OK
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if _FAST_OK:
        key = (t.dtype, t.data_ptr() & 15, n & 15, n == 1, n < _I32)
        comp = _FLAT_CACHE.get(key)
        if comp is None:
            try:
                comp = _sign_flat_kernel.warmup(t, n, _BLOCK, grid=grid)
            except Exception:
                _FAST_OK = False
            else:
                _FLAT_CACHE[key] = comp
        if comp is not None:
            try:
                comp[grid](t, n)
                return
            except Exception:
                _FLAT_CACHE.pop(key, None)
                _FAST_OK = False
    _sign_flat_kernel[grid](t, n, _BLOCK)


def _launch_strided(t, n):
    dims, strides = _collapse(t)
    rank = len(dims)
    if rank > _MAX_DIMS:
        raise RuntimeError("_foreach_sign_: rank exceeds supported maximum")
    pad_d = dims + [1] * (_MAX_DIMS - rank)
    pad_s = strides + [0] * (_MAX_DIMS - rank)
    _sign_strided_kernel[((n + _BLOCK - 1) // _BLOCK,)](
        t, n, *pad_d, *pad_s, rank, _BLOCK
    )


def _launch_fused(ptrs, m, hi, uniform, nums, tl_dt, dt):
    """One fused launch for a small same-dtype group (ptrs padded to 16)."""
    global _FAST_OK
    grid = (m, (hi + _BLOCK - 1) // _BLOCK)
    key = (dt, hi < _I32) if uniform else (dt, 1, hi < _I32)
    comp = _MULTI_CACHE.get(key) if _FAST_OK else None
    if comp is None and _FAST_OK:
        try:
            if uniform:
                comp = _sign_multi_u_kernel.warmup(*ptrs, hi, tl_dt, _BLOCK, grid=grid)
            else:
                comp = _sign_multi_kernel.warmup(*ptrs, *nums, tl_dt, _BLOCK, grid=grid)
        except Exception:
            _FAST_OK = False
        else:
            _MULTI_CACHE[key] = comp
    if comp is not None:
        try:
            if uniform:
                comp[grid](*ptrs, hi)
            else:
                comp[grid](*ptrs, *nums)
            return
        except Exception:
            _MULTI_CACHE.pop(key, None)
            _FAST_OK = False
    if uniform:
        _sign_multi_u_kernel[grid](*ptrs, hi, tl_dt, _BLOCK)
    else:
        _sign_multi_kernel[grid](*ptrs, *nums, tl_dt, _BLOCK)


def _launch_group(grp, dt, tl_dt):
    """Contiguous, same-dtype, non-empty tensors."""
    k = len(grp)
    if k == 1:
        t = grp[0]
        _launch_flat(t, t.numel())
        return
    nums = [t.numel() for t in grp]
    hi = max(nums)
    if hi * grp[0].element_size() > _FUSE_MAX_BYTES:
        for i in range(k):
            _launch_flat(grp[i], nums[i])
        return
    uniform = min(nums) == hi
    start = 0
    while start < k:
        chunk = grp[start : start + _MAX_T]
        cn = nums[start : start + _MAX_T]
        start += _MAX_T
        m = len(chunk)
        if m == 1:
            _launch_flat(chunk[0], cn[0])
            continue
        ptrs = [t.data_ptr() for t in chunk]
        if m < _MAX_T:
            ptrs += [ptrs[-1]] * (_MAX_T - m)
        if uniform:
            _launch_fused(ptrs, m, hi, True, None, tl_dt, dt)
        else:
            if m < _MAX_T:
                cn = cn + [0] * (_MAX_T - m)
            _launch_fused(ptrs, m, hi, False, cn, tl_dt, dt)


def _foreach_sign_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_SIGN_")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_sign_(self)
    tensors = self
    kind = type(tensors)
    if kind is not list and kind is not tuple:
        if not isinstance(tensors, (list, tuple)):
            raise TypeError("argument 'self' must be tuple of Tensors")
    n_t = len(tensors)
    if n_t == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Single-tensor fast path: skip the grouping machinery entirely.
    if n_t == 1:
        t = tensors[0]
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt.is_complex:
            raise NotImplementedError(
                "\"foreach_unary_op_cuda_\" not implemented for '%s'"
                % _COMPLEX_NAME.get(dt, str(dt))
            )
        n = t.numel()
        if n:
            if t.is_contiguous() and n < _I32:
                _launch_flat(t, n)
            else:
                _launch_strided(t, n)
        return None

    # Pass 1: validate the whole list (ATen raises before touching any element)
    # and classify each tensor.  Metadata only, no launches, no data movement.
    flat = {}
    strided = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dt = t.dtype
        if dt.is_complex:
            raise NotImplementedError(
                "\"foreach_unary_op_cuda_\" not implemented for '%s'"
                % _COMPLEX_NAME.get(dt, str(dt))
            )
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous() and n < _I32 and dt in _TL_DTYPE:
            grp = flat.get(dt)
            if grp is None:
                flat[dt] = [t]
            else:
                grp.append(t)
        else:
            strided.append((t, n))

    # Pass 2: launches.
    for dt, grp in flat.items():
        _launch_group(grp, dt, _TL_DTYPE[dt])
    for t, n in strided:
        _launch_strided(t, n)
    return None
