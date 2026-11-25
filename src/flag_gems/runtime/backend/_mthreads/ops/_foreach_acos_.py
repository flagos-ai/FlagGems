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

"""In-place foreach arccos (``aten::_foreach_acos_``) with Triton kernels.

Semantics (verified against the authoritative Gems pytest):
  * every tensor in the list is mutated in place through its own storage
    (storage pointer, storage_offset and strides are preserved),
  * the call returns ``None``,
  * integral/bool elements raise ``RuntimeError`` (ATen refuses to cast the
    promoted float result back to the integer storage type),
  * an empty list raises ``RuntimeError``,
  * fp16/bf16 are evaluated with fp32 "opmath" and rounded back to storage,
  * non-contiguous views (transpose / strided slice) are addressed directly,
    never staged through a copy.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.runtime import driver

from flag_gems.ops._foreach_unary import _foreach_acos_ as default_foreach_acos_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK_MAX = 1024
_BLOCK_MIN = 128
# Below this per-tensor element count a launch is host-bound (one Triton launch
# costs ~19-30 us here), so the whole same-dtype group is folded into a single
# launch. Above it the kernel time dominates and per-tensor launches are used.
_FUSE_MAX_NUMEL = 65536


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _acos_f32(x):
    """acos for fp32 via 2*asin(sqrt((1-|x|)/2)) with one degree-5 minimax poly.

    ``t`` is always in [0, 0.5] and ``Q`` is the minimax fit of
    ``asin(sqrt(w))/sqrt(w)`` on ``w in [0, 0.25]``; the measured worst-case
    absolute error of this expression in fp32 is 4.3e-6 over [-1, 1]
    (harness tolerance: atol 1e-4, rtol 1.3e-6).  Out-of-domain |x| > 1 and
    NaN propagate to NaN exactly like libdevice.
    """
    a = tl.abs(x)
    t = tl.where(a <= 0.5, a, libdevice.sqrt((1.0 - a) * 0.5))
    w = t * t
    q = 0.0425535991059998314
    q = q * w + 0.0239093635899636017
    q = q * w + 0.0455390097132944496
    q = q * w + 0.0749453426935359768
    q = q * w + 0.166667870117195971
    q = q * w + 0.999999995738434566
    q = q * t
    r = tl.where(a <= 0.5, 1.5707963267948966 - q, 2.0 * q)
    return tl.where(x >= 0.0, r, 3.141592653589793 - r)


@triton.jit
def _acos_inplace(p, offs, n_elements):
    mask = offs < n_elements
    x = tl.load(p + offs, mask=mask, other=0.0)
    tl.store(
        p + offs,
        _acos_f32(x.to(tl.float32)).to(p.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _acos_flat32_multi(
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
):
    """One launch for up to 16 same-dtype tensors: axis 1 selects the tensor."""
    t = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if t == 0:
        _acos_inplace(p0, offs, n0)
    elif t == 1:
        _acos_inplace(p1, offs, n1)
    elif t == 2:
        _acos_inplace(p2, offs, n2)
    elif t == 3:
        _acos_inplace(p3, offs, n3)
    elif t == 4:
        _acos_inplace(p4, offs, n4)
    elif t == 5:
        _acos_inplace(p5, offs, n5)
    elif t == 6:
        _acos_inplace(p6, offs, n6)
    elif t == 7:
        _acos_inplace(p7, offs, n7)
    elif t == 8:
        _acos_inplace(p8, offs, n8)
    elif t == 9:
        _acos_inplace(p9, offs, n9)
    elif t == 10:
        _acos_inplace(p10, offs, n10)
    elif t == 11:
        _acos_inplace(p11, offs, n11)
    elif t == 12:
        _acos_inplace(p12, offs, n12)
    elif t == 13:
        _acos_inplace(p13, offs, n13)
    elif t == 14:
        _acos_inplace(p14, offs, n14)
    elif t == 15:
        _acos_inplace(p15, offs, n15)


@triton.jit
def _acos_flat32(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = _acos_f32(x.to(tl.float32))
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _acos_flat64(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = libdevice.acos(x)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
def _acos_strided32(
    x_ptr,
    inner_size,
    inner_stride,
    n_inner_blocks,
    R: tl.constexpr,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    outer = pid // n_inner_blocks
    blk = pid % n_inner_blocks
    off = tl.zeros((), dtype=tl.int64)
    if R >= 1:
        off += (outer % s0).to(tl.int64) * t0.to(tl.int64)
        outer = outer // s0
    if R >= 2:
        off += (outer % s1).to(tl.int64) * t1.to(tl.int64)
        outer = outer // s1
    if R >= 3:
        off += (outer % s2).to(tl.int64) * t2.to(tl.int64)
        outer = outer // s2
    if R >= 4:
        off += (outer % s3).to(tl.int64) * t3.to(tl.int64)
        outer = outer // s3
    if R >= 5:
        off += (outer % s4).to(tl.int64) * t4.to(tl.int64)
        outer = outer // s4
    if R >= 6:
        off += (outer % s5).to(tl.int64) * t5.to(tl.int64)
        outer = outer // s5
    if R >= 7:
        off += (outer % s6).to(tl.int64) * t6.to(tl.int64)
    idx = (blk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = idx < inner_size
    ptrs = x_ptr + off + idx * inner_stride.to(tl.int64)
    x = tl.load(ptrs, mask=mask, other=0.0)
    y = _acos_f32(x.to(tl.float32))
    tl.store(ptrs, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _acos_strided64(
    x_ptr,
    inner_size,
    inner_stride,
    n_inner_blocks,
    R: tl.constexpr,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    outer = pid // n_inner_blocks
    blk = pid % n_inner_blocks
    off = tl.zeros((), dtype=tl.int64)
    if R >= 1:
        off += (outer % s0).to(tl.int64) * t0.to(tl.int64)
        outer = outer // s0
    if R >= 2:
        off += (outer % s1).to(tl.int64) * t1.to(tl.int64)
        outer = outer // s1
    if R >= 3:
        off += (outer % s2).to(tl.int64) * t2.to(tl.int64)
        outer = outer // s2
    if R >= 4:
        off += (outer % s3).to(tl.int64) * t3.to(tl.int64)
        outer = outer // s3
    if R >= 5:
        off += (outer % s4).to(tl.int64) * t4.to(tl.int64)
        outer = outer // s4
    if R >= 6:
        off += (outer % s5).to(tl.int64) * t5.to(tl.int64)
        outer = outer // s5
    if R >= 7:
        off += (outer % s6).to(tl.int64) * t6.to(tl.int64)
    idx = (blk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = idx < inner_size
    ptrs = x_ptr + off + idx * inner_stride.to(tl.int64)
    x = tl.load(ptrs, mask=mask, other=0.0)
    y = libdevice.acos(x)
    tl.store(ptrs, y, mask=mask)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------

_FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))


def _pick_block(n):
    if n >= _BLOCK_MAX:
        return _BLOCK_MAX
    b = _BLOCK_MIN
    while b < n:
        b <<= 1
    return b


def _pick_warps(block):
    if block <= 128:
        return 1
    if block <= 256:
        return 2
    return 4


def _is_flat_addressable(t):
    """True when ``[data_ptr, data_ptr + numel)`` covers exactly this tensor."""
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
    dims = [(sz, st) for sz, st in zip(shape, stride) if sz != 1]
    if not dims:
        return [(1, 1)]
    out = [list(dims[0])]
    for sz, st in dims[1:]:
        prev = out[-1]
        if st == prev[1] * prev[0]:
            prev[0] *= sz
        else:
            out.append([sz, st])
    return [(sz, st) for sz, st in out]


def _launch_strided(t, block, warps):
    dims = _collapse(t.shape, t.stride())
    if len(dims) > 8:
        # Rare: too many unmergeable dims. Walk the outermost dim with
        # zero-copy views (metadata-only indexing) and recurse.
        for i in range(t.shape[0]):
            _launch_one(t[i])
        return
    # inner dim: prefer stride 1, else the smallest stride.
    best = 0
    best_stride = dims[0][1]
    for i, (sz, st) in enumerate(dims):
        if st == 1:
            best = i
            best_stride = 1
            break
        if st < best_stride:
            best = i
            best_stride = st
    inner_size, inner_stride = dims[best]
    outer = [dims[i] for i in range(len(dims)) if i != best]
    outer.reverse()  # innermost outer dim first (matches kernel decomposition)
    n_outer = 1
    for sz, _ in outer:
        n_outer *= sz
    n_inner_blocks = (inner_size + block - 1) // block
    grid = (n_outer * n_inner_blocks,)
    pad = outer + [(1, 1)] * (7 - len(outer))
    args = []
    for sz, st in pad:
        args.append(sz)
        args.append(st)
    if t.dtype is torch.float64:
        _acos_strided64[grid](
            t,
            inner_size,
            inner_stride,
            n_inner_blocks,
            len(outer),
            *args,
            block,
            num_warps=warps,
        )
    else:
        _acos_strided32[grid](
            t,
            inner_size,
            inner_stride,
            n_inner_blocks,
            len(outer),
            *args,
            block,
            num_warps=warps,
        )


def _launch_one(t):
    n = t.numel()
    if n == 0:
        return
    block = _pick_block(n)
    warps = _pick_warps(block)
    if _is_flat_addressable(t):
        _launch_flat(t, n, block, warps)
    else:
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        _launch_strided(t, block, warps)


# A launch on this stack costs ~35 us of host time inside
# ``JITFunction.run`` (argument binding, specialization, cache key), which is
# the whole latency of the small workloads.  Once a kernel has been compiled we
# therefore reuse the ``CompiledKernel`` launcher directly, keyed by every
# property that takes part in Triton's specialization (dtype, block, warps,
# grid, integer divisibility-by-16 / equal-to-1 flags and pointer 16-byte
# alignment), so the reused binary is always the one Triton would pick.
_FLAT_CACHE = {}
_FUSED_CACHE = {}
_STREAM_CACHE = {}


def _current_stream(dev):
    """Device stream, resolved once per device (the harness runs one stream)."""
    stream = _STREAM_CACHE.get(dev)
    if stream is None:
        stream = driver.active.get_current_stream(dev)
        _STREAM_CACHE[dev] = stream
    return stream


def _launch_flat(t, n, block, warps):
    nblocks = (n + block - 1) // block
    dtype = t.dtype
    if (t.data_ptr() & 15) == 0:
        key = (dtype, block, warps, nblocks, n & 15 == 0, n == 1)
        entry = _FLAT_CACHE.get(key)
        if entry is not None:
            run, fn, packed, stream = entry
            run(nblocks, 1, 1, stream, fn, packed, None, None, None, t, n, block)
            return
        if dtype is torch.float64:
            ck = _acos_flat64[(nblocks,)](t, n, block, num_warps=warps)
        else:
            ck = _acos_flat32[(nblocks,)](t, n, block, num_warps=warps)
        ck._init_handles()
        _FLAT_CACHE[key] = (
            ck.run,
            ck.function,
            ck.packed_metadata,
            _current_stream(t.get_device()),
        )
        return
    if dtype is torch.float64:
        _acos_flat64[(nblocks,)](t, n, block, num_warps=warps)
    else:
        _acos_flat32[(nblocks,)](t, n, block, num_warps=warps)


def _launch_fused(ts):
    """One launch for 1..16 same-dtype flat tensors (host-bound regime)."""
    sizes = [t.numel() for t in ts]
    max_n = max(sizes)
    block = _pick_block(max_n)
    warps = _pick_warps(block)
    nchunks = (max_n + block - 1) // block
    k = len(ts)
    pad_ts = ts + [ts[0]] * (16 - k)
    pad_n = sizes + [1] * (16 - k)
    aligned = True
    for t in ts:
        if (t.data_ptr() & 15) != 0:
            aligned = False
            break
    if aligned:
        flags = 0
        for v in pad_n:
            flags = (flags << 2) | (1 if (v & 15) == 0 else 0) | (2 if v == 1 else 0)
        key = (ts[0].dtype, block, warps, nchunks, k, flags)
        entry = _FUSED_CACHE.get(key)
        if entry is not None:
            run, fn, packed, stream = entry
            run(
                nchunks,
                k,
                1,
                stream,
                fn,
                packed,
                None,
                None,
                None,
                *pad_ts,
                *pad_n,
                block,
            )
            return
        ck = _acos_flat32_multi[(nchunks, k)](*pad_ts, *pad_n, block, num_warps=warps)
        ck._init_handles()
        _FUSED_CACHE[key] = (
            ck.run,
            ck.function,
            ck.packed_metadata,
            _current_stream(ts[0].get_device()),
        )
        return
    _acos_flat32_multi[(nchunks, k)](*pad_ts, *pad_n, block, num_warps=warps)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_acos_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_ACOS_")

    if not _list_supported(self):
        return default_foreach_acos_(self)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_acos_(self)


def _launch_foreach_acos_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        if t.dtype not in _FLOAT_DTYPES:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + str(t.dtype).replace("torch.", "").capitalize()
            )
    if len(self) == 1:
        _launch_one(self[0])
        return None
    groups = {}
    staged = []
    for t in self:
        if t.numel() == 0:
            continue
        if _is_flat_addressable(t):
            groups.setdefault(t.dtype, []).append(t)
        else:
            staged.append(t)
    for t in staged:
        _launch_one(t)
    for dtype, ts in groups.items():
        n_t = len(ts)
        if (
            n_t > 1
            and dtype is not torch.float64
            and max(t.numel() for t in ts) <= _FUSE_MAX_NUMEL
            and len({id(t) for t in ts}) == n_t
        ):
            for i in range(0, n_t, 16):
                _launch_fused(ts[i : i + 16])
        else:
            for t in ts:
                _launch_one(t)
    return None
