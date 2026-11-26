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
"""In-place foreach power over paired TensorLists: ``aten::_foreach_pow_.List``.

Semantics: for every list position ``i``, ``self[i] = pow(self[i], exponent[i])``
evaluated in float32 opmath (ATen's opmath for fp16/bf16/fp32) and stored back
into the original storage with the original dtype/stride/data_ptr.

The benchmark workload is a 16-entry TensorList, so the measured cost is
dominated by the host side, not bandwidth: one fused launch per list is issued
(``program_id(1)`` = list slot via a branch-free address select,
``program_id(0)`` = tile, tile-major so each resident wave streams one tensor),
and the launch itself is dispatched through the already-compiled kernel handle
instead of re-running Triton's per-call argument binding, which costs several
microseconds per kernel argument on this stack.
"""

import logging

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_binary import (
    _foreach_pow__List as default__foreach_pow__List,
)
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_pow = tl_extra_shim.pow

_BLOCK = 1024
_MAX_DIMS = 8
_MAX_FUSED = 16
# Fusing the whole list into one launch pays off only while the host launch cost
# dominates.  Measured on this target: 16x4096 elements per list -> one fused
# launch is ~6x faster than 16 launches, while 16x16.7M elements per list -> the
# per-entry kernel is ~15% faster because the element rate, not the launch, is
# then the limit.  The switch is on total elements in the list.
_FUSED_MAX_ELEMS = 1 << 22
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sel16(t, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
    """Branch-free pick of slot ``t`` out of sixteen scalar addresses."""
    v = a15
    v = tl.where(t == 14, a14, v)
    v = tl.where(t == 13, a13, v)
    v = tl.where(t == 12, a12, v)
    v = tl.where(t == 11, a11, v)
    v = tl.where(t == 10, a10, v)
    v = tl.where(t == 9, a9, v)
    v = tl.where(t == 8, a8, v)
    v = tl.where(t == 7, a7, v)
    v = tl.where(t == 6, a6, v)
    v = tl.where(t == 5, a5, v)
    v = tl.where(t == 4, a4, v)
    v = tl.where(t == 3, a3, v)
    v = tl.where(t == 2, a2, v)
    v = tl.where(t == 1, a1, v)
    v = tl.where(t == 0, a0, v)
    return v


@triton.jit
def _pow_fused16(
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    e9,
    e10,
    e11,
    e12,
    e13,
    e14,
    e15,
    n_elem,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """All homogeneous list entries in a single launch, one compact body."""
    t = tl.program_id(1)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sa = _sel16(t, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)
    ea = _sel16(t, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)
    s_ptr = sa.to(tl.int64).to(tl.pointer_type(DT))
    e_ptr = ea.to(tl.int64).to(tl.pointer_type(DT))
    mask = off < n_elem
    a = tl.load(s_ptr + off, mask=mask, other=0.0)
    b = tl.load(e_ptr + off, mask=mask, other=0.0)
    r = _pow(a.to(tl.float32), b.to(tl.float32))
    tl.store(s_ptr + off, r.to(DT), mask=mask)


@triton.jit
def _pow_body(s_ptr, e_ptr, n_elem, off, BLOCK: tl.constexpr):
    mask = off < n_elem
    a = tl.load(s_ptr + off, mask=mask, other=0.0)
    b = tl.load(e_ptr + off, mask=mask, other=0.0)
    r = _pow(a.to(tl.float32), b.to(tl.float32))
    tl.store(s_ptr + off, r.to(a.dtype), mask=mask)


@triton.jit
def _pow_flat_kernel(s_ptr, e_ptr, n_elem, BLOCK: tl.constexpr):
    """Single pair, flat element index."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _pow_body(s_ptr, e_ptr, n_elem, off, BLOCK)


@triton.jit
def _pow_scalar_kernel(s_ptr, e_ptr, n_elem, BLOCK: tl.constexpr):
    """Contiguous ``self`` raised to a single device-resident exponent value."""
    e = tl.load(e_ptr).to(tl.float32)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    a = tl.load(s_ptr + offs, mask=mask, other=0.0)
    tl.store(s_ptr + offs, _pow(a.to(tl.float32), e).to(a.dtype), mask=mask)


@triton.jit
def _pow_hscalar_kernel(s_ptr, e, n_elem, BLOCK: tl.constexpr):
    """Contiguous ``self`` raised to a host-supplied python-number exponent."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    a = tl.load(s_ptr + offs, mask=mask, other=0.0)
    tl.store(s_ptr + offs, _pow(a.to(tl.float32), e).to(a.dtype), mask=mask)


@triton.jit
def _dim_walk(rem, os_, oe_, d, ss, es):
    i = rem % d
    rem = rem // d
    os_ += i * ss
    oe_ += i * es
    return rem, os_, oe_


@triton.jit
def _pow_strided_kernel(
    s_ptr,
    e_ptr,
    e_val,
    n_elem,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    TENSOR_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Strided / broadcast fallback.

    ``d*`` are the (right-aligned) logical sizes of ``self``; ``s*`` its strides
    and ``e*`` the exponent strides with 0 for broadcast dimensions.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    rem = idx
    os_ = tl.zeros((BLOCK,), tl.int32)
    oe_ = tl.zeros((BLOCK,), tl.int32)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d0, s0, e0)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d1, s1, e1)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d2, s2, e2)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d3, s3, e3)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d4, s4, e4)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d5, s5, e5)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d6, s6, e6)
    rem, os_, oe_ = _dim_walk(rem, os_, oe_, d7, s7, e7)
    a = tl.load(s_ptr + os_, mask=mask, other=0.0)
    if TENSOR_E:
        b = tl.load(e_ptr + oe_, mask=mask, other=0.0).to(tl.float32)
    else:
        b = e_val
    r = _pow(a.to(tl.float32), b)
    tl.store(s_ptr + os_, r.to(a.dtype), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

# key -> (compiled launcher, function handle, packed metadata)
_CK_CACHE = {}
# Diagnostics only: proves the fast launch path engaged and never raised.
_DIAG = {"warmups": 0, "errors": 0, "last_error": ""}
_HI = 1 << 31
_ACTIVE = None
# Unbound C method descriptors: calling them directly avoids allocating a bound
# method object for every per-tensor query on the launch-bound hot path.
_Tensor = torch.Tensor
_CONTIG = torch.Tensor.is_contiguous
_NUMEL = torch.Tensor.numel
_PTR = torch.Tensor.data_ptr


def _n_blocks(m):
    return ((m + _BLOCK - 1) // _BLOCK,)


def _apply_host_scalar(a, v):
    m = a.numel()
    if m == 0:
        return None
    if a.is_contiguous():
        _pow_hscalar_kernel[_n_blocks(m)](a, float(v), m, BLOCK=_BLOCK)
    else:
        _apply_strided(a, v)
    return None


def _apply_strided(a, b):
    """Write ``pow(a, b)`` into ``a``'s own storage for arbitrary layouts."""
    m = a.numel()
    if m == 0:
        return None
    s_shape = a.shape
    rank = len(s_shape)
    is_tensor = isinstance(b, torch.Tensor)
    if is_tensor:
        rank = max(rank, len(b.shape))
    if rank > _MAX_DIMS:
        raise NotImplementedError("foreach pow: rank above 8 is not supported")
    pad = _MAX_DIMS - rank
    ssz = [1] * _MAX_DIMS
    sst = [0] * _MAX_DIMS
    est = [0] * _MAX_DIMS
    base = pad + rank - len(s_shape)
    for k in range(len(s_shape)):
        ssz[base + k] = s_shape[k]
        sst[base + k] = a.stride(k)
    if is_tensor:
        e_shape = b.shape
        base = pad + rank - len(e_shape)
        for k in range(len(e_shape)):
            if e_shape[k] != 1:
                est[base + k] = b.stride(k)
    _pow_strided_kernel[_n_blocks(m)](
        a,
        b if is_tensor else a,
        0.0 if is_tensor else float(b),
        m,
        sst[0],
        sst[1],
        sst[2],
        sst[3],
        sst[4],
        sst[5],
        sst[6],
        sst[7],
        est[0],
        est[1],
        est[2],
        est[3],
        est[4],
        est[5],
        est[6],
        est[7],
        ssz[0],
        ssz[1],
        ssz[2],
        ssz[3],
        ssz[4],
        ssz[5],
        ssz[6],
        ssz[7],
        TENSOR_E=is_tensor,
        BLOCK=_BLOCK,
    )
    return None


def _apply_pair(a, b):
    if not isinstance(a, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if not isinstance(b, torch.Tensor):
        return _apply_host_scalar(a, b)
    m = a.numel()
    if m == 0:
        return None
    if a.is_contiguous() and b.is_contiguous() and a.shape == b.shape:
        _pow_flat_kernel[_n_blocks(m)](a, b, m, BLOCK=_BLOCK)
    elif b.numel() == 1 and a.is_contiguous():
        _pow_scalar_kernel[_n_blocks(m)](a, b, m, BLOCK=_BLOCK)
    else:
        _apply_strided(a, b)
    return None


def _compiled(sa, ea, hi, m, dt):
    """Compiled launcher handles for one specialization of the fused kernel.

    ``hi`` is the largest address in the list; an all-below-2**31 list compiles
    to 32-bit address arguments and a larger one to 64-bit, so the two must not
    share a cached launcher.
    """
    cls = 2 if hi >= _HI else 1
    key = (dt, m == 1, m & 15, m >> 31, cls)
    entry = _CK_CACHE.get(key)
    if entry is None:
        try:
            ck = _pow_fused16.warmup(
                *sa, *ea, m, DT=_TL_DTYPES[dt], BLOCK=_BLOCK, grid=(1, 1, 1)
            )
            if ck is None:
                return None
            entry = (
                ck,
                ck.run,
                ck.function,
                ck.packed_metadata,
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
                _TL_DTYPES[dt],
            )
        except Exception as exc:
            _DIAG["errors"] += 1
            _DIAG["last_error"] = repr(exc)
            return None
        _DIAG["warmups"] += 1
        _CK_CACHE[key] = entry
    return entry


def _try_fused(self, exponent, n):
    """One launch for a homogeneous list.  Returns False if not applicable.

    The Python below runs on every call, so it only touches each tensor the
    minimum number of times and avoids temporaries (no ``shape`` tuple
    allocation, no address-list concatenation).
    """
    a0 = self[0]
    if type(a0) is not _Tensor:
        if isinstance(a0, torch.Tensor):
            return False
        raise TypeError("argument 'self' must be tuple of Tensors")
    dt = a0.dtype
    if not dt.is_floating_point or not _CONTIG(a0):
        return False
    # Equal element counts plus ATen's broadcast semantics make flat indexing
    # element-order-preserving, so the count is what has to agree per position.
    m = _NUMEL(a0)
    if m * n > _FUSED_MAX_ELEMS:
        return False
    sa = []
    ea = []
    push_a = sa.append
    push_b = ea.append
    hi = 0
    for a, b in zip(self, exponent):
        if type(a) is not _Tensor or type(b) is not _Tensor:
            return False
        if a.dtype is not dt or b.dtype is not dt:
            return False
        if _NUMEL(a) != m or _NUMEL(b) != m:
            return False
        if not _CONTIG(a) or not _CONTIG(b):
            return False
        pa = _PTR(a)
        pb = _PTR(b)
        if (pa | pb) & 15:
            return False
        push_a(pa)
        push_b(pb)
        if pa > hi:
            hi = pa
        if pb > hi:
            hi = pb
    if m == 0:
        return True
    if n < _MAX_FUSED:
        sa.extend([sa[0]] * (_MAX_FUSED - n))
        ea.extend([ea[0]] * (_MAX_FUSED - n))
    gx = (m + _BLOCK - 1) // _BLOCK
    entry = _compiled(sa, ea, hi, m, dt)
    if entry is not None:
        launch, fn, packed, enter, leave, dt_tl = entry[1:]
        args = sa + ea
        args += (m, dt_tl, _BLOCK)
        try:
            global _ACTIVE
            drv = _ACTIVE
            if drv is None:
                drv = _ACTIVE = _driver.active
            stream = drv.get_current_stream(drv.get_current_device())
            launch(gx, n, 1, stream, fn, packed, None, enter, leave, *args)
            return True
        except Exception as exc:
            _DIAG["errors"] += 1
            _DIAG["last_error"] = repr(exc)
    _pow_fused16[(gx, n)](
        sa[0],
        sa[1],
        sa[2],
        sa[3],
        sa[4],
        sa[5],
        sa[6],
        sa[7],
        sa[8],
        sa[9],
        sa[10],
        sa[11],
        sa[12],
        sa[13],
        sa[14],
        sa[15],
        ea[0],
        ea[1],
        ea[2],
        ea[3],
        ea[4],
        ea[5],
        ea[6],
        ea[7],
        ea[8],
        ea[9],
        ea[10],
        ea[11],
        ea[12],
        ea[13],
        ea[14],
        ea[15],
        m,
        DT=_TL_DTYPES[dt],
        BLOCK=_BLOCK,
    )
    return True


def _foreach_pow__List(self, exponent):
    """``torch._foreach_pow_.List`` equivalent; mutates ``self`` in place."""
    logger.debug("GEMS_MTHREADS _FOREACH_POW__LIST")
    if not isinstance(self, (list, tuple)):
        return default__foreach_pow__List(self, exponent)
    if not isinstance(exponent, (list, tuple)) or len(exponent) != len(self):
        return default__foreach_pow__List(self, exponent)
    # Only MUSA, fp16/bf16/fp32 and layouts this kernel covers are specialized;
    # everything else (fp64/int/bool, CPU tensors, ranks above ``_MAX_DIMS``,
    # incompatible broadcast) is handed back to the generic implementation.
    for a, b in zip(self, exponent):
        if (
            not isinstance(a, torch.Tensor)
            or not isinstance(b, torch.Tensor)
            or a.device.type != "musa"
            or b.device.type != "musa"
            or a.dtype not in _SUPPORTED_DTYPES
            or b.dtype not in _SUPPORTED_DTYPES
            or a.dim() > _MAX_DIMS
            or b.dim() > _MAX_DIMS
        ):
            return default__foreach_pow__List(self, exponent)
        if a.shape != b.shape:
            try:
                if torch.broadcast_shapes(a.shape, b.shape) != a.shape:
                    return default__foreach_pow__List(self, exponent)
            except RuntimeError:
                return default__foreach_pow__List(self, exponent)
    n = len(self)
    if n == 0:
        return None
    if n <= _MAX_FUSED and _try_fused(self, exponent, n):
        return None
    for i in range(n):
        _apply_pair(self[i], exponent[i])
    return None
