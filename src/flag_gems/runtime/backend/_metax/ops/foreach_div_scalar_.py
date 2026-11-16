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

"""In-place foreach division of a tensor list by one scalar (Triton kernels).

Reproduces aten::_foreach_div_.Scalar semantics:

    for t in self (list order):  t[i] = cast_to_dtype( cast_to_opmath(t[i])
                                                      * (1 / opmath(scalar)) )

The reciprocal is formed in the opmath type (fp32 for fp16/bf16/fp32, fp64 for
fp64), multiplied in opmath and cast back with round-to-nearest -- exactly the
ATen CUDA div-by-CPU-scalar path.  Storage is mutated in place, nothing is
allocated and None is returned.

The op is a pure streaming pass, so latency is host-bound for small payloads
and bandwidth-bound for large ones.  Two things keep the host side small:

* one launch serves the whole list (per-tensor block ranges carved out of a
  single 1-D grid) when every tensor shares a dtype and is contiguous;
* that launch goes through the already-compiled kernel object instead of
  re-running Triton's argument binder, specialization-key construction and
  cache lookup on every call.  Compiled kernels are cached under exactly the
  specialization Triton would derive (dtype, list length, and for every element
  count its divisible-by-16 / equal-to-1 class), the raw packed launcher is
  called directly (no per-call Python wrapper), and the fast launch is only
  used when the data pointers keep the 16-byte alignment the compiled code
  assumes; otherwise the regular JIT path runs.
"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

try:  # correctly-rounded fp32/fp64 reciprocals
    import numpy as _np
except Exception:  # pragma: no cover - struct fallback
    _np = None

_BLOCK = 1024
_WARPS = 4
_MAX_DIMS = 8
_FUSED_MAX = 4
_MAX_ELEMS = 0x7FFFFFFF - 8192
_FLOAT_SET = frozenset((torch.float16, torch.bfloat16, torch.float32, torch.float64))


# ---------------------------------------------------------------------------
# device helpers / kernels
# ---------------------------------------------------------------------------
@triton.jit
def _div_apply(p, off, mask, inv32, inv_hi, inv_lo, F64: tl.constexpr):
    x = tl.load(p + off, mask=mask, other=0.0)
    if F64:
        bits = ((inv_hi.to(tl.int64) & 0xFFFFFFFF) << 32) | (
            inv_lo.to(tl.int64) & 0xFFFFFFFF
        )
        y = x * bits.to(tl.float64, bitcast=True)
    else:
        y = (x.to(tl.float32) * inv32).to(x.dtype)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _div_contig_kernel(
    p,
    n,
    inv32,
    inv_hi,
    inv_lo,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        off = pid * BLOCK + tl.arange(0, BLOCK)
    _div_apply(p, off, off < n, inv32, inv_hi, inv_lo, F64)


@triton.jit(do_not_specialize=["inv_hi", "inv_lo"])
def _div_list_kernel(
    p0,
    p1,
    p2,
    p3,
    n0,
    n1,
    n2,
    n3,
    inv32,
    inv_hi,
    inv_lo,
    BLOCK: tl.constexpr,
    LIST_LEN: tl.constexpr,
    F64: tl.constexpr,
):
    """One launch for a whole contiguous, single-dtype tensor list.

    The grid holds sum_i cdiv(n_i, BLOCK) programs; the first c0 programs serve
    tensor 0, the next c1-c0 serve tensor 1, and so on.
    """
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    c0 = (n0 + (BLOCK - 1)) // BLOCK
    if pid < c0:
        off = pid * BLOCK + lanes
        _div_apply(p0, off, off < n0, inv32, inv_hi, inv_lo, F64)
    else:
        if LIST_LEN > 1:
            c1 = c0 + (n1 + (BLOCK - 1)) // BLOCK
            if pid < c1:
                off = (pid - c0) * BLOCK + lanes
                _div_apply(p1, off, off < n1, inv32, inv_hi, inv_lo, F64)
            else:
                if LIST_LEN > 2:
                    c2 = c1 + (n2 + (BLOCK - 1)) // BLOCK
                    if pid < c2:
                        off = (pid - c1) * BLOCK + lanes
                        _div_apply(p2, off, off < n2, inv32, inv_hi, inv_lo, F64)
                    else:
                        if LIST_LEN > 3:
                            off = (pid - c2) * BLOCK + lanes
                            _div_apply(p3, off, off < n3, inv32, inv_hi, inv_lo, F64)


@triton.jit
def _div_strided_kernel(
    p,
    n,
    inv32,
    inv_hi,
    inv_lo,
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
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    if I64:
        rem = offs.to(tl.int64)
        off = rem * 0
    else:
        rem = offs
        off = offs * 0

    idx = rem % d0
    rem = rem // d0
    off += idx * s0
    if NDIM > 1:
        idx = rem % d1
        rem = rem // d1
        off += idx * s1
    if NDIM > 2:
        idx = rem % d2
        rem = rem // d2
        off += idx * s2
    if NDIM > 3:
        idx = rem % d3
        rem = rem // d3
        off += idx * s3
    if NDIM > 4:
        idx = rem % d4
        rem = rem // d4
        off += idx * s4
    if NDIM > 5:
        idx = rem % d5
        rem = rem // d5
        off += idx * s5
    if NDIM > 6:
        idx = rem % d6
        rem = rem // d6
        off += idx * s6
    if NDIM > 7:
        idx = rem % d7
        rem = rem // d7
        off += idx * s7

    _div_apply(p, off, mask, inv32, inv_hi, inv_lo, F64)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
_INV_CACHE = {}
_LAUNCH_CACHE = {}
_DEVICE = [-1]


def _to_float_scalar(scalar):
    if isinstance(scalar, (int, float)) and not isinstance(scalar, torch.Tensor):
        return float(scalar)
    if isinstance(scalar, torch.Tensor):
        if scalar.numel() != 1:
            raise RuntimeError("foreach_div_scalar_: scalar must be one value")
        return float(scalar.item())
    if hasattr(scalar, "item"):
        return float(scalar.item())
    return float(scalar)


def _inverses(s):
    """opmath reciprocals: fp32 value plus the fp64 value's two 32-bit halves."""
    got = _INV_CACHE.get(s)
    if got is not None:
        return got
    if _np is not None:
        with _np.errstate(all="ignore"):
            inv32 = float(_np.float32(1.0) / _np.float32(s))
            bits = int(_np.float64(1.0 / _np.float64(s)).view(_np.int64))
        bits &= 0xFFFFFFFFFFFFFFFF
    else:  # pragma: no cover
        import struct

        s32 = struct.unpack("<f", struct.pack("<f", s))[0]
        inv32 = struct.unpack("<f", struct.pack("<f", 1.0 / s32))[0]
        bits = struct.unpack("<Q", struct.pack("<d", 1.0 / s))[0]
    hi = bits >> 32
    lo = bits & 0xFFFFFFFF
    if hi >= 0x80000000:
        hi -= 0x100000000
    if lo >= 0x80000000:
        lo -= 0x100000000
    out = (inv32, hi, lo)
    if len(_INV_CACHE) > 256:
        _INV_CACHE.clear()
    _INV_CACHE[s] = out
    return out


def _launch_list(kernel_args, grid0, key):
    """Launch the cached compiled list kernel for one specialization key.

    The packed launcher is called directly with the arguments Triton's own
    launch path would pass: grid, stream, function, (no global/profile scratch),
    packed metadata, no launch metadata and no hooks, then the kernel arguments
    followed by the constexpr parameters BLOCK/LIST_LEN/F64, which this backend
    parses as trailing scalars.
    """
    if _DEVICE[0] < 0:
        _DEVICE[0] = _driver.active.get_current_device()
    dev = _DEVICE[0]
    stream = _driver.active.get_current_stream(dev)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        dtype, count = key[0], key[1]
        f64 = dtype == torch.float64
        ck = _div_list_kernel.warmup(
            *kernel_args,
            BLOCK=_BLOCK,
            LIST_LEN=count,
            F64=f64,
            num_warps=_WARPS,
            grid=(1,),
        )
        if hasattr(ck, "result"):
            ck = ck.result()
        ck._init_handles()
        tail = (_BLOCK, count, 1 if f64 else 0)
        run_obj = ck.run
        if run_obj.global_scratch_size or run_obj.profile_scratch_size:
            # needs the wrapper's scratch allocation; use the regular JIT path
            _div_list_kernel[(grid0,)](
                *kernel_args,
                BLOCK=_BLOCK,
                LIST_LEN=count,
                F64=f64,
                num_warps=_WARPS,
            )
            return
        entry = (run_obj.launch, ck.function, ck.packed_metadata, tail)
        if len(_LAUNCH_CACHE) > 64:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = entry
    launcher, function, packed_metadata, tail = entry
    launcher(
        grid0,
        1,
        1,
        stream,
        function,
        None,
        None,
        packed_metadata,
        None,
        None,
        None,
        *kernel_args,
        *tail,
    )


def _div_one_(t, inv32, inv_hi, inv_lo):
    dt = t.dtype
    if dt not in _FLOAT_SET:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type " + str(dt)
        )
    n = int(t.numel())
    if n == 0:
        return
    f64 = dt == torch.float64
    if t.is_contiguous():
        _div_contig_kernel[(-(-n // _BLOCK),)](
            t,
            n,
            inv32,
            inv_hi,
            inv_lo,
            BLOCK=_BLOCK,
            F64=f64,
            I64=n > _MAX_ELEMS,
            num_warps=_WARPS,
        )
        return
    dims = []
    for sz, st in zip(t.shape, t.stride()):
        sz = int(sz)
        st = int(st)
        if sz <= 1 or st == 0:
            # size-1 dims are free; an expanded dim only needs one visit
            continue
        dims.append((st, sz))
    dims = sorted(dims)
    merged = []
    for st, sz in dims:
        if merged and merged[-1][0] * merged[-1][1] == st:
            merged[-1][1] *= sz
        else:
            merged.append([st, sz])
    if not merged:
        merged = [[0, 1]]
    if len(merged) > _MAX_DIMS:
        raise NotImplementedError("foreach_div_scalar_: rank too large")
    n_covered = 1
    max_off = 0
    for st, sz in merged:
        n_covered *= sz
        max_off += (sz - 1) * st
    ndim = len(merged)
    shapes = [p[1] for p in merged] + [1] * (_MAX_DIMS - ndim)
    strides = [p[0] for p in merged] + [0] * (_MAX_DIMS - ndim)
    _div_strided_kernel[(-(-n_covered // _BLOCK),)](
        t,
        n_covered,
        inv32,
        inv_hi,
        inv_lo,
        shapes[0],
        shapes[1],
        shapes[2],
        shapes[3],
        shapes[4],
        shapes[5],
        shapes[6],
        shapes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        BLOCK=_BLOCK,
        NDIM=ndim,
        F64=f64,
        I64=max_off > _MAX_ELEMS,
        num_warps=_WARPS,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def foreach_div_scalar_(self=None, scalar=None, *args, **kwargs):
    if self is None:
        self = kwargs.pop("tensors", kwargs.pop("input", None))
    if scalar is None and args:
        scalar = args[0]
    if scalar is None:
        scalar = kwargs.pop("value", None)
    if self is None or scalar is None:
        raise TypeError("run(self, scalar) requires a tensor list and a scalar")

    got = _INV_CACHE.get(scalar) if type(scalar) is float else None
    if got is None:
        got = _inverses(_to_float_scalar(scalar))
    inv32, inv_hi, inv_lo = got

    if isinstance(self, torch.Tensor):
        _div_one_(self, inv32, inv_hi, inv_lo)
        return None
    tensors = self if type(self) is list else list(self)
    cnt = len(tensors)

    if 1 < cnt <= _FUSED_MAX:
        head = tensors[0]
        dt = head.dtype
        if dt in _FLOAT_SET and head.is_contiguous() and head.is_cuda:
            sizes = []
            nblocks = 0
            aligned = head.data_ptr()
            cls = 0
            ok = True
            for t in tensors:
                if t.dtype is not dt or not t.is_contiguous() or not t.is_cuda:
                    ok = False
                    break
                m = t.numel()
                if m == 0 or m > _MAX_ELEMS:
                    ok = False
                    break
                aligned |= t.data_ptr()
                sizes.append(m)
                nblocks += -(-m // _BLOCK)
                # element-count class Triton specializes on
                cls = (cls << 2) | (2 if (m & 15) == 0 else (1 if m == 1 else 0))
            if ok and not (aligned & 15):
                while len(sizes) < _FUSED_MAX:
                    sizes.append(0)
                pads = tensors
                if cnt < _FUSED_MAX:
                    pads = tensors + [head] * (_FUSED_MAX - cnt)
                _launch_list(
                    (
                        pads[0],
                        pads[1],
                        pads[2],
                        pads[3],
                        sizes[0],
                        sizes[1],
                        sizes[2],
                        sizes[3],
                        inv32,
                        inv_hi,
                        inv_lo,
                    ),
                    nblocks,
                    (dt, cnt, cls),
                )
                return None

    for t in tensors:
        _div_one_(t, inv32, inv_hi, inv_lo)
    return None
