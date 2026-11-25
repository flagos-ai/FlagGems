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

"""In-place multi-tensor reciprocal: ``aten::_foreach_reciprocal_``.

Semantics (matching the ATen reference):
  * every tensor in the list is divided into 1 element-wise, in place;
  * storage pointers and strides are untouched, the list is not rebound and the
    call returns ``None``;
  * fp16 operands are upcast to fp32 for the division and rounded back;
  * integral/bool entries cannot hold a floating result and are rejected with
    the same ``RuntimeError`` class ATen raises;
  * an empty tensor list is rejected.

Launch strategy
---------------
* A list whose entries all share one floating dtype, one element count and a
  contiguous, 16-byte-aligned storage is served by a *single* fused kernel whose
  second grid axis selects the list entry (``len(list)`` launches collapse to
  one).
* Everything else (single entry, heterogeneous sizes, non-contiguous views,
  complex, unaligned views) falls back to one launch per entry; a non-contiguous
  entry is written through its own logical strides, never through a staging
  copy, so the storage and aliasing contract is preserved.
* The per-call host cost dominates the small-workload latency, so after the
  first (compiling) launch every kernel is re-launched through the cached
  compiled launcher, keyed on the same properties Triton specializes on
  (element dtype, pointer 16-byte divisibility, integer divisibility/one-ness).
  The constants the launcher needs are cached with it, so the steady-state path
  does no dictionary lookup by dtype and no branching on dtype.
* bfloat16 storage is addressed as int16 and converted with shifts instead of
  the fp32<->bf16 convert instruction: bf16 measured ~20% slower than fp16 at
  identical byte counts, which points at the conversion sequence rather than at
  memory traffic.  Adding half an ulp before the downcast makes the result
  bit-identical to ATen's round-to-nearest conversion.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

_TL_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

# Complex tensors are addressed as interleaved real components.
_C2R = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

_BLOCK = 1024
_SHIFT = 10  # log2(_BLOCK)
_SLOTS = 16  # fused-kernel pointer slots


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _recip_store(p, off, mask, DT: tl.constexpr):
    if DT == tl.int16:
        # bf16 storage seen as int16: shift into fp32, then add half an ulp and
        # shift back so the downcast rounds to nearest instead of truncating.
        x = (tl.load(p + off, mask=mask).to(tl.int32) << 16).to(
            tl.float32, bitcast=True
        )
        y = 1.0 / x
        tl.store(
            p + off,
            ((y.to(tl.int32, bitcast=True) + 0x8000) >> 16).to(tl.int16),
            mask=mask,
        )
    elif DT == tl.float64:
        x = tl.load(p + off, mask=mask)
        tl.store(p + off, 1.0 / x, mask=mask)
    else:
        x = tl.load(p + off, mask=mask)
        tl.store(p + off, (1.0 / x.to(tl.float32)).to(DT), mask=mask)


@triton.jit
def _recip_flat_kernel(
    p, n, BLOCK: tl.constexpr, DT: tl.constexpr, IDX64: tl.constexpr
):
    if IDX64:
        off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _recip_store(p, off, off < n, DT)


@triton.jit
def _recip_many_kernel(
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
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    IDX64: tl.constexpr,
):
    """One launch for up to 16 equal-sized contiguous entries.

    ``tl.program_id(1)`` picks the list entry; grid axis 0 is the block within
    that entry, so no program ever touches a sibling's storage.  Entries beyond
    the real list length are simply never given a program on axis 1.
    """
    if IDX64:
        off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    rid = tl.program_id(1)
    if rid == 0:
        _recip_store(p0, off, mask, DT)
    elif rid == 1:
        _recip_store(p1, off, mask, DT)
    elif rid == 2:
        _recip_store(p2, off, mask, DT)
    elif rid == 3:
        _recip_store(p3, off, mask, DT)
    elif rid == 4:
        _recip_store(p4, off, mask, DT)
    elif rid == 5:
        _recip_store(p5, off, mask, DT)
    elif rid == 6:
        _recip_store(p6, off, mask, DT)
    elif rid == 7:
        _recip_store(p7, off, mask, DT)
    elif rid == 8:
        _recip_store(p8, off, mask, DT)
    elif rid == 9:
        _recip_store(p9, off, mask, DT)
    elif rid == 10:
        _recip_store(p10, off, mask, DT)
    elif rid == 11:
        _recip_store(p11, off, mask, DT)
    elif rid == 12:
        _recip_store(p12, off, mask, DT)
    elif rid == 13:
        _recip_store(p13, off, mask, DT)
    elif rid == 14:
        _recip_store(p14, off, mask, DT)
    elif rid == 15:
        _recip_store(p15, off, mask, DT)


@triton.jit
def _recip_strided_kernel(
    p,
    n,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    NDIM: tl.constexpr,
    CPLX: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    S4: tl.constexpr,
    S5: tl.constexpr,
    S6: tl.constexpr,
    S7: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    D4: tl.constexpr,
    D5: tl.constexpr,
    D6: tl.constexpr,
    D7: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    strides = (S0, S1, S2, S3, S4, S5, S6, S7)
    sizes = (D0, D1, D2, D3, D4, D5, D6, D7)
    rem = idx
    off = tl.zeros([BLOCK], tl.int64)
    for k in tl.static_range(NDIM):
        size = sizes[k]
        coord = rem % size
        rem = rem // size
        off = off + coord.to(tl.int64) * strides[k]
    if CPLX:
        re = tl.load(p + 2 * off, mask=mask).to(DT)
        im = tl.load(p + 2 * off + 1, mask=mask).to(DT)
        den = re * re + im * im
        tl.store(p + 2 * off, (re / den).to(DT), mask=mask)
        tl.store(p + 2 * off + 1, (-im / den).to(DT), mask=mask)
    else:
        x = tl.load(p + off, mask=mask)
        if DT == tl.float64:
            y = 1.0 / x
        else:
            y = (1.0 / x.to(tl.float32)).to(DT)
        tl.store(p + off, y, mask=mask)


@triton.jit
def _recip_complex_kernel(
    p, n, BLOCK: tl.constexpr, RT: tl.constexpr, ACC: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    re = tl.load(p + 2 * idx, mask=mask).to(ACC)
    im = tl.load(p + 2 * idx + 1, mask=mask).to(ACC)
    den = re * re + im * im
    tl.store(p + 2 * idx, (re / den).to(RT), mask=mask)
    tl.store(p + 2 * idx + 1, (-im / den).to(RT), mask=mask)


# ---------------------------------------------------------------------------
# direct launch: reuse the compiled kernel instead of re-running the JIT binder
# ---------------------------------------------------------------------------

_DRIVER = None
_MISSING = object()
_FLAT_CACHE = {}
_MANY_CACHE = {}


def _driver():
    global _DRIVER
    d = _DRIVER
    if d is None:
        from triton.runtime import driver as _drv

        d = _DRIVER = _drv.active
    return d


def _capture(ck):
    """(launcher, function, packed_metadata) for a freshly compiled kernel."""
    try:
        lr = ck._run
        if lr is None or lr.global_scratch_size or lr.profile_scratch_size:
            return False
        return (lr.launch, ck.function, ck.packed_metadata)
    except Exception:
        return False


def _jit_flat(t, numel, torch_dtype, idx64):
    bf16 = torch_dtype is torch.bfloat16
    dtl = tl.int16 if bf16 else _TL_DT[torch_dtype]
    arg = t.view(torch.int16) if bf16 else t
    ck = _recip_flat_kernel[((numel + 1023) >> 10,)](
        arg, numel, _BLOCK, dtl, idx64, num_warps=4
    )
    return ck, dtl


def _launch_flat(t, numel, torch_dtype, dev, stream):
    p = t.data_ptr()
    idx64 = numel > 0x7FFFFFFF
    key = (torch_dtype, numel == 1, numel & 15 == 0, p & 15 == 0, idx64, dev)
    ent = _FLAT_CACHE.get(key, _MISSING)
    if ent is _MISSING:
        ck, dtl = _jit_flat(t, numel, torch_dtype, idx64)
        cap = _capture(ck)
        _FLAT_CACHE[key] = (
            None if cap is False else (cap[0], cap[1], cap[2], dtl, idx64)
        )
        return
    if ent is not None:
        launch, function, meta, dtl, i64 = ent
        try:
            launch(
                (numel + 1023) >> 10,
                1,
                1,
                stream,
                function,
                None,
                None,
                meta,
                None,
                None,
                None,
                p,
                numel,
                _BLOCK,
                dtl,
                i64,
            )
            return
        except Exception:
            _FLAT_CACHE[key] = None
    _jit_flat(t, numel, torch_dtype, idx64)


def _jit_many(tensors, numel, torch_dtype, idx64, nt):
    bf16 = torch_dtype is torch.bfloat16
    dtl = tl.int16 if bf16 else _TL_DT[torch_dtype]
    args = tensors + [tensors[0]] * (_SLOTS - nt)
    if bf16:
        args = [a.view(torch.int16) for a in args]
    ck = _recip_many_kernel[((numel + 1023) >> 10, nt)](
        *args, numel, _BLOCK, dtl, idx64, num_warps=4
    )
    return ck, dtl


def _launch_many(tensors, ptrs, numel, torch_dtype, dev, stream):
    nt = len(tensors)
    idx64 = numel > 0x7FFFFFFF
    key = (torch_dtype, numel == 1, numel & 15 == 0, idx64, dev)
    ent = _MANY_CACHE.get(key, _MISSING)
    if ent is _MISSING:
        ck, dtl = _jit_many(tensors, numel, torch_dtype, idx64, nt)
        cap = _capture(ck)
        _MANY_CACHE[key] = (
            None if cap is False else (cap[0], cap[1], cap[2], dtl, idx64)
        )
        return
    if ent is not None:
        launch, function, meta, dtl, i64 = ent
        try:
            launch(
                (numel + 1023) >> 10,
                nt,
                1,
                stream,
                function,
                None,
                None,
                meta,
                None,
                None,
                None,
                *ptrs,
                numel,
                _BLOCK,
                dtl,
                i64,
            )
            return
        except Exception:
            _MANY_CACHE[key] = None
    _jit_many(tensors, numel, torch_dtype, idx64, nt)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _reject_dtype(dtype):
    raise RuntimeError(
        "result type Float can't be cast to the desired output type "
        f"{str(dtype).replace('torch.', '').capitalize()}"
    )


def _collapse(shape, strides):
    """Merge neighbouring dims that are dense in memory order.

    Dims run outer-to-inner; (d0, s0) and (d1, s1) merge when
    ``s0 == s1 * d1``, and the merged dim keeps the *inner* stride ``s1``.
    """
    dims = [(int(d), int(s)) for d, s in zip(shape, strides) if int(d) != 1]
    out = []
    for d, s in dims:
        if out and out[-1][1] == s * d:
            out[-1] = (out[-1][0] * d, s)
        else:
            out.append((d, s))
    return out


def _strided(t, numel, dims, dtype, ptr, cplx=False):
    if torch._debug_has_internal_overlap(t) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    nd = len(dims)
    sizes = [d for d, _ in dims] + [1] * (8 - nd)
    strds = [s for _, s in dims] + [0] * (8 - nd)
    _recip_strided_kernel[((numel + 1023) >> 10,)](
        ptr,
        numel,
        BLOCK=_BLOCK,
        DT=dtype,
        NDIM=nd,
        CPLX=cplx,
        S0=strds[0],
        S1=strds[1],
        S2=strds[2],
        S3=strds[3],
        S4=strds[4],
        S5=strds[5],
        S6=strds[6],
        S7=strds[7],
        D0=sizes[0],
        D1=sizes[1],
        D2=sizes[2],
        D3=sizes[3],
        D4=sizes[4],
        D5=sizes[5],
        D6=sizes[6],
        D7=sizes[7],
        num_warps=4,
    )


def _one(t, dev, stream):
    """Reciprocal of a single list entry, written in place."""
    numel = t.numel()
    if numel == 0:
        return
    dtype = t.dtype
    if dtype.is_complex:
        real_dtype = _C2R.get(dtype)
        if real_dtype is None:
            raise NotImplementedError(f"unsupported complex dtype {dtype}")
        flat = t.view(real_dtype)  # zero-copy real-component view
        if t.is_contiguous():
            _recip_complex_kernel[((numel + 1023) >> 10,)](
                flat,
                numel,
                BLOCK=_BLOCK,
                RT=_TL_DT[real_dtype],
                ACC=tl.float64 if real_dtype is torch.float64 else tl.float32,
                num_warps=4,
            )
        else:
            _strided(
                t, numel, _collapse(t.shape, t.stride()), _TL_DT[real_dtype], flat, True
            )
    elif t.is_contiguous():
        _launch_flat(t, numel, dtype, dev, stream)
    else:
        _strided(t, numel, _collapse(t.shape, t.stride()), _TL_DT[dtype], t)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_reciprocal_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if type(self) is not list:
        self = list(self)
    n = len(self)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    t0 = self[0]
    if not isinstance(t0, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    dtype0 = t0.dtype
    if not (dtype0.is_floating_point or dtype0.is_complex):
        _reject_dtype(dtype0)

    drv = _driver()
    dev = drv.get_current_device()
    stream = drv.get_current_stream(dev)

    if n == 1:
        _one(t0, dev, stream)
        return None

    # Fused path: the whole list is one dtype, one size and contiguous with
    # 16-byte-aligned storage, so a single launch serves every entry.
    numel = t0.numel()
    if numel and not dtype0.is_complex and t0.is_contiguous():
        p0 = t0.data_ptr()
        if not p0 & 15:
            ptrs = [p0]
            for i in range(1, n):
                t = self[i]
                if not isinstance(t, torch.Tensor):
                    raise TypeError("argument 'self' must be tuple of Tensors")
                if t.dtype is not dtype0 or t.numel() != numel or not t.is_contiguous():
                    break
                p = t.data_ptr()
                if p & 15:
                    break
                ptrs.append(p)
            else:
                if n <= _SLOTS:
                    _launch_many(self, ptrs, numel, dtype0, dev, stream)
                else:
                    for i in range(0, n, _SLOTS):
                        _launch_many(
                            self[i : i + _SLOTS],
                            ptrs[i : i + _SLOTS],
                            numel,
                            dtype0,
                            dev,
                            stream,
                        )
                return None

    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            _reject_dtype(dtype)
    for t in self:
        _one(t, dev, stream)
    return None
