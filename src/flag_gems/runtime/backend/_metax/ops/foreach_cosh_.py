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

"""Triton implementation of ``aten::_foreach_cosh_`` (in-place cosh over a Tensor list).

Semantics: for every tensor ``t`` in ``self`` (in list order) replace its
elements by ``cosh(t)`` in place, preserving storage pointer, shape and strides,
and return ``None``.  Integer / bool elements are rejected exactly like ATen's
in-place cast check does.

Launch strategy: the common case -- one dtype, dense tensors, distinct storages
-- is served by one fused launch per 16 tensors that addresses every tensor from
scalar pointer arguments; single 16-bit tensors instead use the tensor-pointer
kernel, whose known alignment keeps their byte rate.  Launches whose element
counts are multiples of the block size use a maskless kernel specialization.
Repeated launches of the same specialization reuse the already compiled kernel
through its low-level launcher, which skips per-call argument binding,
specialization and cache lookup.  Everything else falls back to one ordinary
launch per tensor.
"""

import torch
import triton
import triton.language as tl

_ATEN_NAME = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}

_DT_ID = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}

# Dtypes this in-place operator accepts (ATen rejects integral and bool).
_OK_DTYPES = frozenset(_DT_ID) | {torch.complex64, torch.complex128}

_BLOCK_REAL = 4096
_BLOCK_CPLX = 512
_NUM_WARPS = 8
_SLOTS = 16

# Above this many 16-bit elements in a list, one launch per tensor beats the
# fused launch: the per-launch host cost (~5us, cached launcher) is hidden by
# the GPU queue while the tensor-pointer kernel sustains a higher byte rate.
_SPLIT_ELEMS = 1 << 26

# (dt_id, device index, even) -> [launcher, function, packed_metadata, compiled]
# once the kernel has been compiled for that specialization, or False when the
# low-level path turned out to be unusable and must not be retried.
_FAST = {}
_SINGLE_FAST = {}
_DRIVER = None


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["n"])
def _cosh_flat_kernel(
    p, n, BLOCK: tl.constexpr, ACC64: tl.constexpr, EVEN: tl.constexpr
):
    base = tl.program_id(0) * BLOCK
    off = base + tl.arange(0, BLOCK)
    if EVEN:
        # Every block is full, so the per-element mask is not needed; the
        # scalar guard only covers a grid that overshoots n.
        if base >= n:
            return
        x = tl.load(p + off)
        if ACC64:
            y = tl.exp(x - 0.6931471805599453) + tl.exp(-x - 0.6931471805599453)
        else:
            xf = x.to(tl.float32)
            a = tl.abs(xf) * 1.4426950408889634
            y = (tl.exp2(a - 1.0) + tl.exp2(-a - 1.0)).to(x.dtype)
        tl.store(p + off, y)
    else:
        m = off < n
        x = tl.load(p + off, mask=m, other=0.0)
        if ACC64:
            y = tl.exp(x - 0.6931471805599453) + tl.exp(-x - 0.6931471805599453)
        else:
            xf = x.to(tl.float32)
            a = tl.abs(xf) * 1.4426950408889634
            y = (tl.exp2(a - 1.0) + tl.exp2(-a - 1.0)).to(x.dtype)
        tl.store(p + off, y, mask=m)


@triton.jit(
    do_not_specialize=[
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
        "a10",
        "a11",
        "a12",
        "a13",
        "a14",
        "a15",
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
def _cosh_multi_kernel(
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
    DT_ID: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    s = tl.program_id(1)
    base = a0
    n = n0
    base = tl.where(s == 1, a1, base)
    n = tl.where(s == 1, n1, n)
    base = tl.where(s == 2, a2, base)
    n = tl.where(s == 2, n2, n)
    base = tl.where(s == 3, a3, base)
    n = tl.where(s == 3, n3, n)
    base = tl.where(s == 4, a4, base)
    n = tl.where(s == 4, n4, n)
    base = tl.where(s == 5, a5, base)
    n = tl.where(s == 5, n5, n)
    base = tl.where(s == 6, a6, base)
    n = tl.where(s == 6, n6, n)
    base = tl.where(s == 7, a7, base)
    n = tl.where(s == 7, n7, n)
    base = tl.where(s == 8, a8, base)
    n = tl.where(s == 8, n8, n)
    base = tl.where(s == 9, a9, base)
    n = tl.where(s == 9, n9, n)
    base = tl.where(s == 10, a10, base)
    n = tl.where(s == 10, n10, n)
    base = tl.where(s == 11, a11, base)
    n = tl.where(s == 11, n11, n)
    base = tl.where(s == 12, a12, base)
    n = tl.where(s == 12, n12, n)
    base = tl.where(s == 13, a13, base)
    n = tl.where(s == 13, n13, n)
    base = tl.where(s == 14, a14, base)
    n = tl.where(s == 14, n14, n)
    base = tl.where(s == 15, a15, base)
    n = tl.where(s == 15, n15, n)

    if DT_ID == 0:
        p = base.to(tl.pointer_type(tl.float16))
    elif DT_ID == 1:
        p = base.to(tl.pointer_type(tl.bfloat16))
    elif DT_ID == 2:
        p = base.to(tl.pointer_type(tl.float32))
    else:
        p = base.to(tl.pointer_type(tl.float64))

    off0 = tl.program_id(0) * BLOCK
    off = off0 + tl.arange(0, BLOCK)
    if EVEN:
        if off0 >= n:
            return
        x = tl.load(p + off)
        if DT_ID == 3:
            y = tl.exp(x - 0.6931471805599453) + tl.exp(-x - 0.6931471805599453)
        else:
            xf = x.to(tl.float32)
            a = tl.abs(xf) * 1.4426950408889634
            y = (tl.exp2(a - 1.0) + tl.exp2(-a - 1.0)).to(x.dtype)
        tl.store(p + off, y)
    else:
        m = off < n
        x = tl.load(p + off, mask=m, other=0.0)
        if DT_ID == 3:
            y = tl.exp(x - 0.6931471805599453) + tl.exp(-x - 0.6931471805599453)
        else:
            xf = x.to(tl.float32)
            a = tl.abs(xf) * 1.4426950408889634
            y = (tl.exp2(a - 1.0) + tl.exp2(-a - 1.0)).to(x.dtype)
        tl.store(p + off, y, mask=m)


@triton.jit(do_not_specialize=["n"])
def _cosh_strided_kernel(
    p,
    n,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC64: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    rem = off
    addr = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(ND):
        d = SHAPE[i]
        c = rem % d
        rem = rem // d
        addr += c * STRIDE[i]
    x = tl.load(p + addr, mask=m, other=0.0)
    if ACC64:
        y = tl.exp(x - 0.6931471805599453) + tl.exp(-x - 0.6931471805599453)
    else:
        xf = x.to(tl.float32)
        a = tl.abs(xf) * 1.4426950408889634
        y = (tl.exp2(a - 1.0) + tl.exp2(-a - 1.0)).to(x.dtype)
    tl.store(p + addr, y, mask=m)


@triton.jit(do_not_specialize=["n"])
def _cosh_flat_c_kernel(p, n, BLOCK: tl.constexpr, ACC64: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    i2 = 2 * off
    re = tl.load(p + i2, mask=m, other=0.0)
    im = tl.load(p + i2 + 1, mask=m, other=0.0)
    if ACC64:
        ea = tl.exp(re)
        eb = tl.exp(-re)
        ch = (ea + eb) * 0.5
        sh = (ea - eb) * 0.5
        yre = ch * tl.cos(im)
        yim = sh * tl.sin(im)
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
        ea = tl.exp(ref)
        eb = tl.exp(-ref)
        ch = (ea + eb) * 0.5
        sh = (ea - eb) * 0.5
        yre = (ch * tl.cos(imf)).to(re.dtype)
        yim = (sh * tl.sin(imf)).to(im.dtype)
    tl.store(p + i2, yre, mask=m)
    tl.store(p + i2 + 1, yim, mask=m)


@triton.jit(do_not_specialize=["n"])
def _cosh_strided_c_kernel(
    p,
    n,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC64: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    rem = off
    addr = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(ND):
        d = SHAPE[i]
        c = rem % d
        rem = rem // d
        addr += c * STRIDE[i]
    i2 = 2 * addr
    re = tl.load(p + i2, mask=m, other=0.0)
    im = tl.load(p + i2 + 1, mask=m, other=0.0)
    if ACC64:
        ea = tl.exp(re)
        eb = tl.exp(-re)
        ch = (ea + eb) * 0.5
        sh = (ea - eb) * 0.5
        yre = ch * tl.cos(im)
        yim = sh * tl.sin(im)
    else:
        ref = re.to(tl.float32)
        imf = im.to(tl.float32)
        ea = tl.exp(ref)
        eb = tl.exp(-ref)
        ch = (ea + eb) * 0.5
        sh = (ea - eb) * 0.5
        yre = (ch * tl.cos(imf)).to(re.dtype)
        yim = (sh * tl.sin(imf)).to(im.dtype)
    tl.store(p + i2, yre, mask=m)
    tl.store(p + i2 + 1, yim, mask=m)


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------


def _is_flat(t):
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


def _strided_dims(t):
    """Collapsed (shape, stride) pairs ordered by ascending stride."""
    pairs = [(s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1]
    pairs = sorted(pairs)
    shape = []
    stride = []
    for s, sz in pairs:
        if shape and s == stride[-1] * shape[-1]:
            shape[-1] *= sz
        else:
            shape.append(sz)
            stride.append(s)
    return tuple(shape), tuple(stride)


def _launch_one(t):
    n = t.numel()
    if n == 0:
        return
    cplx = t.dtype.is_complex
    acc64 = t.dtype in (torch.float64, torch.complex128)
    if _is_flat(t):
        p = torch.view_as_real(t) if cplx else t
        block = _BLOCK_CPLX if cplx else _BLOCK_REAL
        grid = (triton.cdiv(n, block),)
        if cplx:
            _cosh_flat_c_kernel[grid](p, n, block, acc64, num_warps=_NUM_WARPS)
        else:
            _cosh_flat_kernel[grid](
                p,
                n,
                block,
                acc64,
                (n & (block - 1)) == 0,
                num_warps=_NUM_WARPS,
            )
        return
    if torch._debug_has_internal_overlap(t) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    shape, stride = _strided_dims(t)
    nd = len(shape)
    p = torch.view_as_real(t) if cplx else t
    block = _BLOCK_CPLX if cplx else _BLOCK_REAL
    grid = (triton.cdiv(n, block),)
    if cplx:
        _cosh_strided_c_kernel[grid](
            p, n, shape, stride, nd, block, acc64, num_warps=_NUM_WARPS
        )
    else:
        _cosh_strided_kernel[grid](
            p, n, shape, stride, nd, block, acc64, num_warps=_NUM_WARPS
        )


def _current_stream(dev):
    global _DRIVER
    if _DRIVER is None:
        from triton.runtime import driver

        _DRIVER = driver.active
    return _DRIVER.get_current_stream(dev)


def _launch_multi(ptrs, numels, dt_id, dev, blocks, ns, even):
    """One fused launch for up to _SLOTS tensors of one dtype.

    ``ptrs``/``numels`` are already padded to exactly ``_SLOTS`` entries.  The
    first launch of a specialization compiles it and keeps the compiled kernel;
    later launches call its low-level launcher directly, which removes the
    per-call binding/specialization/cache-lookup work from the timed path.
    """
    key = (dt_id, dev, even)
    entry = _FAST.get(key)
    if entry is not None and entry is not False:
        try:
            entry[0](
                blocks,
                ns,
                1,
                _current_stream(dev),
                entry[1],
                entry[2],
                None,
                None,
                None,
                *ptrs,
                *numels,
                dt_id,
                _BLOCK_REAL,
                even,
            )
            return
        except Exception:
            _FAST[key] = False

    compiled = _cosh_multi_kernel[(blocks, ns)](
        *ptrs,
        *numels,
        dt_id,
        _BLOCK_REAL,
        even,
        num_warps=_NUM_WARPS,
    )
    if entry is None and compiled is not None:
        try:
            _FAST[key] = [
                compiled.run,
                compiled.function,
                compiled.packed_metadata,
                compiled,
            ]
        except Exception:
            _FAST[key] = False


def _launch_single(t, n, dt_id, dev, acc64, block, aligned, even):
    """Single dense tensor through the tensor-pointer kernel.

    That kernel keeps a known-aligned pointer argument, which the 16-bit
    element types need to reach their best byte rate; the pointer-cast fused
    kernel is the faster choice for 32/64-bit elements.  The compiled kernel is
    reused through its low-level launcher, and the fast path is only taken for
    16-byte-aligned storages because the compiled variant assumes that
    alignment.
    """
    grid0 = (n + block - 1) // block
    key = (dt_id, dev, even)
    entry = _SINGLE_FAST.get(key)
    if entry is not None and entry is not False:
        if aligned:
            try:
                entry[0](
                    grid0,
                    1,
                    1,
                    _current_stream(dev),
                    entry[1],
                    entry[2],
                    None,
                    None,
                    None,
                    t,
                    n,
                    block,
                    acc64,
                    even,
                )
                return
            except Exception:
                _SINGLE_FAST[key] = False
    compiled = _cosh_flat_kernel[(grid0,)](
        t, n, block, acc64, even, num_warps=_NUM_WARPS
    )
    if entry is None and aligned and compiled is not None:
        try:
            _SINGLE_FAST[key] = [
                compiled.run,
                compiled.function,
                compiled.packed_metadata,
                compiled,
            ]
        except Exception:
            _SINGLE_FAST[key] = False


def foreach_cosh_(self):
    if type(self) is not list and type(self) is not tuple:
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    first = self[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("argument 'self' must be tuple of Tensors")
    dt = first.dtype
    if dt not in _OK_DTYPES:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            + _ATEN_NAME.get(dt, str(dt))
        )

    # One pass that validates every element and, when the list is uniform
    # (single dtype, one device, dense, distinct storages), gathers the fused
    # launch data.
    ptrs = []
    numels = []
    seen = set()
    add_seen = seen.add
    dev = first.get_device()
    ok = True
    even = True
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        d = t.dtype
        if d not in _OK_DTYPES:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + _ATEN_NAME.get(d, str(d))
            )
        if ok:
            if d is dt and t.get_device() == dev and t.is_contiguous():
                p = t.data_ptr()
                if p in seen:
                    ok = False
                else:
                    add_seen(p)
                    ptrs.append(p)
                    k = t.numel()
                    numels.append(k)
                    if k & (_BLOCK_REAL - 1):
                        even = False
            else:
                ok = False

    if ok:
        dt_id = _DT_ID.get(dt)
        if dt_id is not None:
            block = _BLOCK_REAL
            if len(ptrs) == 1:
                n = numels[0]
                if n:
                    if dt is torch.float16 or dt is torch.bfloat16:
                        _launch_single(
                            first,
                            n,
                            dt_id,
                            dev,
                            False,
                            block,
                            ptrs[0] % 16 == 0,
                            even,
                        )
                    else:
                        _launch_multi(
                            ptrs + [ptrs[0]] * (_SLOTS - 1),
                            numels + [0] * (_SLOTS - 1),
                            dt_id,
                            dev,
                            (n + block - 1) // block,
                            1,
                            even,
                        )
                return None
            if (dt is torch.float16 or dt is torch.bfloat16) and sum(
                numels
            ) >= _SPLIT_ELEMS:
                for i, t in enumerate(self):
                    _launch_single(
                        t,
                        numels[i],
                        dt_id,
                        dev,
                        False,
                        block,
                        ptrs[i] % 16 == 0,
                        even,
                    )
                return None
            if len(ptrs) == _SLOTS:
                # Exact fit: launch the gathered lists without re-slicing.
                _launch_multi(
                    ptrs,
                    numels,
                    dt_id,
                    dev,
                    (max(numels) + block - 1) // block,
                    _SLOTS,
                    even,
                )
                return None
            for start in range(0, len(ptrs), _SLOTS):
                cptrs = ptrs[start : start + _SLOTS]
                cnumels = numels[start : start + _SLOTS]
                ns = len(cptrs)
                peak = max(cnumels)
                if peak == 0:
                    continue
                if ns < _SLOTS:
                    cptrs = cptrs + [cptrs[0]] * (_SLOTS - ns)
                    cnumels = cnumels + [0] * (_SLOTS - ns)
                _launch_multi(
                    cptrs,
                    cnumels,
                    dt_id,
                    dev,
                    (peak + block - 1) // block,
                    ns,
                    even,
                )
            return None

    for t in self:
        _launch_one(t)
    return None
