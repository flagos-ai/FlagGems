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

"""In-place elementwise greater-than.

Implements aten::greater_.Tensor(Tensor(a!) self, Tensor other) and
aten::greater_.Scalar(Tensor(a!) self, Scalar other):
    self[m] = cast(self[m] > other[m_bcast], self.dtype)
and returns the same Tensor object (same storage / strides / offset).

The hot path (same-shape contiguous operands whose element count is an exact
multiple of the tile) launches a mask-free kernel; after the first launch for a
given (device, dtype, tile, numel) the compiled kernel handle is reused, so the
per-call Python work of Triton's JIT dispatch (option/knob handling, argument
binding, specialization key, cache lookup, global-value checks) is paid once
rather than on every call.  Every other case falls back to the ordinary
kernel[grid](...) path with the general strided/broadcast/scalar kernels.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---- comparison ("compute") dtype codes -------------------------------------
_CD_F32 = 0
_CD_F64 = 1
_CD_F16 = 2
_CD_BF16 = 3
_CD_I64 = 4
_CD_I32 = 5
_CD_I16 = 6
_CD_I8 = 7
_CD_U8 = 8
_CD_BOOL = 9

_CD_BY_DTYPE = {
    torch.float32: _CD_F32,
    torch.float64: _CD_F64,
    torch.float16: _CD_F16,
    torch.bfloat16: _CD_BF16,
    torch.int64: _CD_I64,
    torch.int32: _CD_I32,
    torch.int16: _CD_I16,
    torch.int8: _CD_I8,
    torch.uint8: _CD_U8,
    torch.bool: _CD_BOOL,
}

# ---- "other" operand modes --------------------------------------------------
_M_SAME = 0  # tensor, linear offsets (flat path) or strided offsets
_M_TAIL = 1  # contiguous tensor covering the trailing dims -> offs % nb
_M_FSCALAR = 2  # python float scalar
_M_ISCALAR = 3  # python int scalar
_M_ONE = 4  # single-element tensor -> one scalar load

_MAX_DIMS = 8
_BLOCK = 1024  # 4 elements/thread for 4-byte dtypes (16 B per thread)
_BLOCK16 = 2048  # 8 elements/thread for 2-byte dtypes (16 B per thread)
_WARPS = 4
_PROMOTE_CACHE = {}
_PREPARED = {}  # (device, dtype, tile, numel) -> prepared launch callable


@triton.jit
def _cmp_gt(a, b, CD: tl.constexpr):
    """Ordered IEEE greater-than evaluated in the promoted common dtype."""
    if CD == 1:
        c = a.to(tl.float64) > b.to(tl.float64)
    elif CD == 2:
        c = a.to(tl.float16) > b.to(tl.float16)
    elif CD == 3:
        c = a.to(tl.bfloat16) > b.to(tl.bfloat16)
    elif CD == 4:
        c = a.to(tl.int64) > b.to(tl.int64)
    elif CD == 5:
        c = a.to(tl.int32) > b.to(tl.int32)
    elif CD == 6:
        c = a.to(tl.int16) > b.to(tl.int16)
    elif CD == 7:
        c = a.to(tl.int8) > b.to(tl.int8)
    elif CD == 8:
        c = a.to(tl.uint8) > b.to(tl.uint8)
    elif CD == 9:
        c = a.to(tl.int32) > b.to(tl.int32)
    else:
        c = a.to(tl.float32) > b.to(tl.float32)
    return c


@triton.jit
def _greater_flat_even_kernel(a_ptr, b_ptr, CD: tl.constexpr, BLOCK: tl.constexpr):
    """Full-tile path: numel is an exact multiple of BLOCK, no mask, no n."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = _cmp_gt(a, b, CD)
    tl.store(a_ptr + offs, tl.where(c, 1, 0).to(a.dtype))


@triton.jit
def _greater_flat_kernel(
    a_ptr,
    b_ptr,
    bs_f: tl.float64,
    bs_i: tl.int64,
    n,
    nb,
    MODE: tl.constexpr,
    CD: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        mask = None
    else:
        mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    if MODE == 0:
        b = tl.load(b_ptr + offs, mask=mask)
    elif MODE == 1:
        b = tl.load(b_ptr + (offs % nb), mask=mask)
    elif MODE == 2:
        b = bs_f
    elif MODE == 3:
        b = bs_i
    else:
        b = tl.load(b_ptr)
    c = _cmp_gt(a, b, CD)
    tl.store(a_ptr + offs, tl.where(c, 1, 0).to(a.dtype), mask=mask)


@triton.jit
def _greater_strided_kernel(
    a_ptr,
    b_ptr,
    bs_f: tl.float64,
    bs_i: tl.int64,
    n,
    nb,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    sa0,
    sa1,
    sa2,
    sa3,
    sa4,
    sa5,
    sa6,
    sa7,
    sb0,
    sb1,
    sb2,
    sb3,
    sb4,
    sb5,
    sb6,
    sb7,
    NDIM: tl.constexpr,
    MODE: tl.constexpr,
    CD: tl.constexpr,
    A_CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    dims = (d0, d1, d2, d3, d4, d5, d6, d7)
    sas = (sa0, sa1, sa2, sa3, sa4, sa5, sa6, sa7)
    sbs = (sb0, sb1, sb2, sb3, sb4, sb5, sb6, sb7)

    rem = offs
    if A_CONTIG:
        a_off = offs
    else:
        a_off = tl.zeros([BLOCK], tl.int64)
    b_off = tl.zeros([BLOCK], tl.int64)
    for i in tl.static_range(0, NDIM):
        sz = dims[i].to(tl.int64)
        idx = rem % sz
        rem = rem // sz
        if not A_CONTIG:
            a_off += idx * sas[i].to(tl.int64)
        if MODE == 0:
            b_off += idx * sbs[i].to(tl.int64)

    a = tl.load(a_ptr + a_off, mask=mask)
    if MODE == 0:
        b = tl.load(b_ptr + b_off, mask=mask)
    elif MODE == 2:
        b = bs_f
    elif MODE == 3:
        b = bs_i
    else:
        b = tl.load(b_ptr)
    c = _cmp_gt(a, b, CD)
    tl.store(a_ptr + a_off, tl.where(c, 1, 0).to(a.dtype), mask=mask)


# ---- host side helpers ------------------------------------------------------
def _broadcast_ok(sa, sb):
    """True when broadcast(sa, sb) == sa (PyTorch in-place requirement)."""
    if len(sb) > len(sa):
        return False
    off = len(sa) - len(sb)
    for i, db in enumerate(sb):
        if db != 1 and db != sa[off + i]:
            return False
    return True


def _broadcast_shape(sa, sb):
    nd = max(len(sa), len(sb))
    out = []
    for i in range(nd):
        da = sa[len(sa) - nd + i] if i >= nd - len(sa) else 1
        db = sb[len(sb) - nd + i] if i >= nd - len(sb) else 1
        out.append(max(da, db))
    return out


def _collapse(a, b):
    """(size, self_stride, other_stride) list, inner dim first, size-1 dropped."""
    ash = a.shape
    ast = a.stride()
    bsh = b.shape
    bst = b.stride()
    na = len(ash)
    nb = len(bsh)
    dims = []
    for i in range(na - 1, -1, -1):
        size = ash[i]
        s_stride = ast[i]
        j = i - (na - nb)
        if j >= 0 and bsh[j] != 1:
            o_stride = bst[j]
        else:
            o_stride = 0
        if size == 1:
            continue
        # merge with the previously appended (inner) dim when the layout is dense
        if dims:
            psize, pss, pos = dims[-1]
            if pss == s_stride * size and pos == o_stride * size:
                dims[-1] = (psize * size, pss, pos)
                continue
        dims.append((size, s_stride, o_stride))
    if not dims:
        dims.append((1, 0, 0))
    return dims


def _launch_flat(a, b, mode, cd, n, nb, bs_f, bs_i, block, warps):
    wide = n > 0x7FFFFFFF
    even = (n % block) == 0
    grid = ((n + block - 1) // block,)
    _greater_flat_kernel[grid](
        a,
        b,
        bs_f,
        bs_i,
        n,
        nb,
        MODE=mode,
        CD=cd,
        EVEN=even,
        WIDE=wide,
        BLOCK=block,
        num_warps=warps,
    )


def _launch_strided(a, b, mode, cd, n, bs_f, bs_i, block, warps):
    dims = _collapse(a, b)
    ndim = len(dims)
    if ndim > _MAX_DIMS:
        raise RuntimeError("greater_: rank after collapse exceeds supported maximum")
    sizes = [1] * _MAX_DIMS
    sas = [0] * _MAX_DIMS
    sbs = [0] * _MAX_DIMS
    for i, (sz, ss, os) in enumerate(dims):
        sizes[i] = sz
        sas[i] = ss
        sbs[i] = os
    grid = ((n + block - 1) // block,)
    _greater_strided_kernel[grid](
        a,
        b,
        bs_f,
        bs_i,
        n,
        1,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        sas[0],
        sas[1],
        sas[2],
        sas[3],
        sas[4],
        sas[5],
        sas[6],
        sas[7],
        sbs[0],
        sbs[1],
        sbs[2],
        sbs[3],
        sbs[4],
        sbs[5],
        sbs[6],
        sbs[7],
        NDIM=ndim,
        MODE=mode,
        CD=cd,
        A_CONTIG=a.is_contiguous(),
        BLOCK=block,
        num_warps=warps,
    )


def _run_tensor(a, b):
    sa = a.shape
    sb = b.shape
    if not _broadcast_ok(sa, sb):
        raise RuntimeError(
            "output with shape %s doesn't match the broadcast shape %s"
            % (list(sa), _broadcast_shape(sa, sb))
        )
    n = a.numel()
    if n == 0:
        return
    adt = a.dtype
    bdt = b.dtype
    if adt is bdt:
        cd = _CD_BY_DTYPE.get(adt)
    else:
        key = (adt, bdt)
        cd = _PROMOTE_CACHE.get(key, -1)
        if cd == -1:
            cd = _CD_BY_DTYPE.get(torch.promote_types(adt, bdt))
            _PROMOTE_CACHE[key] = cd
    if cd is None:
        cd = _CD_F32
    if adt is torch.float16 or adt is torch.bfloat16:
        block = _BLOCK16
    else:
        block = _BLOCK
    warps = _WARPS
    bnb = b.numel()
    if a.is_contiguous():
        if bnb == 1:
            _launch_flat(a, b, _M_ONE, cd, n, 1, 0.0, 0, block, warps)
            return
        if b.is_contiguous():
            if b.shape == sa:
                _launch_flat(a, b, _M_SAME, cd, n, bnb, 0.0, 0, block, warps)
                return
            eff = list(sb)
            while eff and eff[0] == 1:
                eff.pop(0)
            if (
                eff
                and len(eff) <= len(sa)
                and tuple(eff) == tuple(sa[len(sa) - len(eff) :])
            ):
                _launch_flat(a, b, _M_TAIL, cd, n, bnb, 0.0, 0, block, warps)
                return
    if bnb == 1:
        _launch_strided(a, b, _M_ONE, cd, n, 0.0, 0, block, warps)
    else:
        _launch_strided(a, b, _M_SAME, cd, n, 0.0, 0, block, warps)


def _run_scalar(a, other):
    n = a.numel()
    if n == 0:
        return
    dt = a.dtype
    is_int = isinstance(other, int)
    if is_int:
        mode = _M_ISCALAR
        bs_f = 0.0
        bs_i = int(other)
        cd = _CD_BY_DTYPE.get(dt)
        if cd is None:
            cd = _CD_F32
    else:
        mode = _M_FSCALAR
        bs_f = float(other)
        bs_i = 0
        if dt.is_floating_point:
            cd = _CD_BY_DTYPE.get(dt)
            if cd is None:
                cd = _CD_F32
        else:
            # float scalar with an integer/bool tensor promotes to the default float
            cd = _CD_BY_DTYPE.get(torch.get_default_dtype(), _CD_F32)
    if cd == _CD_BF16:
        # fp64 scalar -> bf16 conversion fails in the backend lowering pipeline
        # ("PassManager::run failed"); comparing in fp32 is exact for every
        # bf16-representable value and matches the wrapped-scalar rule.
        cd = _CD_F32
    if dt is torch.float16 or dt is torch.bfloat16:
        block = _BLOCK16
    else:
        block = _BLOCK
    warps = _WARPS
    if a.is_contiguous():
        _launch_flat(a, a, mode, cd, n, 1, bs_f, bs_i, block, warps)
    else:
        _launch_strided(a, a, mode, cd, n, bs_f, bs_i, block, warps)


def greater_(self, other):
    logger.debug("GEMS_HYGON GREATER_")
    a = self
    if type(other) is torch.Tensor:
        n = a.numel()
        if n:
            if other.shape == a.shape and a.is_contiguous() and other.is_contiguous():
                dt = a.dtype
                if dt is other.dtype:
                    cd = _CD_BY_DTYPE.get(dt)
                    if cd is not None:
                        if dt is torch.float16 or dt is torch.bfloat16:
                            block = _BLOCK16
                        else:
                            block = _BLOCK
                        if not (n & (block - 1)):
                            key = (a.device, dt, block, n)
                            runner = _PREPARED.get(key)
                            if runner is not None:
                                runner(a, other, cd, block)
                            else:
                                g = n // block
                                compiled = _greater_flat_even_kernel[(g, 1, 1)](
                                    a, other, CD=cd, BLOCK=block
                                )
                                _PREPARED[key] = compiled[(g, 1, 1)]
                            return a
        _run_tensor(a, other)
    elif isinstance(other, torch.Tensor):
        _run_tensor(a, other)
    else:
        _run_scalar(a, other)
    return a
