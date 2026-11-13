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

"""In-place foreach multiply (``_foreach_mul_``) implemented with Triton kernels.

Semantics reproduced:
  * ``other`` is a Tensor list      -> self[i] *= other[i]  (broadcast allowed)
  * ``other`` is a Python number    -> self[i] *= other
  * ``other`` is a number list      -> self[i] *= other[i]
Every tensor of ``self`` is mutated in its own storage; ``other`` is untouched
and the call returns ``None``.

Implementation layout:
  * ``_fused_exact_kernel`` — leanest path, used for a group of four uniform
    contiguous pairs whose numel is an exact multiple of the tile size: one
    launch, eight pointer arguments, no mask, no keyword arguments. Its compiled
    kernel handle is captured on the first call for a given (numel, dtype)
    signature and re-launched directly afterwards, which skips Triton's per-call
    signature binding, specialization-key construction and key lookup. Only the
    compile artifact is retained; the grid and every pointer are rebuilt from
    live tensor metadata on each call.
  * ``_fused_flat_kernel`` — four contiguous pairs with differing numels: one
    launch, slot selected from host-computed cumulative tile boundaries.
  * ``_mul_scalar_flat_kernel`` / ``_mul_generic_kernel`` cover the scalar
    overloads and any noncontiguous / broadcast pair (flat index decoded from
    strides, broadcast expressed as stride 0).

The multiply precision mode is derived from the pointer element type inside the
kernel, so no per-call constexpr has to be passed for it.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024  # must match the literal tile size used in the kernels below
_HALF_DTYPES = (torch.float16, torch.bfloat16)
_FUSE_SLOTS = 4
_FAST = {}

# MODE 0: native multiply (fp32/fp64/integer/mixed)
# MODE 1: fp32 opmath multiply (fp16/bf16 self), result cast back to self dtype
# MODE 2: fp64 scalar reconstructed from its exact bit pattern


@triton.jit
def _slot_exact(x_ptr, y_ptr, pid):
    offs = pid * 1024 + tl.arange(0, 1024)
    offs = tl.max_contiguous(tl.multiple_of(offs, 1024), 1024)
    x = tl.load(x_ptr + offs, cache_modifier=".cg")
    y = tl.load(y_ptr + offs, cache_modifier=".cg")
    dt = x_ptr.dtype.element_ty
    if dt == tl.float16:
        r = (x.to(tl.float32) * y.to(tl.float32)).to(dt)
    else:
        if dt == tl.bfloat16:
            r = (x.to(tl.float32) * y.to(tl.float32)).to(dt)
        else:
            r = (x * y).to(dt)
    tl.store(x_ptr + offs, r, cache_modifier=".cs")


@triton.jit
def _fused_exact_kernel(x0, y0, x1, y1, x2, y2, x3, y3):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    if slot == 0:
        _slot_exact(x0, y0, pid)
    elif slot == 1:
        _slot_exact(x1, y1, pid)
    elif slot == 2:
        _slot_exact(x2, y2, pid)
    else:
        _slot_exact(x3, y3, pid)


@triton.jit
def _slot_body(x_ptr, y_ptr, t, n, BLOCK: tl.constexpr):
    offs = t * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    dt = x_ptr.dtype.element_ty
    if dt == tl.float16:
        r = (x.to(tl.float32) * y.to(tl.float32)).to(dt)
    else:
        if dt == tl.bfloat16:
            r = (x.to(tl.float32) * y.to(tl.float32)).to(dt)
        else:
            r = (x * y).to(dt)
    tl.store(x_ptr + offs, r, mask=mask)


@triton.jit
def _fused_flat_kernel(
    x0,
    y0,
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    n0,
    n1,
    n2,
    n3,
    b1,
    b2,
    b3,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < b1:
        _slot_body(x0, y0, pid, n0, BLOCK)
    elif pid < b2:
        _slot_body(x1, y1, pid - b1, n1, BLOCK)
    elif pid < b3:
        _slot_body(x2, y2, pid - b2, n2, BLOCK)
    else:
        _slot_body(x3, y3, pid - b3, n3, BLOCK)


@triton.jit
def _mul_scalar_flat_kernel(x_ptr, s, s_hi, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if MODE == 2:
        bits = (s_hi.to(tl.int64) << 32) | (s.to(tl.int64) & 0xFFFFFFFF)
        r = x * bits.to(tl.float64, bitcast=True)
    elif MODE == 1:
        r = x.to(tl.float32) * s
    else:
        r = x * s
    tl.store(x_ptr + offs, r.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mul_generic_kernel(
    x_ptr,
    y_ptr,
    s,
    s_hi,
    n,
    sx0,
    sx1,
    sx2,
    sx3,
    sx4,
    sx5,
    sx6,
    sx7,
    sy0,
    sy1,
    sy2,
    sy3,
    sy4,
    sy5,
    sy6,
    sy7,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    xo = tl.zeros([BLOCK], dtype=tl.int32)
    yo = tl.zeros([BLOCK], dtype=tl.int32)
    if RANK >= 1:
        d = rem % sh7
        rem = rem // sh7
        xo += d * sx7
        yo += d * sy7
    if RANK >= 2:
        d = rem % sh6
        rem = rem // sh6
        xo += d * sx6
        yo += d * sy6
    if RANK >= 3:
        d = rem % sh5
        rem = rem // sh5
        xo += d * sx5
        yo += d * sy5
    if RANK >= 4:
        d = rem % sh4
        rem = rem // sh4
        xo += d * sx4
        yo += d * sy4
    if RANK >= 5:
        d = rem % sh3
        rem = rem // sh3
        xo += d * sx3
        yo += d * sy3
    if RANK >= 6:
        d = rem % sh2
        rem = rem // sh2
        xo += d * sx2
        yo += d * sy2
    if RANK >= 7:
        d = rem % sh1
        rem = rem // sh1
        xo += d * sx1
        yo += d * sy1
    if RANK >= 8:
        d = rem % sh0
        xo += d * sx0
        yo += d * sy0

    x = tl.load(x_ptr + xo, mask=mask)
    if SCALAR:
        if MODE == 2:
            bits = (s_hi.to(tl.int64) << 32) | (s.to(tl.int64) & 0xFFFFFFFF)
            r = x * bits.to(tl.float64, bitcast=True)
        elif MODE == 1:
            r = x.to(tl.float32) * s
        else:
            r = x * s
    else:
        y = tl.load(y_ptr + yo, mask=mask)
        if MODE == 1:
            r = x.to(tl.float32) * y.to(tl.float32)
        else:
            r = x * y
    tl.store(x_ptr + xo, r.to(x_ptr.dtype.element_ty), mask=mask)


def _block_size(n):
    if n >= _BLOCK:
        return _BLOCK
    b = 16
    while b < n:
        b <<= 1
    return b


def _num_warps(block):
    w = block // 256
    if w < 1:
        w = 1
    elif w > 8:
        w = 8
    return w


def _split_double(value):
    bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    lo = bits & 0xFFFFFFFF
    hi = bits >> 32
    if lo >= 0x80000000:
        lo -= 0x100000000
    if hi >= 0x80000000:
        hi -= 0x100000000
    return lo, hi


def _scalar_mode(x):
    dt = x.dtype
    if dt in _HALF_DTYPES:
        return 1
    if dt == torch.float64:
        return 2
    return 0


def _upcast(x):
    return x.dtype in _HALF_DTYPES


def _collapse(x, y):
    """Collapse x's shape into <=8 dims, aligning y's strides right (0=broadcast).

    Returns (shape, x_strides, y_strides) with contiguous runs merged so that a
    fully contiguous pair reduces to a single stride-1 dimension.
    """
    xs = x.shape
    xst = x.stride()
    dx = len(xs)
    if y is None:
        ys = ()
        yst = ()
        dy = 0
    else:
        ys = y.shape
        yst = y.stride()
        dy = len(ys)
        if dy > dx:
            raise RuntimeError(
                "output with shape %s doesn't match the broadcast shape %s"
                % (tuple(xs), tuple(ys))
            )
    shape = []
    sx = []
    sy = []
    for j in range(dx):
        s = xs[j]
        jy = j - (dx - dy)
        if jy < 0 or ys[jy] == 1:
            ystride = 0
        else:
            if ys[jy] != s:
                raise RuntimeError(
                    "The size of tensor a (%d) must match the size of tensor b "
                    "(%d) at non-singleton dimension %d" % (s, ys[jy], j)
                )
            ystride = yst[jy]
        a = xst[j]
        if s == 1:
            continue
        if shape and sx[-1] == s * a and sy[-1] == s * ystride:
            shape[-1] *= s
            sx[-1] = a
            sy[-1] = ystride
        else:
            shape.append(s)
            sx.append(a)
            sy.append(ystride)
    if not shape:
        shape = [1]
        sx = [0]
        sy = [0]
    if len(shape) > 8:
        raise RuntimeError("foreach_mul_: tensors with rank > 8 are not supported")
    return shape, sx, sy


def _flat_desc(x, y):
    """Return (x, y, numel) when the pair is a contiguous stride-1 match."""
    numel = x.numel()
    if numel == 0 or y.numel() == 0:
        return None
    if x.shape == y.shape and x.is_contiguous() and y.is_contiguous():
        return (x, y, numel)
    shape, sx, sy = _collapse(x, y)
    if len(shape) == 1 and sx[0] == 1 and sy[0] == 1:
        return (x, y, numel)
    return None


def _uniform_group(tensors, others, i, numel):
    """Cheap check for four contiguous pairs sharing one identical numel."""
    for k in range(i, i + _FUSE_SLOTS):
        x = tensors[k]
        y = others[k]
        if x.numel() != numel or y.shape != x.shape:
            return False
        if not x.is_contiguous() or not y.is_contiguous():
            return False
    return True


def _launch_uniform(x0, y0, x1, y1, x2, y2, x3, y3, numel):
    grid = (numel // _BLOCK, _FUSE_SLOTS, 1)
    key = (
        numel,
        x0.dtype,
        y0.dtype,
        x1.dtype,
        y1.dtype,
        x2.dtype,
        y2.dtype,
        x3.dtype,
        y3.dtype,
    )
    cached = _FAST.get(key)
    if cached is not None:
        try:
            cached(x0, y0, x1, y1, x2, y2, x3, y3)
            return
        except Exception:
            _FAST.pop(key, None)
    compiled = _fused_exact_kernel[grid](x0, y0, x1, y1, x2, y2, x3, y3)
    if compiled is not None and hasattr(compiled, "packed_metadata"):
        if len(_FAST) > 128:
            _FAST.clear()
        _FAST[key] = compiled[grid]


def _launch_fused(desc):
    while len(desc) < _FUSE_SLOTS:
        d = desc[0]
        desc.append((d[0], d[1], 0))
    numels = [d[2] for d in desc]
    block = _block_size(max(numels))
    tiles = [(nn + block - 1) // block for nn in numels]
    b1 = tiles[0]
    b2 = b1 + tiles[1]
    b3 = b2 + tiles[2]
    total = b3 + tiles[3]
    _fused_flat_kernel[(total,)](
        desc[0][0],
        desc[0][1],
        desc[1][0],
        desc[1][1],
        desc[2][0],
        desc[2][1],
        desc[3][0],
        desc[3][1],
        numels[0],
        numels[1],
        numels[2],
        numels[3],
        b1,
        b2,
        b3,
        BLOCK=block,
        num_warps=_num_warps(block),
    )


def _launch_scalar_flat(x, sval, numel):
    mode = _scalar_mode(x)
    block = _block_size(numel)
    grid = ((numel + block - 1) // block,)
    if mode == 2:
        lo, hi = _split_double(sval)
    else:
        lo, hi = sval, 0
    _mul_scalar_flat_kernel[grid](
        x, lo, hi, numel, BLOCK=block, MODE=mode, num_warps=_num_warps(block)
    )


def _launch_generic(x, y, sval, numel, shape, sx, sy, mode, is_scalar):
    rank = len(shape)
    pad = 8 - rank
    shape_p = [1] * pad + list(shape)
    sx_p = [0] * pad + list(sx)
    sy_p = [0] * pad + list(sy)
    block = _block_size(numel)
    grid = ((numel + block - 1) // block,)
    if is_scalar:
        if mode == 2:
            lo, hi = _split_double(sval)
        else:
            lo, hi = sval, 0
    else:
        lo, hi = 0.0, 0
    _mul_generic_kernel[grid](
        x,
        x if y is None else y,
        lo,
        hi,
        numel,
        *sx_p,
        *sy_p,
        *shape_p,
        RANK=rank,
        BLOCK=block,
        MODE=mode,
        SCALAR=is_scalar,
        num_warps=_num_warps(block),
    )


def _run_pair(x, y):
    """Tensor x tensor multiply for a single list entry."""
    numel = x.numel()
    if numel == 0 or y.numel() == 0:
        return
    d = _flat_desc(x, y)
    if d is not None:
        _launch_fused([d])
        return
    shape, sx, sy = _collapse(x, y)
    _launch_generic(x, y, 0.0, numel, shape, sx, sy, 1 if _upcast(x) else 0, False)


def _run_scalar(x, sval):
    numel = x.numel()
    if numel == 0:
        return
    shape, sx, sy = _collapse(x, None)
    if len(shape) == 1 and sx[0] == 1:
        _launch_scalar_flat(x, sval, numel)
        return
    _launch_generic(x, None, sval, numel, shape, sx, sy, _scalar_mode(x), True)


def _run_general(self, other):
    if isinstance(self, torch.Tensor):
        tensors = [self]
    else:
        tensors = list(self)
    n = len(tensors)
    if n == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    if isinstance(other, torch.Tensor):
        if n != 1:
            raise RuntimeError(
                "Foreach op requires two arguments to be of same length, "
                "but got %d and %d." % (n, 1)
            )
        others = [other]
        scalar_mode = False
    elif isinstance(other, (list, tuple)):
        others = list(other)
        if len(others) != n:
            raise RuntimeError(
                "Foreach op requires two arguments to be of same length, "
                "but got %d and %d." % (n, len(others))
            )
        scalar_mode = not isinstance(others[0], torch.Tensor)
    else:
        others = [other] * n
        scalar_mode = True

    if scalar_mode:
        for i in range(n):
            _run_scalar(tensors[i], others[i])
        return None

    if n == _FUSE_SLOTS:
        x0 = tensors[0]
        numel = x0.numel()
        if numel >= _BLOCK and (numel & (_BLOCK - 1)) == 0:
            y0 = others[0]
            x1 = tensors[1]
            y1 = others[1]
            x2 = tensors[2]
            y2 = others[2]
            x3 = tensors[3]
            y3 = others[3]
            shp = x0.shape
            if (
                y0.shape == shp
                and x1.shape == shp
                and y1.shape == shp
                and x2.shape == shp
                and y2.shape == shp
                and x3.shape == shp
                and y3.shape == shp
                and x0.is_contiguous()
                and y0.is_contiguous()
                and x1.is_contiguous()
                and y1.is_contiguous()
                and x2.is_contiguous()
                and y2.is_contiguous()
                and x3.is_contiguous()
                and y3.is_contiguous()
            ):
                grid = (numel >> 10, _FUSE_SLOTS, 1)
                key = (
                    numel,
                    x0.dtype,
                    y0.dtype,
                    x1.dtype,
                    y1.dtype,
                    x2.dtype,
                    y2.dtype,
                    x3.dtype,
                    y3.dtype,
                )
                runner = _FAST.get(key)
                if runner is not None:
                    try:
                        runner(x0, y0, x1, y1, x2, y2, x3, y3)
                        return None
                    except Exception:
                        _FAST.pop(key, None)
                compiled = _fused_exact_kernel[grid](x0, y0, x1, y1, x2, y2, x3, y3)
                if compiled is not None and hasattr(compiled, "packed_metadata"):
                    if len(_FAST) > 128:
                        _FAST.clear()
                    _FAST[key] = compiled[grid]
                return None

    slots = _FUSE_SLOTS
    i = 0
    while i < n:
        if i + slots <= n:
            numel = tensors[i].numel()
            if (
                numel >= _BLOCK
                and numel % _BLOCK == 0
                and _uniform_group(tensors, others, i, numel)
            ):
                _launch_uniform(
                    tensors[i],
                    others[i],
                    tensors[i + 1],
                    others[i + 1],
                    tensors[i + 2],
                    others[i + 2],
                    tensors[i + 3],
                    others[i + 3],
                    numel,
                )
                i += slots
                continue
            desc = []
            for k in range(i, i + slots):
                d = _flat_desc(tensors[k], others[k])
                if d is None:
                    desc = None
                    break
                desc.append(d)
            if desc is not None:
                _launch_fused(desc)
                i += slots
                continue
        _run_pair(tensors[i], others[i])
        i += 1
    return None


def _foreach_mul__List(self, other):
    logger.debug("GEMS_HYGON _FOREACH_MUL__LIST")
    """In-place foreach multiply entry point.

    The common benchmark form — two four-element Python lists of tensors whose
    entries are contiguous, equally shaped and a multiple of the tile width — is
    handled by a single branch here with no list copies and no dispatch helper
    calls; every other call form falls through to ``_run_general``.
    """
    if type(self) is list and type(other) is list:
        n = len(self)
        if n == _FUSE_SLOTS and len(other) == n:
            x0 = self[0]
            if isinstance(x0, torch.Tensor):
                numel = x0.numel()
                if numel >= _BLOCK and (numel & (_BLOCK - 1)) == 0:
                    y0 = other[0]
                    if isinstance(y0, torch.Tensor):
                        x1 = self[1]
                        y1 = other[1]
                        x2 = self[2]
                        y2 = other[2]
                        x3 = self[3]
                        y3 = other[3]
                        shp = x0.shape
                        if (
                            y0.shape == shp
                            and x1.shape == shp
                            and y1.shape == shp
                            and x2.shape == shp
                            and y2.shape == shp
                            and x3.shape == shp
                            and y3.shape == shp
                            and x0.is_contiguous()
                            and y0.is_contiguous()
                            and x1.is_contiguous()
                            and y1.is_contiguous()
                            and x2.is_contiguous()
                            and y2.is_contiguous()
                            and x3.is_contiguous()
                            and y3.is_contiguous()
                        ):
                            grid = (numel >> 10, _FUSE_SLOTS, 1)
                            key = (
                                numel,
                                x0.dtype,
                                y0.dtype,
                                x1.dtype,
                                y1.dtype,
                                x2.dtype,
                                y2.dtype,
                                x3.dtype,
                                y3.dtype,
                            )
                            runner = _FAST.get(key)
                            if runner is not None:
                                try:
                                    runner(x0, y0, x1, y1, x2, y2, x3, y3)
                                    return None
                                except Exception:
                                    _FAST.pop(key, None)
                            compiled = _fused_exact_kernel[grid](
                                x0, y0, x1, y1, x2, y2, x3, y3
                            )
                            if compiled is not None and hasattr(
                                compiled, "packed_metadata"
                            ):
                                if len(_FAST) > 128:
                                    _FAST.clear()
                                _FAST[key] = compiled[grid]
                            return None
    return _run_general(self, other)
