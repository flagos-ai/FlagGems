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

"""foreach_abs: elementwise abs over a list of CUDA tensors.

Baseline (V1): one Triton launch per input tensor.
  * dense (contiguous, or dense after a zero-copy memory-order permute) -> flat kernel
  * otherwise -> rank-specialized strided-decode kernel writing a dense buffer
  * complex inputs -> magnitude of interleaved (re, im) pairs, real output dtype
Outputs follow ATen's empty_like(preserve_format) layout: contiguous inputs get
contiguous outputs, transposed/dense-permuted inputs keep their stride pattern,
genuinely non-dense inputs get a dense buffer (values only, strides differ).
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8

_UNSIGNED = tuple(
    d
    for d in (
        getattr(torch, "uint8", None),
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    )
    if d is not None
)

_SCALAR_NAMES = (
    ["n"]
    + ["d%d" % i for i in range(_MAX_RANK)]
    + ["e%d" % i for i in range(_MAX_RANK)]
)

_MAX_SLOTS = 16
_MULTI_SCALAR_NAMES = ["n%d" % i for i in range(_MAX_SLOTS)] + ["total", "dt_id"]


_FLAT_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_warps=8),
]


@triton.autotune(configs=_FLAT_CONFIGS, key=["n", "dt_id"])
@triton.jit(do_not_specialize=_SCALAR_NAMES + ["dt_id"])
def _abs_flat_kernel(x_ptr, y_ptr, n, dt_id, BLOCK: tl.constexpr, DO_ABS: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    # Unmasked (fully vectorized) path for every complete tile; only the final
    # partial tile pays for predication.
    if start + BLOCK <= n:
        v = tl.load(x_ptr + offs)
        if DO_ABS:
            v = tl.abs(v)
        tl.store(y_ptr + offs, v)
    else:
        m = offs < n
        v = tl.load(x_ptr + offs, mask=m)
        if DO_ABS:
            v = tl.abs(v)
        tl.store(y_ptr + offs, v, mask=m)


@triton.jit
def _abs_slot(x_ptr, y_ptr, n, pid, BLOCK: tl.constexpr, DO_ABS: tl.constexpr):
    """One tile of one list element; inlined into the fused multi-slot kernel."""
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    if start < n:
        if start + BLOCK <= n:
            v = tl.load(x_ptr + offs)
            if DO_ABS:
                v = tl.abs(v)
            tl.store(y_ptr + offs, v)
        else:
            m = offs < n
            v = tl.load(x_ptr + offs, mask=m)
            if DO_ABS:
                v = tl.abs(v)
            tl.store(y_ptr + offs, v, mask=m)


@triton.autotune(configs=_FLAT_CONFIGS, key=["total", "dt_id", "S"])
@triton.jit(do_not_specialize=_MULTI_SCALAR_NAMES)
def _abs_multi_kernel(
    x0,
    y0,
    n0,
    x1,
    y1,
    n1,
    x2,
    y2,
    n2,
    x3,
    y3,
    n3,
    x4,
    y4,
    n4,
    x5,
    y5,
    n5,
    x6,
    y6,
    n6,
    x7,
    y7,
    n7,
    x8,
    y8,
    n8,
    x9,
    y9,
    n9,
    x10,
    y10,
    n10,
    x11,
    y11,
    n11,
    x12,
    y12,
    n12,
    x13,
    y13,
    n13,
    x14,
    y14,
    n14,
    x15,
    y15,
    n15,
    total,
    dt_id,
    BLOCK: tl.constexpr,
    S: tl.constexpr,
    DO_ABS: tl.constexpr,
):
    """All same-dtype dense list elements in one launch.

    program_id(1) selects the list element, program_id(0) the tile inside it, so
    every CTA does exactly one tile of exactly one element (no serial chain).
    """
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    if S > 0:
        if slot == 0:
            _abs_slot(x0, y0, n0, pid, BLOCK, DO_ABS)
    if S > 1:
        if slot == 1:
            _abs_slot(x1, y1, n1, pid, BLOCK, DO_ABS)
    if S > 2:
        if slot == 2:
            _abs_slot(x2, y2, n2, pid, BLOCK, DO_ABS)
    if S > 3:
        if slot == 3:
            _abs_slot(x3, y3, n3, pid, BLOCK, DO_ABS)
    if S > 4:
        if slot == 4:
            _abs_slot(x4, y4, n4, pid, BLOCK, DO_ABS)
    if S > 5:
        if slot == 5:
            _abs_slot(x5, y5, n5, pid, BLOCK, DO_ABS)
    if S > 6:
        if slot == 6:
            _abs_slot(x6, y6, n6, pid, BLOCK, DO_ABS)
    if S > 7:
        if slot == 7:
            _abs_slot(x7, y7, n7, pid, BLOCK, DO_ABS)
    if S > 8:
        if slot == 8:
            _abs_slot(x8, y8, n8, pid, BLOCK, DO_ABS)
    if S > 9:
        if slot == 9:
            _abs_slot(x9, y9, n9, pid, BLOCK, DO_ABS)
    if S > 10:
        if slot == 10:
            _abs_slot(x10, y10, n10, pid, BLOCK, DO_ABS)
    if S > 11:
        if slot == 11:
            _abs_slot(x11, y11, n11, pid, BLOCK, DO_ABS)
    if S > 12:
        if slot == 12:
            _abs_slot(x12, y12, n12, pid, BLOCK, DO_ABS)
    if S > 13:
        if slot == 13:
            _abs_slot(x13, y13, n13, pid, BLOCK, DO_ABS)
    if S > 14:
        if slot == 14:
            _abs_slot(x14, y14, n14, pid, BLOCK, DO_ABS)
    if S > 15:
        if slot == 15:
            _abs_slot(x15, y15, n15, pid, BLOCK, DO_ABS)


@triton.jit(do_not_specialize=_SCALAR_NAMES)
def _abs_strided_kernel(
    x_ptr,
    y_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    DO_ABS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rem = offs.to(tl.int64)
    off = tl.zeros((BLOCK,), tl.int64)
    if RANK >= 1:
        off += (rem % d0) * e0
        rem = rem // d0
    if RANK >= 2:
        off += (rem % d1) * e1
        rem = rem // d1
    if RANK >= 3:
        off += (rem % d2) * e2
        rem = rem // d2
    if RANK >= 4:
        off += (rem % d3) * e3
        rem = rem // d3
    if RANK >= 5:
        off += (rem % d4) * e4
        rem = rem // d4
    if RANK >= 6:
        off += (rem % d5) * e5
        rem = rem // d5
    if RANK >= 7:
        off += (rem % d6) * e6
        rem = rem // d6
    if RANK >= 8:
        off += (rem % d7) * e7
        rem = rem // d7
    v = tl.load(x_ptr + off, mask=m)
    if DO_ABS:
        v = tl.abs(v)
    tl.store(y_ptr + offs, v, mask=m)


@triton.jit(do_not_specialize=["n"])
def _abs_complex_flat_kernel(xp, y_ptr, n, BLOCK: tl.constexpr, IS_F64: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if IS_F64:
        x_ptr = tl.cast(xp, tl.pointer_type(tl.float64))
    else:
        x_ptr = tl.cast(xp, tl.pointer_type(tl.float32))
    re = tl.load(x_ptr + 2 * offs, mask=m)
    im = tl.load(x_ptr + 2 * offs + 1, mask=m)
    tl.store(y_ptr + offs, tl.sqrt(re * re + im * im), mask=m)


@triton.jit(do_not_specialize=_SCALAR_NAMES)
def _abs_complex_strided_kernel(
    xp,
    y_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    IS_F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    rem = offs.to(tl.int64)
    off = tl.zeros((BLOCK,), tl.int64)
    if RANK >= 1:
        off += (rem % d0) * e0
        rem = rem // d0
    if RANK >= 2:
        off += (rem % d1) * e1
        rem = rem // d1
    if RANK >= 3:
        off += (rem % d2) * e2
        rem = rem // d2
    if RANK >= 4:
        off += (rem % d3) * e3
        rem = rem // d3
    if RANK >= 5:
        off += (rem % d4) * e4
        rem = rem // d4
    if RANK >= 6:
        off += (rem % d5) * e5
        rem = rem // d5
    if RANK >= 7:
        off += (rem % d6) * e6
        rem = rem // d6
    if RANK >= 8:
        off += (rem % d7) * e7
        rem = rem // d7
    if IS_F64:
        x_ptr = tl.cast(xp, tl.pointer_type(tl.float64))
    else:
        x_ptr = tl.cast(xp, tl.pointer_type(tl.float32))
    re = tl.load(x_ptr + 2 * off, mask=m)
    im = tl.load(x_ptr + 2 * off + 1, mask=m)
    tl.store(y_ptr + offs, tl.sqrt(re * re + im * im), mask=m)


_DT_ID = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int8: 4,
    torch.int16: 5,
    torch.int32: 6,
    torch.int64: 7,
    torch.bool: 8,
}


def _merge_dims(shape, stride):
    """Merge adjacent dims of a memory-order (descending stride) view.

    Returns (sizes, strides) ordered innermost-first.
    """
    mshp = []
    mstr = []
    for s, st in zip(reversed(list(shape)), reversed(list(stride))):
        if mshp and st != 0 and mstr[-1] != 0 and st == mshp[-1] * mstr[-1]:
            ps = mshp.pop()
            mstr.pop()
            mshp.append(ps * s)
            mstr.append(st)
        else:
            mshp.append(s)
            mstr.append(st)
    return mshp, mstr


def _launch_multi(tensors):
    """Fuse every dense same-dtype element of the list into few launches."""
    dt = tensors[0].dtype
    dt_id = _DT_ID.get(dt, 9)
    do_abs = not (dt == torch.bool or dt in _UNSIGNED)
    outs = []
    slots = []
    for t in tensors:
        out = torch.empty(t.shape, dtype=dt, device=t.device)
        outs.append(out)
        n = t.numel()
        if n:
            slots.append((t, out, n))
    for base in range(0, len(slots), _MAX_SLOTS):
        chunk = slots[base : base + _MAX_SLOTS]
        s = len(chunk)
        total = 0
        n_max = 0
        args = []
        for k in range(_MAX_SLOTS):
            if k < s:
                x, y, n = chunk[k]
                args.append(x)
                args.append(y)
                args.append(n)
                total += n
                if n > n_max:
                    n_max = n
            else:
                # Padding slots are inert: n = 0 makes every tile a no-op.
                args.append(chunk[0][0])
                args.append(chunk[0][1])
                args.append(0)
        args.append(total)
        args.append(dt_id)
        grid = lambda META, _nm=n_max, _s=s: (  # noqa: E731
            (_nm + META["BLOCK"] - 1) // META["BLOCK"],
            _s,
        )
        _abs_multi_kernel[grid](*args, S=s, DO_ABS=do_abs)
    return outs


def _launch_one(t):
    dt = t.dtype
    dev = t.device
    n = t.numel()
    is_complex = dt.is_complex
    if is_complex:
        is_f64 = dt == torch.complex128
        out_dtype = torch.float64 if is_f64 else torch.float32
    else:
        is_f64 = False
        out_dtype = dt
    if n == 0:
        return torch.empty(t.shape, dtype=out_dtype, device=dev)

    do_abs = not (dt == torch.bool or dt in _UNSIGNED)
    dt_id = _DT_ID.get(dt, 9)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    flat_grid = lambda META: ((n + META["BLOCK"] - 1) // META["BLOCK"],)  # noqa: E731
    dense = (t.dim() == 0) or t.is_contiguous()

    if dense:
        out = torch.empty(t.shape, dtype=out_dtype, device=dev)
        if is_complex:
            _abs_complex_flat_kernel[grid](
                t.data_ptr(),
                out,
                n,
                BLOCK=_BLOCK,
                IS_F64=is_f64,
                num_warps=_NUM_WARPS,
            )
        else:
            _abs_flat_kernel[flat_grid](t, out, n, dt_id, DO_ABS=do_abs)
        return out

    perm = sorted(range(t.dim()), key=lambda d: -t.stride(d))
    inv = [0] * t.dim()
    for i, d in enumerate(perm):
        inv[d] = i
    tp = t.permute(perm)
    out_flat = torch.empty(n, dtype=out_dtype, device=dev)
    out = out_flat.view(tp.shape).permute(inv)

    if tp.is_contiguous():
        if is_complex:
            _abs_complex_flat_kernel[grid](
                t.data_ptr(),
                out_flat,
                n,
                BLOCK=_BLOCK,
                IS_F64=is_f64,
                num_warps=_NUM_WARPS,
            )
        else:
            _abs_flat_kernel[flat_grid](t, out_flat, n, dt_id, DO_ABS=do_abs)
        return out

    mshp, mstr = _merge_dims(tp.shape, tp.stride())
    rank = len(mshp)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach_abs: unsupported tensor rank for strided access")
    pad = _MAX_RANK - rank
    dims = mshp + [1] * pad
    strs = mstr + [0] * pad
    if is_complex:
        _abs_complex_strided_kernel[grid](
            t.data_ptr(),
            out_flat,
            n,
            *dims,
            *strs,
            BLOCK=_BLOCK,
            RANK=rank,
            IS_F64=is_f64,
            num_warps=_NUM_WARPS,
        )
    else:
        _abs_strided_kernel[grid](
            t,
            out_flat,
            n,
            *dims,
            *strs,
            BLOCK=_BLOCK,
            RANK=rank,
            DO_ABS=do_abs,
            num_warps=_NUM_WARPS,
        )
    return out


def foreach_abs(self, *args, **kwargs):
    tensors = self
    if tensors is None:
        for key in ("self", "tensors", "input", "inputs"):
            if key in kwargs:
                tensors = kwargs[key]
                break
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    tensors = list(tensors)
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if len(tensors) == 1:
        return [_launch_one(tensors[0])]
    dt0 = tensors[0].dtype
    if not dt0.is_complex:
        fusable = True
        for t in tensors:
            if t.dtype != dt0 or not (t.dim() == 0 or t.is_contiguous()):
                fusable = False
                break
        if fusable:
            return _launch_multi(tensors)
    return [_launch_one(t) for t in tensors]
