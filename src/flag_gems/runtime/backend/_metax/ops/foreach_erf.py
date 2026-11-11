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

"""foreach_erf: elementwise erf over a sequence of tensors.

One fresh contiguous output per input; inputs may be arbitrary views (read
through their real strides, writes always coalesced). All contiguous tensors of
one call that share a compute mode (float64 vs everything else) are fused into a
single 2-D launch, so a list of N small tensors costs one launch instead of N.
"""

import torch
import triton
import triton.language as tl

_MAX_DIMS = 8
_MAXT = 16

_ERF_KEEP_DTYPE = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _erf_tile(x_ptr, y_ptr, offs, n, EVEN: tl.constexpr, F64: tl.constexpr):
    if EVEN:
        x = tl.load(x_ptr + offs)
    else:
        x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    if F64:
        y = tl.math.erf(x)
    else:
        y = tl.math.erf(x.to(tl.float32))
    y = y.to(y_ptr.dtype.element_ty)
    if EVEN:
        tl.store(y_ptr + offs, y)
    else:
        tl.store(y_ptr + offs, y, mask=offs < n)


@triton.jit(do_not_specialize=["n"])
def _erf_contig_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    _erf_tile(x_ptr, y_ptr, offs, n, EVEN, F64)


@triton.jit(
    do_not_specialize=[
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
def _erf_multi_kernel(
    x0,
    y0,
    x1,
    y1,
    x2,
    y2,
    x3,
    y3,
    x4,
    y4,
    x5,
    y5,
    x6,
    y6,
    x7,
    y7,
    x8,
    y8,
    x9,
    y9,
    x10,
    y10,
    x11,
    y11,
    x12,
    y12,
    x13,
    y13,
    x14,
    y14,
    x15,
    y15,
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
    EVEN: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if slot == 0:
        _erf_tile(x0, y0, offs, n0, EVEN, F64)
    elif slot == 1:
        _erf_tile(x1, y1, offs, n1, EVEN, F64)
    elif slot == 2:
        _erf_tile(x2, y2, offs, n2, EVEN, F64)
    elif slot == 3:
        _erf_tile(x3, y3, offs, n3, EVEN, F64)
    elif slot == 4:
        _erf_tile(x4, y4, offs, n4, EVEN, F64)
    elif slot == 5:
        _erf_tile(x5, y5, offs, n5, EVEN, F64)
    elif slot == 6:
        _erf_tile(x6, y6, offs, n6, EVEN, F64)
    elif slot == 7:
        _erf_tile(x7, y7, offs, n7, EVEN, F64)
    elif slot == 8:
        _erf_tile(x8, y8, offs, n8, EVEN, F64)
    elif slot == 9:
        _erf_tile(x9, y9, offs, n9, EVEN, F64)
    elif slot == 10:
        _erf_tile(x10, y10, offs, n10, EVEN, F64)
    elif slot == 11:
        _erf_tile(x11, y11, offs, n11, EVEN, F64)
    elif slot == 12:
        _erf_tile(x12, y12, offs, n12, EVEN, F64)
    elif slot == 13:
        _erf_tile(x13, y13, offs, n13, EVEN, F64)
    elif slot == 14:
        _erf_tile(x14, y14, offs, n14, EVEN, F64)
    else:
        _erf_tile(x15, y15, offs, n15, EVEN, F64)


@triton.jit(
    do_not_specialize=[
        "n",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
    ]
)
def _erf_strided_kernel(
    x_ptr,
    y_ptr,
    n,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    F64: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        in_off = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        in_off = tl.zeros([BLOCK], dtype=tl.int32)
    mask = offs < n

    q = offs
    if RANK > 7:
        i = q % a7
        q = q // a7
        in_off += i * b7
    if RANK > 6:
        i = q % a6
        q = q // a6
        in_off += i * b6
    if RANK > 5:
        i = q % a5
        q = q // a5
        in_off += i * b5
    if RANK > 4:
        i = q % a4
        q = q // a4
        in_off += i * b4
    if RANK > 3:
        i = q % a3
        q = q // a3
        in_off += i * b3
    if RANK > 2:
        i = q % a2
        q = q // a2
        in_off += i * b2
    if RANK > 1:
        i = q % a1
        q = q // a1
        in_off += i * b1
    if RANK > 0:
        i = q % a0
        in_off += i * b0

    x = tl.load(x_ptr + in_off, mask=mask, other=0.0)

    if F64:
        y = tl.math.erf(x)
    else:
        y = tl.math.erf(x.to(tl.float32))
    y = y.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)


# --------------------------------------------------------------------------- #
# host helpers
# --------------------------------------------------------------------------- #
def _config(n, small):
    if n <= 8192:
        return 256, 2
    if n <= (1 << 20):
        return 1024, 4
    if small:
        # 16-bit elements: 8192 elements per CTA = 64 B per thread, four
        # 128-bit vectors in flight per thread.
        return 8192, 8
    return 4096, 8


def _collapse(t):
    """Collapse a strided view to the shortest equivalent (shape, stride) list."""
    dims = []
    prev_st = None
    for s, st in zip(t.shape, t.stride()):
        if s == 1:
            continue
        if prev_st is not None and prev_st == st * s:
            dims[-1] = (dims[-1][0] * s, st)
        else:
            dims.append((s, st))
        prev_st = st
    return dims


def _launch_one(t, out, n, f64):
    dt = t.dtype
    BLOCK, warps = _config(n, dt is torch.float16 or dt is torch.bfloat16)
    _erf_contig_kernel[((n + BLOCK - 1) // BLOCK,)](
        t,
        out,
        n,
        BLOCK=BLOCK,
        EVEN=(n % BLOCK) == 0,
        F64=f64,
        I64=n > 2147483647,
        num_warps=warps,
    )


def _launch_multi(chunk, f64):
    m = len(chunk)
    maxn = 0
    for e in chunk:
        if e[2] > maxn:
            maxn = e[2]
    BLOCK, warps = _config(
        maxn,
        chunk[0][0].dtype is torch.float16 or chunk[0][0].dtype is torch.bfloat16,
    )
    even = True
    for e in chunk:
        if e[2] != maxn:
            even = False
            break
    if even:
        even = (maxn % BLOCK) == 0

    args = []
    for t, out, n in chunk:
        args.append(t)
        args.append(out)
    x0, y0 = args[0], args[1]
    while len(args) < 2 * _MAXT:
        args.append(x0)
        args.append(y0)
    for e in chunk:
        args.append(e[2])
    while len(args) < 2 * _MAXT + _MAXT:
        args.append(0)

    _erf_multi_kernel[((maxn + BLOCK - 1) // BLOCK, m)](
        *args,
        BLOCK=BLOCK,
        EVEN=even,
        F64=f64,
        I64=maxn > 2147483647,
        num_warps=warps,
    )


def _launch_strided(t, out, n, f64):
    dims = _collapse(t)
    rank = len(dims)
    if rank == 0:
        dims = [(1, 1)]
        rank = 1
    if rank > _MAX_DIMS:
        raise RuntimeError("foreach_erf: unsupported rank %d" % rank)

    a = [1] * _MAX_DIMS
    b = [0] * _MAX_DIMS
    for idx, (s, st) in enumerate(dims):
        a[idx] = s
        b[idx] = st

    BLOCK, warps = _config(n, t.dtype is torch.float16 or t.dtype is torch.bfloat16)
    _erf_strided_kernel[((n + BLOCK - 1) // BLOCK,)](
        t,
        out,
        n,
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7],
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        RANK=rank,
        BLOCK=BLOCK,
        F64=f64,
        I64=n > 2147483647,
        num_warps=warps,
    )


def _apply(items):
    k = len(items)
    outs = [None] * k
    f32g = []
    f64g = []
    for i in range(k):
        t = items[i]
        if not isinstance(t, torch.Tensor):
            outs[i] = _apply(list(t))
            continue
        dt = t.dtype
        if dt is torch.float64:
            f64 = True
            odt = dt
        elif dt is torch.float32 or dt is torch.float16 or dt is torch.bfloat16:
            f64 = False
            odt = dt
        elif dt is torch.complex64 or dt is torch.complex128:
            # ATen parity: the CUDA foreach unary kernels (and this target's
            # erf_cuda) reject complex dtypes, so mirror the reference rejection.
            name = "ComplexFloat" if dt is torch.complex64 else "ComplexDouble"
            raise NotImplementedError(
                "\"foreach_unary_op_cuda\" not implemented for '%s'" % name
            )
        else:
            f64 = False
            odt = torch.float32

        out = torch.empty(t.shape, dtype=odt, device=t.device)
        outs[i] = out
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            if f64:
                f64g.append((t, out, n))
            else:
                f32g.append((t, out, n))
        else:
            _launch_strided(t, out, n, f64)

    for group, f64 in ((f32g, False), (f64g, True)):
        m = len(group)
        if m == 0:
            continue
        if m == 1:
            e = group[0]
            _launch_one(e[0], e[1], e[2], f64)
        else:
            for s in range(0, m, _MAXT):
                chunk = group[s : s + _MAXT]
                if len(chunk) == 1:
                    e = chunk[0]
                    _launch_one(e[0], e[1], e[2], f64)
                else:
                    _launch_multi(chunk, f64)
    return outs


def _extract(args, kwargs):
    if kwargs:
        if "self" in kwargs:
            return kwargs["self"]
        if "tensors" in kwargs:
            return kwargs["tensors"]
        if "input" in kwargs:
            return kwargs["input"]
    if len(args) == 1:
        return args[0]
    return list(args)


def foreach_erf(*args, **kwargs):
    obj = _extract(args, kwargs)
    if isinstance(obj, torch.Tensor):
        return _apply([obj])
    items = list(obj)
    if not items:
        # ATen parity: torch._foreach_erf([]) raises this exact error.
        raise RuntimeError("Tensor list must have at least one tensor.")
    return _apply(items)
