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

"""In-place foreach subtract (ScalarList / scalar / TensorList) in Triton.

Semantics reproduce ``torch._foreach_sub_`` on this build:
  * length check: ``len(self) == len(scalars)`` (no broadcast) -> RuntimeError
  * empty list -> RuntimeError
  * in-place: the observable result is the mutated storage; ``run`` returns None
  * mutation lands in the original storage (contiguous and strided views)

Performance model (measured): latency = fixed host cost + bytes / peak stream
bandwidth, with the kernels already at peak bandwidth.  The fixed host cost is
therefore the only tunable part, so the steady state reuses the compiled
launcher directly instead of paying Triton's per-call binding again.
"""

import operator
import struct

import torch
import triton
import triton.language as tl

try:  # used only to reach the already-compiled launcher for the steady state
    from triton.runtime import driver as _triton_driver
except Exception:  # pragma: no cover - fall back to the plain JIT launch path
    _triton_driver = None

# dtype codes used as a constexpr switch inside the kernels
_DT_F16 = 0  # float16 / bfloat16  (fp32 opmath)
_DT_F32 = 1  # float32
_DT_F64 = 2  # float64
_DT_INT = 3  # integral

_I32_LIMIT = 1 << 30
_I32_MAX = 1 << 31

_DTYPE_CODES = {
    torch.float16: _DT_F16,
    torch.bfloat16: _DT_F16,
    torch.float32: _DT_F32,
    torch.float64: _DT_F64,
}

_FUSE_MAX = 4

# (dtype, numel, members) -> bound launcher + the constants that kernel was
# compiled with, or None when the steady-state path is not usable.
_FAST_LAUNCH = {}


@triton.jit
def _fsub_flat(
    X, S_BITS, S_F, n, BLOCK: tl.constexpr, DT: tl.constexpr, I64: tl.constexpr
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(X + offs, mask=m)
    if DT == 0:  # float16 / bfloat16 with fp32 opmath
        y = (x.to(tl.float32) - S_F).to(x.dtype)
    elif DT == 1:  # float32
        y = x - S_F
    elif DT == 2:  # float64, scalar carried as its exact int64 bit pattern
        y = x - tl.cast(S_BITS, tl.float64, bitcast=True)
    else:  # integral
        y = (x.to(tl.int64) - tl.cast(S_BITS, tl.int64)).to(x.dtype)
    tl.store(X + offs, y, mask=m)


@triton.jit
def _fsub_fused_f(
    X0,
    o1,
    o2,
    o3,
    s0,
    s1,
    s2,
    s3,
    n,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    I64: tl.constexpr,
):
    """One launch for up to 4 same-dtype, same-numel, contiguous tensors.

    Axis 0 of the grid is the block index (x-fastest, so consecutively issued
    CTAs stream one member's contiguous region) and axis 1 selects the list
    member, given as element offsets o* relative to X0.
    """
    pid = tl.program_id(0)
    i = tl.program_id(1)
    off = tl.where(i == 1, o1, tl.where(i == 2, o2, tl.where(i == 3, o3, 0)))
    s = tl.where(i == 1, s1, tl.where(i == 2, s2, tl.where(i == 3, s3, s0)))
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if I64:
        p = X0 + off.to(tl.int64) + offs
    else:
        p = X0 + off + offs
    x = tl.load(p, mask=m)
    if DT == 0:
        y = (x.to(tl.float32) - s).to(x.dtype)
    else:
        y = x - s
    tl.store(p, y, mask=m)


@triton.jit
def _fsub_rows(
    X,
    S_BITS,
    S_F,
    inner_n,
    inner_stride,
    bpr,
    d0,
    t0,
    d1,
    t1,
    d2,
    t2,
    d3,
    t3,
    d4,
    t4,
    d5,
    t5,
    d6,
    t6,
    ND: tl.constexpr,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // bpr
    col = pid - row * bpr
    off = (row % d0) * t0
    row = row // d0
    if ND > 1:
        off += (row % d1) * t1
        row = row // d1
    if ND > 2:
        off += (row % d2) * t2
        row = row // d2
    if ND > 3:
        off += (row % d3) * t3
        row = row // d3
    if ND > 4:
        off += (row % d4) * t4
        row = row // d4
    if ND > 5:
        off += (row % d5) * t5
        row = row // d5
    if ND > 6:
        off += (row % d6) * t6
        row = row // d6
    idx = col * BLOCK + tl.arange(0, BLOCK)
    m = idx < inner_n
    if I64:
        po = off.to(tl.int64) + idx.to(tl.int64) * inner_stride
    else:
        po = off + idx * inner_stride
    x = tl.load(X + po, mask=m)
    if DT == 0:  # float16 / bfloat16 with fp32 opmath
        y = (x.to(tl.float32) - S_F).to(x.dtype)
    elif DT == 1:  # float32
        y = x - S_F
    elif DT == 2:  # float64, scalar carried as its exact int64 bit pattern
        y = x - tl.cast(S_BITS, tl.float64, bitcast=True)
    else:  # integral
        y = (x.to(tl.int64) - tl.cast(S_BITS, tl.int64)).to(x.dtype)
    tl.store(X + po, y, mask=m)


@triton.jit
def _fsub_t_flat(X, Y, n, BLOCK: tl.constexpr, DT: tl.constexpr, I64: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(X + offs, mask=m)
    y = tl.load(Y + offs, mask=m)
    if DT == 0:
        r = (x.to(tl.float32) - y.to(tl.float32)).to(x.dtype)
    else:
        r = x - y
    tl.store(X + offs, r, mask=m)


def _pick_block(code, itemsize, n):
    if code == _DT_F16:
        block = 2048
    elif code == _DT_F64 or (code == _DT_INT and itemsize >= 8):
        block = 512
    else:
        block = 1024
    if n < block:
        block = max(32, triton.next_power_of_2(n))
    return block


def _pick_warps(block):
    return max(1, min(4, block >> 7))


def _scalar_args(value, code):
    """Return (bits for the int64/floats bitcast, plain python float)."""
    if code == _DT_F64:
        try:
            d = float(value)
        except Exception:
            d = float(operator.index(value))
        return struct.unpack("<q", struct.pack("<d", d))[0], 0.0
    if code == _DT_INT:
        try:
            return int(operator.index(value)), 0.0
        except TypeError:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type Long"
            )
    return 0, float(value)


def _plan_strided(t):
    """Pick the inner (unit/small stride) axis and collapse the outer dims.

    Returns (inner_n, inner_stride, rows, outer_dims).
    """
    dims = [(int(sz), int(st)) for sz, st in zip(t.shape, t.stride()) if int(sz) != 1]
    collapsed = []
    for sz, st in dims:
        if collapsed and collapsed[-1][1] == st * sz:
            psz, _ = collapsed[-1]
            collapsed[-1] = (psz * sz, st)
        else:
            collapsed.append((sz, st))
    best = -1
    best_key = None
    for i, (sz, st) in enumerate(collapsed):
        key = (0 if st == 1 else 1, abs(st), -sz)
        if best_key is None or key < best_key:
            best_key = key
            best = i
    if best < 0:
        return 1, 0, 1, []
    inner_n, inner_stride = collapsed[best]
    outer = collapsed[:best] + collapsed[best + 1 :]
    rows = 1
    for sz, _ in outer:
        rows *= sz
    return inner_n, inner_stride, rows, outer


def _launch_scalar(t, value):
    dtype = t.dtype
    code = _DTYPE_CODES.get(dtype, _DT_INT)
    if code == _DT_INT and dtype is torch.bool:
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is not supported."
        )
    n = t.numel()
    if n == 0:
        return
    bits, sf = _scalar_args(value, code)
    block = _pick_block(code, t.element_size(), n)
    warps = _pick_warps(block)
    i64 = n >= _I32_LIMIT
    if t.is_contiguous():
        grid = ((n + block - 1) // block,)
        _fsub_flat[grid](t, bits, sf, n, BLOCK=block, DT=code, I64=i64, num_warps=warps)
        return
    inner_n, inner_stride, rows, outer = _plan_strided(t)
    bpr = (inner_n + block - 1) // block
    grid = (rows * bpr,)
    dims = [1, 1, 1, 1, 1, 1, 1]
    sts = [0, 0, 0, 0, 0, 0, 0]
    for i, (sz, st) in enumerate(outer[:7]):
        dims[i] = sz
        sts[i] = st
    _fsub_rows[grid](
        t,
        bits,
        sf,
        inner_n,
        inner_stride,
        bpr,
        dims[0],
        sts[0],
        dims[1],
        sts[1],
        dims[2],
        sts[2],
        dims[3],
        sts[3],
        dims[4],
        sts[4],
        dims[5],
        sts[5],
        dims[6],
        sts[6],
        ND=min(len(outer), 7),
        BLOCK=block,
        DT=code,
        I64=i64,
        num_warps=warps,
    )


def _launch_group(tensors, values, i, j):
    """One fused launch for 2..4 same-dtype / same-numel / contiguous members."""
    t0 = tensors[i]
    nt = j - i
    n = t0.numel()
    dtype = t0.dtype
    itemsize = t0.element_size()
    p0 = t0.data_ptr()
    o1 = (tensors[i + 1].data_ptr() - p0) // itemsize
    o2 = (tensors[i + 2].data_ptr() - p0) // itemsize if nt > 2 else 0
    o3 = (tensors[i + 3].data_ptr() - p0) // itemsize if nt > 3 else 0
    # Aliased members (overlapping ranges) must keep the sequential semantics.
    if (
        abs(o1) < n
        or (nt > 2 and (abs(o2) < n or abs(o2 - o1) < n))
        or (nt > 3 and (abs(o3) < n or abs(o3 - o1) < n or abs(o3 - o2) < n))
    ):
        for k in range(i, j):
            _launch_scalar(tensors[k], values[k])
        return
    s0 = float(values[i])
    s1 = float(values[i + 1])
    s2 = float(values[i + 2]) if nt > 2 else 0.0
    s3 = float(values[i + 3]) if nt > 3 else 0.0
    # A reused launcher is only valid inside the assumptions its binary was
    # compiled with: 32-bit typed offsets (|offset| + numel < 2**31 keeps the
    # inferred argument type and the in-kernel index arithmetic i32) and the
    # 16-element divisibility Triton specialised on.
    gate = (
        n > 1
        and (n & 15) == 0
        and ((o1 | o2 | o3) & 15) == 0
        and abs(o1) + n < _I32_MAX
        and (nt <= 2 or abs(o2) + n < _I32_MAX)
        and (nt <= 3 or abs(o3) + n < _I32_MAX)
    )
    if gate:
        fast = _FAST_LAUNCH.get((dtype, n, nt))
        if fast is not None:
            launcher, function, packed, gx, gy, dev, block, code, i64 = fast
            # Raw device addresses: the generated launcher accepts PyLong and
            # then skips both the data_ptr() call and the per-pointer driver
            # attribute query it performs for tensor arguments.
            launcher(
                gx,
                gy,
                1,
                _triton_driver.active.get_current_stream(dev),
                function,
                packed,
                None,
                None,
                None,
                p0,
                o1,
                o2,
                o3,
                s0,
                s1,
                s2,
                s3,
                n,
                block,
                code,
                i64,
            )
            return
    code = _DTYPE_CODES[dtype]
    block = _pick_block(code, itemsize, n)
    warps = _pick_warps(block)
    i64 = n >= _I32_LIMIT
    g = (n + block - 1) // block
    cache = None
    before = None
    if gate and _triton_driver is not None:
        try:
            dev = _triton_driver.active.get_current_device()
            cache = _fsub_fused_f.device_caches[dev][0]
            before = tuple(cache.keys())
        except Exception:
            cache = None
    _fsub_fused_f[(g, nt)](
        t0,
        o1,
        o2,
        o3,
        s0,
        s1,
        s2,
        s3,
        n,
        BLOCK=block,
        DT=code,
        I64=i64,
        num_warps=warps,
    )
    if cache is not None:
        try:
            entry = None
            for k, ck in cache.items():
                if k not in before:
                    entry = (
                        ck.run,
                        ck.function,
                        ck.packed_metadata,
                        g,
                        nt,
                        dev,
                        block,
                        code,
                        i64,
                    )
                    break
            _FAST_LAUNCH[(dtype, n, nt)] = entry
        except Exception:
            _FAST_LAUNCH[(dtype, n, nt)] = None


def _can_join(t, t0, n0):
    return t.dtype is t0.dtype and t.is_contiguous() and t.numel() == n0 and n0 > 0


def _launch_tensor(dst, src):
    n = dst.numel()
    if n == 0:
        return
    code = _DTYPE_CODES.get(dst.dtype, _DT_INT)
    block = _pick_block(code, dst.element_size(), n)
    warps = _pick_warps(block)
    grid = ((n + block - 1) // block,)
    _fsub_t_flat[grid](
        dst, src, n, BLOCK=block, DT=code, I64=(n >= _I32_LIMIT), num_warps=warps
    )


def _as_list(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise TypeError("expected a sequence of tensors, got %r" % (type(obj),))


def _is_number(obj):
    return isinstance(obj, (int, float, complex)) and not isinstance(obj, bool)


def _run_scalars(tensors, values):
    nt = len(tensors)
    i = 0
    while i < nt:
        t0 = tensors[i]
        dtype = t0.dtype
        code = _DTYPE_CODES.get(dtype)
        if code is not None and code <= _DT_F32:
            n0 = t0.numel()
            if n0 > 0 and t0.is_contiguous():
                j = i + 1
                while j < nt and j - i < _FUSE_MAX:
                    t = tensors[j]
                    if t.dtype is not dtype or t.numel() != n0 or not t.is_contiguous():
                        break
                    j += 1
                if j - i > 1:
                    _launch_group(tensors, values, i, j)
                    i = j
                    continue
        _launch_scalar(t0, values[i])
        i += 1


def foreach_sub_scalar_list_(self, scalars=None, *args, **kwargs):
    tensors = self if type(self) is list else _as_list(self)
    nt = len(tensors)
    if nt == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if kwargs:
        if "scalars" in kwargs and scalars is None:
            scalars = kwargs.pop("scalars")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (list(kwargs),))
    if type(scalars) is list:
        if len(scalars) != nt:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
        if type(scalars[0]) is torch.Tensor:
            for dst, src in zip(tensors, scalars):
                if dst.shape != src.shape:
                    raise RuntimeError(
                        "Tensor list must have same number of elements as"
                        " tensor list."
                    )
                _launch_tensor(dst, src)
        else:
            _run_scalars(tensors, scalars)
        return None
    if scalars is None and args:
        scalars = args[0]
    if isinstance(scalars, torch.Tensor):
        _run_scalars(tensors, [scalars] * nt)
        return None
    if isinstance(scalars, (int, float)) and not isinstance(scalars, bool):
        _run_scalars(tensors, [scalars] * nt)
        return None
    items = list(scalars)
    if len(items) != nt:
        raise RuntimeError(
            "Tensor list must have same number of elements as scalar list."
        )
    if nt > 0 and isinstance(items[0], torch.Tensor):
        for dst, src in zip(tensors, items):
            if dst.shape != src.shape:
                raise RuntimeError(
                    "Tensor list must have same number of elements as tensor list."
                )
            _launch_tensor(dst, src)
        return None
    _run_scalars(tensors, items)
    return None
