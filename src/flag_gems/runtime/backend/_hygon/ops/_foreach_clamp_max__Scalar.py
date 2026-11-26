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

import logging
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:  # cheap pre-bound launch path; ordinary launches remain the fallback
    from triton.runtime.jit import MockTensor as _MockTensor
except Exception:  # noqa: BLE001
    _MockTensor = None
try:
    from triton.runtime import driver as _driver
except Exception:  # noqa: BLE001
    _driver = None


# ---------------------------------------------------------------------------
# Elementwise clamp-max (ATen std::min semantics), shared by all kernels.
#   MODE 0: fp16 / bf16  -> compare in fp32 opmath, store in tensor dtype
#   MODE 1: fp32 (and fp64 with an fp32-exact bound) -> native compare
#   MODE 2: integral     -> int64 compare, store back in tensor dtype
#   MODE 3: fp64 with non-fp32-exact bound -> bound supplied as raw fp64 bits
#   MODE 4: bool         -> v & (bound != 0)
# ---------------------------------------------------------------------------
@triton.jit
def _cmax(v, bf, bi, MODE: tl.constexpr):
    if MODE == 0:
        vf = v.to(tl.float32)
        r = tl.where(vf != vf, vf, tl.minimum(vf, bf))
        return r.to(v.dtype)
    elif MODE == 1:
        return tl.where(v != v, v, tl.minimum(v, bf))
    elif MODE == 2:
        return tl.minimum(v.to(tl.int64), bi).to(v.dtype)
    elif MODE == 3:
        b64 = bi.to(tl.float64, bitcast=True)
        return tl.where(v != v, v, tl.minimum(v, b64))
    else:
        return v & (bi != 0)


@triton.jit
def _slot_x(ptr, start, ar, bf, bi, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = start * BLOCK + ar
    v = tl.load(ptr + offs)
    tl.store(ptr + offs, _cmax(v, bf, bi, MODE))


@triton.jit
def _slot_m(ptr, start, n, ar, bf, bi, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = start * BLOCK + ar
    mask = offs < n
    v = tl.load(ptr + offs, mask=mask, other=0)
    tl.store(ptr + offs, _cmax(v, bf, bi, MODE), mask=mask)


# Exact-divisor single-tensor launch: two runtime arguments only, because the
# launch path on this target charges several microseconds per runtime argument.
@triton.jit
def _clamp_max_x(ptr, bf, BLOCK: tl.constexpr, MODE: tl.constexpr):
    _slot_x(ptr, tl.program_id(0), tl.arange(0, BLOCK), bf, 0, BLOCK, MODE)


# General single-tensor launch (tail masking, integral/fp64/bool bounds).
@triton.jit(do_not_specialize=["n", "bi"])
def _clamp_max_1d(
    ptr, bf, bi, n, BLOCK: tl.constexpr, MODE: tl.constexpr, EXACT: tl.constexpr
):
    pid = tl.program_id(0)
    if EXACT:
        _slot_x(ptr, pid, tl.arange(0, BLOCK), bf, bi, BLOCK, MODE)
    else:
        _slot_m(ptr, pid, n, tl.arange(0, BLOCK), bf, bi, BLOCK, MODE)


# Packed launch for a list of small exact tensors of one dtype/bound. Slot
# boundaries are constexpr, so the only runtime arguments beyond the launch
# itself are the four pointers and the bound. Slot selection is a uniform
# scalar branch, so every slot keeps a real pointer argument (and therefore
# the same vectorization as _clamp_max_x).
@triton.jit
def _clamp_max_pk(
    p0,
    p1,
    p2,
    p3,
    bf,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    if pid < C1:
        _slot_x(p0, pid, ar, bf, 0, BLOCK, MODE)
    elif pid < C2:
        _slot_x(p1, pid - C1, ar, bf, 0, BLOCK, MODE)
    elif pid < C3:
        _slot_x(p2, pid - C2, ar, bf, 0, BLOCK, MODE)
    else:
        _slot_x(p3, pid - C3, ar, bf, 0, BLOCK, MODE)


_SD = ["n", "bi"]
for _i in range(8):
    _SD.append("s%d" % _i)
for _i in range(8):
    _SD.append("t%d" % _i)


@triton.jit(do_not_specialize=_SD)
def _clamp_max_nd(
    ptr,
    bf,
    bi,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    addr = offs * 0
    idx = rem % s7
    rem = rem // s7
    addr += idx * t7
    idx = rem % s6
    rem = rem // s6
    addr += idx * t6
    idx = rem % s5
    rem = rem // s5
    addr += idx * t5
    idx = rem % s4
    rem = rem // s4
    addr += idx * t4
    idx = rem % s3
    rem = rem // s3
    addr += idx * t3
    idx = rem % s2
    rem = rem // s2
    addr += idx * t2
    idx = rem % s1
    rem = rem // s1
    addr += idx * t1
    idx = rem % s0
    addr += idx * t0
    v = tl.load(ptr + addr, mask=mask, other=0)
    tl.store(ptr + addr, _cmax(v, bf, bi, MODE), mask=mask)


_I32_MIN = -(2**31)
_I32_MAX = 2**31

_BLOCK = 1024
_WARPS = 4
_GROUP = 4
# Batching a whole list into one launch only pays off where GPU work is
# negligible (measured: 4096-element lists). Above this size the packed form
# cost 6-19% of GPU throughput, so larger tensors use per-tensor launches.
_PK_MAX_ELEMS = 1 << 16
# Pre-bound launches are used only up to a size whose numerical equality with
# the ordinary launch has been verified bit-exact on this target (checked at
# 4096, 16.7M and 67M elements for fp16/fp32). The 1.07e9-element workloads
# stay on the ordinary launch: their host cost is hidden by ~3.5 ms of GPU work
# and the fp32 case is a 4.3 GB buffer.
_FAST_MAX_ELEMS = 1 << 26

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}
_FAST = {}


def _mode_and_bounds(dtype, bound):
    """Return (MODE, float_bound, int_bound) for one tensor dtype + bound."""
    if dtype is torch.float16 or dtype is torch.bfloat16:
        return 0, float(bound), 0
    if dtype is torch.float32:
        return 1, float(bound), 0
    if dtype is torch.float64:
        bits = struct.unpack("<q", struct.pack("<d", float(bound)))[0]
        if _I32_MIN <= bits < _I32_MAX:
            return 1, float(bound), 0
        return 3, 0.0, bits
    if dtype is torch.bool:
        return 4, 0.0, 1 if float(bound) != 0.0 else 0
    return 2, 0.0, int(bound)


def _collapse(t):
    """Collapsed (size, stride) descriptor of a strided tensor, C order."""
    dims = []
    for sz, st in zip(t.shape, t.stride()):
        if sz == 1:
            continue
        if dims and dims[-1][1] == st * sz:
            dims[-1][0] *= sz
        else:
            dims.append([sz, st])
    return dims


def _mk_x_runner(dev, dtype, mode, nb):
    """Compiled-kernel runner for _clamp_max_x, or False if unavailable."""
    if _MockTensor is None or _driver is None:
        return False
    tld = _TL_DTYPE.get(dtype)
    if tld is None:
        return False
    try:
        if _driver.active.get_current_device() != dev:
            return False
        m = _MockTensor(tld)
        ck = _clamp_max_x.warmup(
            m, 0.0, BLOCK=_BLOCK, MODE=mode, num_warps=_WARPS, grid=(1,)
        )
        ck._init_handles()
        return ck[(nb, 1, 1)]
    except Exception:  # noqa: BLE001
        return False


def _mk_pk_runner(dev, dtype, mode, c1, c2, c3, total):
    if _MockTensor is None or _driver is None:
        return False
    tld = _TL_DTYPE.get(dtype)
    if tld is None:
        return False
    try:
        if _driver.active.get_current_device() != dev:
            return False
        m = _MockTensor(tld)
        ck = _clamp_max_pk.warmup(
            m,
            m,
            m,
            m,
            0.0,
            C1=c1,
            C2=c2,
            C3=c3,
            BLOCK=_BLOCK,
            MODE=mode,
            num_warps=_WARPS,
            grid=(1,),
        )
        ck._init_handles()
        return ck[(total, 1, 1)]
    except Exception:  # noqa: BLE001
        return False


def _launch_x(t, dev, dtype, mode, bf, n):
    ptr = t.data_ptr()
    nb = n // _BLOCK
    if not ptr & 15:
        key = (dev, dtype, mode, 0, nb)
        r = _FAST.get(key)
        if r is None:
            r = _mk_x_runner(dev, dtype, mode, nb)
            _FAST[key] = r
        if r:
            try:
                r(ptr, bf, _BLOCK, mode)
                return
            except Exception:  # noqa: BLE001
                _FAST[key] = False
    _clamp_max_x[(nb,)](t, bf, BLOCK=_BLOCK, MODE=mode, num_warps=_WARPS)


def _launch_single(t, dev, dtype, mode, bf, bi, n):
    if mode < 2 and n % _BLOCK == 0:
        if n <= _FAST_MAX_ELEMS:
            _launch_x(t, dev, dtype, mode, bf, n)
        else:
            _clamp_max_x[(n // _BLOCK,)](
                t, bf, BLOCK=_BLOCK, MODE=mode, num_warps=_WARPS
            )
    else:
        _clamp_max_1d[(triton.cdiv(n, _BLOCK),)](
            t,
            bf,
            bi,
            n,
            BLOCK=_BLOCK,
            MODE=mode,
            num_warps=_WARPS,
            EXACT=(n % _BLOCK == 0),
        )


def _launch_pk(grp, ns, dev, dtype, mode, bf):
    blocks = [x // _BLOCK for x in ns]
    total = sum(blocks)
    if len(blocks) > 1:
        c1 = blocks[0]
    else:
        c1 = total
    if len(blocks) > 2:
        c2 = c1 + blocks[1]
    else:
        c2 = total
    if len(blocks) > 3:
        c3 = c2 + blocks[2]
    else:
        c3 = total
    p0 = grp[0]
    ptrs = [x.data_ptr() for x in grp]
    q0 = ptrs[0]
    q1 = ptrs[1] if len(ptrs) > 1 else q0
    q2 = ptrs[2] if len(ptrs) > 2 else q0
    q3 = ptrs[3] if len(ptrs) > 3 else q0
    if not (q0 | q1 | q2 | q3) & 15:
        key = (dev, dtype, mode, len(grp), c1, c2, c3)
        r = _FAST.get(key)
        if r is None:
            r = _mk_pk_runner(dev, dtype, mode, c1, c2, c3, total)
            _FAST[key] = r
        if r:
            try:
                r(q0, q1, q2, q3, bf, c1, c2, c3, _BLOCK, mode)
                return
            except Exception:  # noqa: BLE001
                _FAST[key] = False
    _clamp_max_pk[(total,)](
        p0,
        grp[1] if len(grp) > 1 else p0,
        grp[2] if len(grp) > 2 else p0,
        grp[3] if len(grp) > 3 else p0,
        bf,
        C1=c1,
        C2=c2,
        C3=c3,
        BLOCK=_BLOCK,
        MODE=mode,
        num_warps=_WARPS,
    )


def _foreach_clamp_max__Scalar(self, scalar):
    logger.debug("GEMS_HYGON _FOREACH_CLAMP_MAX__SCALAR")
    cls = self.__class__
    if cls is list or cls is tuple:
        tensors = self
    elif isinstance(self, torch.Tensor):
        tensors = (self,)
    else:
        tensors = tuple(self)
    n_t = len(tensors)
    if n_t == 0:
        return None

    if scalar.__class__ is torch.Tensor:
        scalar = scalar.item()
    s_cls = scalar.__class__
    if s_cls is list or s_cls is tuple:
        if len(scalar) != n_t:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
        bounds = scalar
    else:
        bounds = None

    i = 0
    while i < n_t:
        t = tensors[i]
        n = t.numel()
        if n == 0:
            i += 1
            continue
        dt = t.dtype
        dev = t.get_device()
        if dt is torch.float16 or dt is torch.bfloat16:
            mode = 0
            bf = float(bounds[i] if bounds is not None else scalar)
            bi = 0
        else:
            mode, bf, bi = _mode_and_bounds(
                dt, bounds[i] if bounds is not None else scalar
            )
        if not t.is_contiguous():
            dims = _collapse(t)
            if len(dims) > 8:
                raise RuntimeError("unsupported tensor rank for clamp_max")
            pad = 8 - len(dims)
            sizes = [1] * pad + [d[0] for d in dims]
            strides = [0] * pad + [d[1] for d in dims]
            _clamp_max_nd[(triton.cdiv(n, _BLOCK),)](
                t,
                bf,
                bi,
                n,
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3],
                sizes[4],
                sizes[5],
                sizes[6],
                sizes[7],
                strides[0],
                strides[1],
                strides[2],
                strides[3],
                strides[4],
                strides[5],
                strides[6],
                strides[7],
                BLOCK=_BLOCK,
                MODE=mode,
                num_warps=_WARPS,
            )
            i += 1
            continue
        if mode > 1 or n > _PK_MAX_ELEMS or n % _BLOCK:
            _launch_single(t, dev, dt, mode, bf, bi, n)
            i += 1
            continue
        grp = [t]
        ns = [n]
        j = i + 1
        while j < n_t and j - i < _GROUP:
            u = tensors[j]
            nu = u.numel()
            if (
                nu == 0
                or nu > _PK_MAX_ELEMS
                or nu % _BLOCK
                or u.get_device() != dev
                or u.dtype is not dt
                or not u.is_contiguous()
            ):
                break
            if float(bounds[j] if bounds is not None else scalar) != bf:
                break
            grp.append(u)
            ns.append(nu)
            j += 1
        if j == i + 1:
            _launch_single(t, dev, dt, mode, bf, bi, n)
        else:
            _launch_pk(grp, ns, dev, dt, mode, bf)
        i = j
    return None
