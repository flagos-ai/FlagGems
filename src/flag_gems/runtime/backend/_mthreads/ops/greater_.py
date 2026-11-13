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

"""In-place elementwise greater-than: aten::greater_.Tensor / aten::greater_.Scalar.

Compares ``self`` against ``other`` elementwise and writes 0/1 into ``self``'s
existing storage, returning ``self`` itself (same object, storage, strides,
dtype, device and storage_offset).
"""

import logging
import struct

import torch
import triton
import triton.language as tl

from flag_gems.ops.greater_ import greater_ as default_greater_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_M_F64 = 0
_M_F32 = 1
_M_I64 = 2

_BOOL = torch.bool
_F64 = torch.float64
_I8 = torch.int8

_MODE_BY_DTYPE = {
    torch.float64: _M_F64,
    torch.float32: _M_F32,
    torch.float16: _M_F32,
    torch.bfloat16: _M_F32,
    torch.int64: _M_I64,
    torch.int32: _M_I64,
    torch.int16: _M_I64,
    torch.int8: _M_I64,
    torch.uint8: _M_I64,
    torch.bool: _M_I64,
}

_STRIDED_INT_ARGS = [
    "inner",
    "s_is",
    "o_is",
    "val",
    "m0",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    "ss0",
    "ss1",
    "ss2",
    "ss3",
    "ss4",
    "ss5",
    "ss6",
    "ss7",
    "os0",
    "os1",
    "os2",
    "os3",
    "os4",
    "os5",
    "os6",
    "os7",
]


# ---------------------------------------------------------------- kernels ---
@triton.jit
def _gt_flat(
    s_ptr, o_ptr, n, MODE: tl.constexpr, OBCAST: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(s_ptr + offs, mask=mask)
    if OBCAST:
        b = tl.load(o_ptr)
    else:
        b = tl.load(o_ptr + offs, mask=mask)
    if MODE == 0:
        c = a.to(tl.float64) > b.to(tl.float64)
    elif MODE == 1:
        c = a.to(tl.float32) > b.to(tl.float32)
    else:
        c = a.to(tl.int64) > b.to(tl.int64)
    tl.store(s_ptr + offs, c.to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["val"])
def _gt_flat_s(s_ptr, val, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(s_ptr + offs, mask=mask)
    if MODE == 0:
        c = a.to(tl.float64) > val.to(tl.int64).to(tl.float64, bitcast=True)
    elif MODE == 1:
        c = a.to(tl.float32) > val.to(tl.float32)
    else:
        c = a.to(tl.int64) > val.to(tl.int64)
    tl.store(s_ptr + offs, c.to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=_STRIDED_INT_ARGS)
def _gt_strided(
    s_ptr,
    o_ptr,
    inner,
    s_is,
    o_is,
    val,
    m0,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    m7,
    ss0,
    ss1,
    ss2,
    ss3,
    ss4,
    ss5,
    ss6,
    ss7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    NOUT: tl.constexpr,
    SCALAR: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rem = tl.program_id(1)
    so = 0
    oo = 0
    if NOUT > 0:
        q = rem // m0
        rem = rem - q * m0
        so = so + q.to(tl.int64) * ss0.to(tl.int64)
        oo = oo + q.to(tl.int64) * os0.to(tl.int64)
    if NOUT > 1:
        q = rem // m1
        rem = rem - q * m1
        so = so + q.to(tl.int64) * ss1.to(tl.int64)
        oo = oo + q.to(tl.int64) * os1.to(tl.int64)
    if NOUT > 2:
        q = rem // m2
        rem = rem - q * m2
        so = so + q.to(tl.int64) * ss2.to(tl.int64)
        oo = oo + q.to(tl.int64) * os2.to(tl.int64)
    if NOUT > 3:
        q = rem // m3
        rem = rem - q * m3
        so = so + q.to(tl.int64) * ss3.to(tl.int64)
        oo = oo + q.to(tl.int64) * os3.to(tl.int64)
    if NOUT > 4:
        q = rem // m4
        rem = rem - q * m4
        so = so + q.to(tl.int64) * ss4.to(tl.int64)
        oo = oo + q.to(tl.int64) * os4.to(tl.int64)
    if NOUT > 5:
        q = rem // m5
        rem = rem - q * m5
        so = so + q.to(tl.int64) * ss5.to(tl.int64)
        oo = oo + q.to(tl.int64) * os5.to(tl.int64)
    if NOUT > 6:
        q = rem // m6
        rem = rem - q * m6
        so = so + q.to(tl.int64) * ss6.to(tl.int64)
        oo = oo + q.to(tl.int64) * os6.to(tl.int64)
    if NOUT > 7:
        q = rem // m7
        rem = rem - q * m7
        so = so + q.to(tl.int64) * ss7.to(tl.int64)
        oo = oo + q.to(tl.int64) * os7.to(tl.int64)

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < inner
    ai = so + offs.to(tl.int64) * s_is.to(tl.int64)
    a = tl.load(s_ptr + ai, mask=mask)
    if SCALAR:
        if MODE == 0:
            b = val.to(tl.int64).to(tl.float64, bitcast=True)
        elif MODE == 1:
            b = val.to(tl.float32)
        else:
            b = val.to(tl.int64)
    else:
        b = tl.load(o_ptr + oo + offs.to(tl.int64) * o_is.to(tl.int64), mask=mask)
    if MODE == 0:
        c = a.to(tl.float64) > b.to(tl.float64)
    elif MODE == 1:
        c = a.to(tl.float32) > b.to(tl.float32)
    else:
        c = a.to(tl.int64) > b.to(tl.int64)
    tl.store(s_ptr + ai, c.to(s_ptr.dtype.element_ty), mask=mask)


# ------------------------------------------------------------------ host ----
_DRIVER = None
_HOT = {}
_ALT = {}


def _driver():
    global _DRIVER
    d = _DRIVER
    if d is None:
        from triton.runtime import driver as _d

        d = _d.active
        _DRIVER = d
    return d


def _bind(ck, g0, tail, with_b=True):
    """Build a lean launcher for ``ck``; ``tail`` are the fixed args."""
    try:
        fr = ck.run
        fn = ck.function
        pm = ck.packed_metadata
    except Exception:
        return None
    d = _driver()
    gd = d.get_current_device
    gs = d.get_current_stream
    if with_b:

        def launch(a, b):
            fr(g0, 1, 1, gs(gd()), fn, pm, None, None, None, a, b, *tail)

    else:

        def launch(a, b):
            fr(g0, 1, 1, gs(gd()), fn, pm, None, None, None, a, *tail)

    return launch


def _block_for(n):
    if n >= 4096:
        return 1024, 4
    b = 32
    while b < n:
        b <<= 1
    if b <= 128:
        return b, 1
    if b <= 512:
        return b, 2
    return b, 4


def _f64_bits(x):
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def _mode_slow(dt):
    if dt is _F64:
        return _M_F64
    if dt.is_floating_point:
        return _M_F32
    return _M_I64


def _check_bcast(self, other):
    try:
        b = torch.broadcast_shapes(self.shape, other.shape)
    except RuntimeError as e:
        raise RuntimeError(str(e)) from None
    if tuple(b) != tuple(self.shape):
        raise RuntimeError(
            "output with shape {} doesn't match the broadcast shape {}".format(
                list(self.shape), list(b)
            )
        )


def _self_view(t, dt):
    if dt is _BOOL:
        if t.dim() == 0:
            return t.unsqueeze(0).view(_I8)
        return t.view(_I8)
    return t


def _flat_launch_cached(s_t, o_t, n, mode, obcast, block, nw, key):
    g0 = (n + block - 1) // block
    ck = _gt_flat[(g0,)](s_t, o_t, n, mode, obcast, block, num_warps=nw)
    launch = _bind(ck, g0, (n, mode, obcast, block))
    if launch is None:

        def launch(a, b, _n=n, _m=mode, _o=obcast, _b=block, _w=nw):
            _gt_flat[((_n + _b - 1) // _b,)](a, b, _n, _m, _o, _b, num_warps=_w)

    _ALT[key] = launch


def _scalar_launch_cached(s_t, val, n, mode, block, nw, key):
    g0 = (n + block - 1) // block
    ck = _gt_flat_s[(g0,)](s_t, val, n, mode, block, num_warps=nw)
    launch = _bind(ck, g0, (val, n, mode, block), False)
    if launch is None:

        def launch(a, b, _v=val, _n=n, _m=mode, _b=block, _w=nw):
            _gt_flat_s[((_n + _b - 1) // _b,)](a, _v, _n, _m, _b, num_warps=_w)

    _ALT[key] = launch


def _strided_meta(sshape, sstride, oshape, ostride):
    r = len(sshape)
    pad = r - len(oshape)
    dims = []
    for i in range(r):
        sz = sshape[i]
        if sz == 1:
            continue
        j = i - pad
        if j < 0:
            os_ = 0
        elif oshape[j] == 1:
            os_ = 0
        else:
            os_ = ostride[j]
        dims.append([sz, sstride[i], os_])
    if not dims:
        return None
    inner_idx = len(dims) - 1
    for i in range(len(dims) - 1, -1, -1):
        if dims[i][1] == 1:
            inner_idx = i
            break
    inner = dims.pop(inner_idx)
    outer = []
    for d in dims:
        if outer:
            p = outer[-1]
            if p[1] * p[0] == d[1] and p[2] * p[0] == d[2]:
                p[0] *= d[0]
                continue
        outer.append(list(d))
    return inner, outer


def _launch_strided(s_ptr, o_ptr, inner, s_is, o_is, val, outer, mode, scalar):
    nout = len(outer)
    outer_n = 1
    for d in outer:
        outer_n *= d[0]
    m = [0] * 8
    acc = 1
    for k in range(nout - 1, -1, -1):
        m[k] = acc
        acc *= outer[k][0]
    ss = [0] * 8
    os_ = [0] * 8
    for k in range(nout):
        ss[k] = outer[k][1]
        os_[k] = outer[k][2]
    block, nw = _block_for(inner)
    grid = ((inner + block - 1) // block, outer_n)
    _gt_strided[grid](
        s_ptr,
        o_ptr,
        inner,
        s_is,
        o_is,
        val,
        m[0],
        m[1],
        m[2],
        m[3],
        m[4],
        m[5],
        m[6],
        m[7],
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        ss[4],
        ss[5],
        ss[6],
        ss[7],
        os_[0],
        os_[1],
        os_[2],
        os_[3],
        os_[4],
        os_[5],
        os_[6],
        os_[7],
        nout,
        scalar,
        mode,
        block,
        num_warps=nw,
    )


def _run_tensor(self, other, mode, n):
    sd = self.dtype
    od = other.dtype
    if other.numel() == 1:
        s_t = _self_view(self, sd)
        o_t = _self_view(other, od)
        if self.is_contiguous():
            key = ("b", sd, od, n)
            launch = _ALT.get(key)
            if launch is None:
                block, nw = _block_for(n)
                _flat_launch_cached(s_t, o_t, n, mode, 1, block, nw, key)
            else:
                launch(s_t, o_t)
            return
        meta = _strided_meta(self.shape, self.stride(), other.shape, other.stride())
        _launch_strided(s_t, o_t, meta[0][0], meta[0][1], 0, 0, meta[1], mode, False)
        return
    if self.is_contiguous() and other.is_contiguous() and other.shape == self.shape:
        s_t = _self_view(self, sd)
        o_t = _self_view(other, od)
        key = ("f", sd, od, n)
        launch = _ALT.get(key)
        if launch is None:
            block, nw = _block_for(n)
            _flat_launch_cached(s_t, o_t, n, mode, 0, block, nw, key)
        else:
            launch(s_t, o_t)
        return
    s_t = _self_view(self, sd)
    o_t = _self_view(other, od)
    meta = _strided_meta(self.shape, self.stride(), other.shape, other.stride())
    _launch_strided(
        s_t, o_t, meta[0][0], meta[0][1], meta[0][2], 0, meta[1], mode, False
    )


def _run_scalar(self, other):
    sd = self.dtype
    if isinstance(other, bool):
        fval = float(other)
        ival = int(other)
        is_float = False
    elif isinstance(other, int):
        fval = float(other)
        ival = other
        is_float = False
    else:
        fval = float(other)
        ival = 0
        is_float = True
    if sd is _F64:
        mode = _M_F64
        val = _f64_bits(fval)
    elif sd.is_floating_point:
        mode = _M_F32
        val = fval
    elif is_float:
        mode = _M_F32
        val = fval
    else:
        mode = _M_I64
        val = ival
    n = self.numel()
    if n == 0:
        return
    s_t = _self_view(self, sd)
    if self.is_contiguous():
        key = ("s", sd, val, n)
        launch = _ALT.get(key)
        if launch is None:
            block, nw = _block_for(n)
            _scalar_launch_cached(s_t, val, n, mode, block, nw, key)
        else:
            launch(s_t, s_t)
        return
    meta = _strided_meta(self.shape, self.stride(), (1,), (0,))
    _launch_strided(s_t, s_t, meta[0][0], meta[0][1], 0, val, meta[1], mode, True)


# ------------------------------------------------------------------- entry --
def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _use_triton_kernel_greater(self, other):
    if not _use_triton_kernel(self):
        return False
    if torch.is_tensor(other):
        return _use_triton_kernel(other)
    return True


def greater_(self, other):
    logger.debug("GEMS_MTHREADS GREATER_")

    if not _use_triton_kernel_greater(self, other):
        return default_greater_(self, other)

    with torch_device_fn.device(self.device):
        return _launch_greater_(self, other)


def _launch_greater_(self, other):
    if not torch.is_tensor(other):
        _run_scalar(self, other)
        return self
    sd = self.dtype
    od = other.dtype
    n = self.numel()
    if sd is od:
        if (
            sd is not _BOOL
            and other.shape == self.shape
            and other.numel() != 1
            and self.is_contiguous()
            and other.is_contiguous()
        ):
            key = (sd, n)
            launch = _HOT.get(key)
            if launch is None:
                if n == 0:
                    return self
                mode = _MODE_BY_DTYPE.get(sd)
                if mode is None:
                    mode = _mode_slow(sd)
                if n >= 1048576:
                    block, nw = 2048, 8
                elif n >= 4096:
                    block, nw = 1024, 4
                else:
                    block, nw = _block_for(n)
                g0 = (n + block - 1) // block
                ck = _gt_flat[(g0,)](self, other, n, mode, 0, block, num_warps=nw)
                launch = _bind(ck, g0, (n, mode, 0, block))
                if launch is None:

                    def launch(a, b, _n=n, _m=mode, _b=block, _w=nw):
                        _gt_flat[((_n + _b - 1) // _b,)](
                            a, b, _n, _m, 0, _b, num_warps=_w
                        )

                _HOT[key] = launch
                return self
            launch(self, other)
            return self
        mode = _MODE_BY_DTYPE.get(sd)
        if mode is None:
            mode = _mode_slow(sd)
    else:
        p = torch.promote_types(sd, od)
        mode = _MODE_BY_DTYPE.get(p)
        if mode is None:
            mode = _mode_slow(p)
    if other.shape != self.shape:
        _check_bcast(self, other)
    if n == 0:
        return self
    _run_tensor(self, other, mode, n)
    return self
