# Copyright 2026 FlagOS Contributors
# Copyright (c) 2024 FlagGems. All rights reserved.
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

"""Triton elementwise true division (aten::true_divide, all common call forms).

Supported call shapes:
    run(a, b)                 -> new tensor (broadcasting allowed)
    run(a, b, out=o)          -> writes into o, returns o
    run(a, scalar) / run(scalar, a)
    run(scalar, scalar)

Semantics: ATen ``true_divide`` promotes integer/bool operands to the default
floating dtype and always computes with floating opmath (fp32, or fp64 for
float64 operands), then stores into the promoted output dtype.  Division is a
real IEEE division so +-inf / NaN / signed-zero behaviour matches torch.
"""

import logging

import torch
import triton
import triton.language as tl

"""Bounded compiled-code cache using public Triton launchers.

Only argument type, alignment and scalar metadata specialize code; tensor data
and returned values are never retained. Current stream and runtime hooks are
resolved by each public CompiledKernel grid launch.
"""
_hygon_launch_cached__CACHE = {}


def _hygon_launch_cached__launch(kernel, grid, *args, **kwargs):
    names = kernel.arg_names
    full_args = args + tuple((kwargs[name] for name in names[len(args) :]))
    options = {name: value for (name, value) in kwargs.items() if name not in names}
    signature = tuple(
        (
            (
                (value.dtype, value.data_ptr() % 16 == 0)
                if isinstance(value, torch.Tensor)
                else value
            )
            for value in full_args
        )
    )
    key = (
        kernel,
        torch.cuda.current_device(),
        signature,
        tuple(sorted(options.items())),
    )
    compiled = _hygon_launch_cached__CACHE.get(key)
    if compiled is None:
        compiled = kernel[grid](*full_args, **options)
        if compiled is not None:
            if len(_hygon_launch_cached__CACHE) >= 256:
                _hygon_launch_cached__CACHE.pop(next(iter(_hygon_launch_cached__CACHE)))
            _hygon_launch_cached__CACHE[key] = compiled
    else:
        dims = tuple(grid) + (1,) * (3 - len(grid))
        compiled[dims](*full_args, stream=torch.cuda.current_stream().cuda_stream)


launch = _hygon_launch_cached__launch
logger = logging.getLogger(__name__)
_BLOCK = 2048
_WARPS = 4
_MAX_DIMS = 8


@triton.jit
def _div_flat_kernel(
    a_ptr, b_ptr, o_ptr, n, sa, sb, so, BLOCK: tl.constexpr, F64: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(a_ptr + off * sa, mask=m, other=0.0)
    b = tl.load(b_ptr + off * sb, mask=m, other=0.0)
    if F64:
        r = a.to(tl.float64) / b.to(tl.float64)
    else:
        r = a.to(tl.float32) / b.to(tl.float32)
    tl.store(o_ptr + off * so, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _div_scalar_flat_kernel(
    a_ptr,
    o_ptr,
    sv,
    n,
    sa,
    so,
    BLOCK: tl.constexpr,
    S_FIRST: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(a_ptr + off * sa, mask=m, other=0.0)
    if F64:
        av = a.to(tl.float64)
    else:
        av = a.to(tl.float32)
    if S_FIRST:
        r = sv / av
    else:
        r = av / sv
    tl.store(o_ptr + off * so, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _div_bcast_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    sv,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
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
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    BLOCK: tl.constexpr,
    B_SCALAR: tl.constexpr,
    S_FIRST: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    rem = idx
    i = rem % s7
    rem = rem // s7
    ao = i * a7
    bo = i * b7
    co = i * c7
    i = rem % s6
    rem = rem // s6
    ao += i * a6
    bo += i * b6
    co += i * c6
    i = rem % s5
    rem = rem // s5
    ao += i * a5
    bo += i * b5
    co += i * c5
    i = rem % s4
    rem = rem // s4
    ao += i * a4
    bo += i * b4
    co += i * c4
    i = rem % s3
    rem = rem // s3
    ao += i * a3
    bo += i * b3
    co += i * c3
    i = rem % s2
    rem = rem // s2
    ao += i * a2
    bo += i * b2
    co += i * c2
    i = rem % s1
    rem = rem // s1
    ao += i * a1
    bo += i * b1
    co += i * c1
    i = rem % s0
    ao += i * a0
    bo += i * b0
    co += i * c0
    a = tl.load(a_ptr + ao, mask=m, other=0.0)
    if B_SCALAR:
        if F64:
            av = a.to(tl.float64)
        else:
            av = a.to(tl.float32)
        if S_FIRST:
            r = sv / av
        else:
            r = av / sv
    else:
        b = tl.load(b_ptr + bo, mask=m, other=0.0)
        if F64:
            r = a.to(tl.float64) / b.to(tl.float64)
        else:
            r = a.to(tl.float32) / b.to(tl.float32)
    tl.store(o_ptr + co, r.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _div_ss_kernel(o_ptr, a, b, n, BLOCK: tl.constexpr, F64: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    if F64:
        r = a.to(tl.float64) / b.to(tl.float64)
    else:
        r = a.to(tl.float32) / b.to(tl.float32)
    tl.store(o_ptr + off, r.to(o_ptr.dtype.element_ty), mask=m)


def _compute_dtype(a, b):
    dt = torch.result_type(a, b)
    if not (dt.is_floating_point or dt.is_complex):
        dt = torch.get_default_dtype()
    return dt


def _flat_stride(t, n):
    """Return a single linear stride when the tensor can be walked flat."""
    if t.numel() == 1:
        return 0
    if t.is_contiguous():
        return 1
    if t.dim() == 1:
        return t.stride(0)
    return None


def _bcast_strides(shape, t, pad):
    nd = len(shape)
    off = nd - t.dim()
    st = [0] * _MAX_DIMS
    for k in range(t.dim()):
        d = off + k
        if shape[d] != 1:
            st[pad + d] = t.stride(k)
    return st


def _launch_bcast(a, b, o, sv, shape, n, b_scalar, s_first, f64):
    nd = len(shape)
    if nd > _MAX_DIMS:
        raise NotImplementedError("true_divide: too many dimensions")
    pad = _MAX_DIMS - nd
    sizes = [1] * pad + list(shape)
    ast = _bcast_strides(shape, a, pad)
    if b_scalar:
        bst = [0] * _MAX_DIMS
    else:
        bst = _bcast_strides(shape, b, pad)
    ost = _bcast_strides(shape, o, pad)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    launch(
        _div_bcast_kernel,
        grid,
        a,
        b,
        o,
        sv,
        n,
        *sizes,
        *ast,
        *bst,
        *ost,
        BLOCK=_BLOCK,
        B_SCALAR=b_scalar,
        S_FIRST=s_first,
        F64=f64,
        num_warps=_WARPS,
    )


def _tensor_tensor(a, b, out):
    dt = _compute_dtype(a, b)
    f64 = dt is torch.float64
    same = a.shape == b.shape
    shape = a.shape if same else torch.broadcast_shapes(a.shape, b.shape)
    o = torch.empty(shape, dtype=dt, device=a.device) if out is None else out
    n = o.numel()
    if n == 0:
        return o
    if same:
        sa = _flat_stride(a, n)
        if sa is not None:
            sb = _flat_stride(b, n)
            if sb is not None:
                so = _flat_stride(o, n)
                if so is not None:
                    grid = ((n + _BLOCK - 1) // _BLOCK,)
                    launch(
                        _div_flat_kernel,
                        grid,
                        a,
                        b,
                        o,
                        n,
                        sa,
                        sb,
                        so,
                        BLOCK=_BLOCK,
                        F64=f64,
                        num_warps=_WARPS,
                    )
                    return o
    _launch_bcast(a, b, o, 0.0, shape, n, False, False, f64)
    return o


def _tensor_scalar(t, s, s_first, out):
    if isinstance(s, bool):
        s = int(s)
    dt = _compute_dtype(t, s)
    f64 = dt is torch.float64
    o = torch.empty(t.shape, dtype=dt, device=t.device) if out is None else out
    n = o.numel()
    if n == 0:
        return o
    sa = _flat_stride(t, n)
    so = _flat_stride(o, n)
    if sa is not None and so is not None:
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        launch(
            _div_scalar_flat_kernel,
            grid,
            t,
            o,
            s,
            n,
            sa,
            so,
            BLOCK=_BLOCK,
            S_FIRST=s_first,
            F64=f64,
            num_warps=_WARPS,
        )
        return o
    _launch_bcast(t, t, o, s, t.shape, n, True, s_first, f64)
    return o


def _scalar_scalar(a, b, out):
    if isinstance(a, bool):
        a = int(a)
    if isinstance(b, bool):
        b = int(b)
    dt = _compute_dtype(a, b)
    f64 = dt is torch.float64
    if out is None:
        out = torch.empty(
            (), dtype=dt, device=torch.device("cuda", torch.cuda.current_device())
        )
    n = out.numel()
    if n == 0:
        return out
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    launch(_div_ss_kernel, grid, out, a, b, n, BLOCK=_BLOCK, F64=f64, num_warps=1)
    return out


def true_divide(self, other, out=None):
    logger.debug("GEMS_HYGON TRUE_DIVIDE")
    a = self
    if type(a) is torch.Tensor:
        if type(other) is torch.Tensor:
            return _tensor_tensor(a, other, out)
        return _tensor_scalar(a, other, False, out)
    if isinstance(a, torch.Tensor):
        if isinstance(other, torch.Tensor):
            return _tensor_tensor(a, other, out)
        return _tensor_scalar(a, other, False, out)
    if isinstance(other, torch.Tensor):
        return _tensor_scalar(other, a, True, out)
    return _scalar_scalar(a, other, out)


def true_divide_tensor(self, other):
    return true_divide(self, other)


def true_divide_(self, other):
    logger.debug("GEMS_HYGON TRUE_DIVIDE_")
    dtype = _compute_dtype(self, other)
    if not torch.can_cast(dtype, self.dtype):
        raise RuntimeError("result type cannot be cast to the in-place output dtype")
    if (
        isinstance(other, torch.Tensor)
        and torch.broadcast_shapes(self.shape, other.shape) != self.shape
    ):
        raise RuntimeError("output shape does not match the broadcast shape")
    return true_divide(self, other, out=self)


def true_divide_tensor_(self, other):
    return true_divide_(self, other)
