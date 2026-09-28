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

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops.arctan2 import arctan2 as default_arctan2

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# out dtype codes
_F32 = 0
_F64 = 1
_F16 = 2
_BF16 = 3


@triton.jit
def _atan2_core(
    x, y, IN1_LOW: tl.constexpr, IN2_LOW: tl.constexpr, OUT_KIND: tl.constexpr
):
    # Compute always in fp32 for fp16/bf16/fp32 inputs; fp64 stays fp64.
    # (fp16->fp32 path must never use masked loads/stores: the MTGPU llc
    # backend segfaults in ISel on the fp16-cast + predicate pattern.)
    if IN1_LOW:
        x = x.to(tl.float32)
    if IN2_LOW:
        y = y.to(tl.float32)
    z = libdevice.atan2(x, y)
    if OUT_KIND == 0:
        z = z.to(tl.float32)
    elif OUT_KIND == 1:
        z = z.to(tl.float64)
    elif OUT_KIND == 2:
        z = z.to(tl.float16)
    elif OUT_KIND == 3:
        z = z.to(tl.bfloat16)
    return z


@triton.jit
def _atan2_1d_main(
    x_ptr,
    y_ptr,
    o_ptr,
    BLOCK: tl.constexpr,
    IN1_LOW: tl.constexpr,
    IN2_LOW: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    # Aligned region only: n_main is a multiple of BLOCK, so no mask/clamp.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    z = _atan2_core(x, y, IN1_LOW, IN2_LOW, OUT_KIND)
    tl.store(o_ptr + offs, z)


@triton.jit
def _atan2_1d_tail(
    x_ptr,
    y_ptr,
    o_ptr,
    start,
    n,
    BLOCK: tl.constexpr,
    IN1_LOW: tl.constexpr,
    IN2_LOW: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    # Tail region (0 < n - start < BLOCK): clamp OOB lanes to n-1 instead of
    # masking, so the fp16-cast path never emits predicates (llc crash).
    offs = start + tl.arange(0, BLOCK)
    safe = tl.minimum(offs, n - 1)
    x = tl.load(x_ptr + safe)
    y = tl.load(y_ptr + safe)
    z = _atan2_core(x, y, IN1_LOW, IN2_LOW, OUT_KIND)
    tl.store(o_ptr + safe, z)


@triton.jit
def _atan2_2d(
    x_ptr,
    y_ptr,
    o_ptr,
    xs0,
    xs1,
    ys0,
    ys1,
    inner,
    BLOCK: tl.constexpr,
    IN1_LOW: tl.constexpr,
    IN2_LOW: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    cblocks = tl.cdiv(inner, BLOCK)
    row = pid // cblocks
    cb = pid % cblocks
    c = cb * BLOCK + tl.arange(0, BLOCK)
    csafe = tl.minimum(c, inner - 1)
    x = tl.load(x_ptr + row.to(tl.int64) * xs0 + csafe.to(tl.int64) * xs1)
    y = tl.load(y_ptr + row.to(tl.int64) * ys0 + csafe.to(tl.int64) * ys1)
    z = _atan2_core(x, y, IN1_LOW, IN2_LOW, OUT_KIND)
    tl.store(o_ptr + row.to(tl.int64) * inner + csafe, z)


@triton.jit
def _atan2_nd(
    x_ptr,
    y_ptr,
    o_ptr,
    xs_ptr,
    ys_ptr,
    shape_ptr,
    n,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
    IN1_LOW: tl.constexpr,
    IN2_LOW: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    idx32 = pid * BLOCK + tl.arange(0, BLOCK)
    safe32 = tl.minimum(idx32, n - 1)
    idxs = safe32.to(tl.int64)
    rem = idxs
    x_off = tl.zeros([BLOCK], dtype=tl.int64)
    y_off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(NDIM):
        k = NDIM - 1 - d
        s = tl.load(shape_ptr + k).to(tl.int64)
        c = rem % s
        x_off += c * tl.load(xs_ptr + k).to(tl.int64)
        y_off += c * tl.load(ys_ptr + k).to(tl.int64)
        rem = rem // s
    x = tl.load(x_ptr + x_off)
    y = tl.load(y_ptr + y_off)
    z = _atan2_core(x, y, IN1_LOW, IN2_LOW, OUT_KIND)
    tl.store(o_ptr + idxs, z)


def _dtype_code(dt):
    if dt == torch.float32:
        return _F32
    if dt == torch.float64:
        return _F64
    if dt == torch.float16:
        return _F16
    if dt == torch.bfloat16:
        return _BF16
    raise TypeError(f"unsupported dtype for atan2: {dt}")


def _collapsible(t, S):
    nd = t.dim()
    if nd <= 2:
        return True
    st = t.stride()
    for d in range(0, nd - 2):
        expect = st[nd - 2]
        for dd in range(d + 1, nd - 1):
            expect *= S[dd]
        if st[d] != expect:
            return False
    return True


_BLOCK = 1024
_BLOCK_SMALL = 256
_BLOCK_MID = 512
_SMALL_N = 32768
_MID_N = 1 << 20
_LOWP = (torch.float16, torch.bfloat16)


def _specialized_arctan2(input, other):
    in1 = input
    in2 = other
    if in1.dtype == in2.dtype:
        common = in1.dtype
        shape = in1.shape
        same_shape = shape == in2.shape
        if not same_shape:
            shape = torch.broadcast_shapes(shape, in2.shape)
    else:
        common = torch.promote_types(in1.dtype, in2.dtype)
        shape = torch.broadcast_shapes(in1.shape, in2.shape)
        same_shape = shape == in1.shape and shape == in2.shape

    out = torch.empty(shape, dtype=common, device=in1.device)
    n = out.numel()
    if n == 0:
        return out
    code = _dtype_code(common)
    in1_low = in1.dtype in _LOWP
    in2_low = in2.dtype in _LOWP

    if same_shape and in1.is_contiguous() and in2.is_contiguous():
        # n-dependent BLOCK: fine grids hide latency on tiny kernels, coarse
        # blocks suit 1G-scale DRAM scheduling, mid sizes favor 512.
        if n < _SMALL_N:
            block = _BLOCK_SMALL
        elif n < _MID_N:
            block = _BLOCK_MID
        else:
            block = _BLOCK
        n_main = (n // block) * block
        if n_main:
            _atan2_1d_main[(n_main // block,)](
                in1,
                in2,
                out,
                BLOCK=block,
                IN1_LOW=in1_low,
                IN2_LOW=in2_low,
                OUT_KIND=code,
            )
        tail = n - n_main
        if tail:
            bt = 1
            while bt < tail:
                bt *= 2
            _atan2_1d_tail[(1,)](
                in1,
                in2,
                out,
                n_main,
                n,
                BLOCK=bt,
                IN1_LOW=in1_low,
                IN2_LOW=in2_low,
                OUT_KIND=code,
            )
        return out

    ex = in1.expand(shape)
    ey = in2.expand(shape)
    if len(shape) == 0:
        inner = 1
        outer = 1
        xs0 = ys0 = 0
        xs1 = ys1 = 0
        use_2d = True
    elif len(shape) == 1:
        inner = shape[0]
        outer = 1
        xs0 = ys0 = 0
        xs1 = ex.stride(0)
        ys1 = ey.stride(0)
        use_2d = True
    elif _collapsible(ex, shape) and _collapsible(ey, shape):
        inner = shape[-1]
        outer = n // inner
        xs0 = ex.stride(-2)
        ys0 = ey.stride(-2)
        xs1 = ex.stride(-1)
        ys1 = ey.stride(-1)
        use_2d = True
    else:
        use_2d = False

    if use_2d:
        grid = (triton.cdiv(inner, _BLOCK) * outer,)
        _atan2_2d[grid](
            ex,
            ey,
            out,
            xs0,
            xs1,
            ys0,
            ys1,
            inner,
            BLOCK=_BLOCK,
            IN1_LOW=in1_low,
            IN2_LOW=in2_low,
            OUT_KIND=code,
        )
        return out

    ndim = len(shape)
    dev = in1.device
    xs_t = torch.tensor(list(ex.stride()), dtype=torch.int32, device=dev)
    ys_t = torch.tensor(list(ey.stride()), dtype=torch.int32, device=dev)
    sh_t = torch.tensor(list(shape), dtype=torch.int32, device=dev)
    grid = (triton.cdiv(n, _BLOCK),)
    _atan2_nd[grid](
        in1,
        in2,
        out,
        xs_t,
        ys_t,
        sh_t,
        n,
        BLOCK=_BLOCK,
        NDIM=ndim,
        IN1_LOW=in1_low,
        IN2_LOW=in2_low,
        OUT_KIND=code,
    )
    return out


def arctan2(input, other):
    logger.debug("GEMS_MTHREADS ARCTAN2")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
        and isinstance(other, torch.Tensor)
        and other.device.type == "musa"
        and other.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_arctan2(input, other)
    return default_arctan2(input, other)
