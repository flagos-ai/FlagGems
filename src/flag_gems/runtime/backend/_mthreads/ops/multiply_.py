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

"""Triton implementation of aten::multiply_ (in-place elementwise multiply).

The host wrapper only inspects metadata (shape / stride / dtype / device) and
launches device kernels.  The write always lands in ``self``'s existing
storage, so storage identity, strides and dtype are preserved and the very
same Python tensor object is returned.

Per-call host cost matters for this operator (one multiply per element), so
after the first launch of a given specialization the compiled kernel runner is
kept and invoked directly instead of re-entering the Triton JIT dispatch path.
"""

import logging
import numbers

import torch
import triton
import triton.language as tl

from flag_gems.ops.multiply_ import multiply_ as default_multiply_
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def _mul_flat_kernel(
    a_ptr,
    b_ptr,
    n,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    if BOOL:
        r = a & b
    elif UPCAST:
        r = a.to(tl.float32) * b.to(tl.float32)
    else:
        r = a * b
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _mul_bcast_kernel(
    a_ptr,
    b_ptr,
    n,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr)
    if BOOL:
        r = a & b
    elif UPCAST:
        r = a.to(tl.float32) * b.to(tl.float32)
    else:
        r = a * b
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _mul_scalar_kernel(
    a_ptr,
    s,
    n,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    if BOOL:
        r = a & (s != 0).to(tl.int8)
    elif UPCAST:
        r = a.to(tl.float32) * s
    else:
        r = a * s
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _mul_strided_kernel(
    a_ptr,
    b_ptr,
    inner,
    tiles,
    S: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // tiles
    col = (pid % tiles) * BLOCK + tl.arange(0, BLOCK)
    mask = col < inner
    lin = row.to(tl.int64)
    a_base = lin * 0
    b_base = lin * 0
    for d in tl.static_range(NDIM - 2, -1, -1):
        idx = lin % S[d]
        lin = lin // S[d]
        a_base += idx * AS[d]
        b_base += idx * BS[d]
    a_off = a_base + col.to(tl.int64) * AS[NDIM - 1]
    b_off = b_base + col.to(tl.int64) * BS[NDIM - 1]
    a = tl.load(a_ptr + a_off, mask=mask)
    b = tl.load(b_ptr + b_off, mask=mask)
    if BOOL:
        r = a & b
    elif UPCAST:
        r = a.to(tl.float32) * b.to(tl.float32)
    else:
        r = a * b
    tl.store(a_ptr + a_off, r, mask=mask)


@triton.jit
def _mul_strided_scalar_kernel(
    a_ptr,
    s,
    inner,
    tiles,
    S: tl.constexpr,
    AS: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    UPCAST: tl.constexpr,
    BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // tiles
    col = (pid % tiles) * BLOCK + tl.arange(0, BLOCK)
    mask = col < inner
    lin = row.to(tl.int64)
    a_base = lin * 0
    for d in tl.static_range(NDIM - 2, -1, -1):
        idx = lin % S[d]
        lin = lin // S[d]
        a_base += idx * AS[d]
    a_off = a_base + col.to(tl.int64) * AS[NDIM - 1]
    a = tl.load(a_ptr + a_off, mask=mask)
    if BOOL:
        r = a & (s != 0).to(tl.int8)
    elif UPCAST:
        r = a.to(tl.float32) * s
    else:
        r = a * s
    tl.store(a_ptr + a_off, r, mask=mask)


# --------------------------------------------------------------------------
# host helpers
# --------------------------------------------------------------------------
_NUM_WARPS = 4
_ITEMSIZE = {}
_LAUNCHERS = {}


def _itemsize_of(dtype):
    size = _ITEMSIZE.get(dtype)
    if size is None:
        size = getattr(dtype, "itemsize", None)
        if size is None:
            size = torch.empty((), dtype=dtype).element_size()
        _ITEMSIZE[dtype] = size
    return size


def _pick_block(n, itemsize):
    block = 512 if itemsize >= 8 else 1024
    if n < block:
        block = triton.next_power_of_2(n if n > 0 else 1)
    return int(block)


def _run_flat(key, grid0, av, bv, n, block, upcast, bool_mode):
    runner = _LAUNCHERS.get(key)
    if runner is not None:
        runner(av, bv, n, block, upcast, bool_mode)
        return
    g3 = (grid0, 1, 1)
    compiled = _mul_flat_kernel[g3](
        av, bv, n, BLOCK=block, UPCAST=upcast, BOOL=bool_mode, num_warps=_NUM_WARPS
    )
    _LAUNCHERS[key] = compiled[g3]


def _run_bcast(key, grid0, av, bv, n, block, upcast, bool_mode):
    runner = _LAUNCHERS.get(key)
    if runner is not None:
        runner(av, bv, n, block, upcast, bool_mode)
        return
    g3 = (grid0, 1, 1)
    compiled = _mul_bcast_kernel[g3](
        av, bv, n, BLOCK=block, UPCAST=upcast, BOOL=bool_mode, num_warps=_NUM_WARPS
    )
    _LAUNCHERS[key] = compiled[g3]


def _run_scalar(key, grid0, av, scalar, n, block, upcast, bool_mode):
    runner = _LAUNCHERS.get(key)
    if runner is not None:
        runner(av, scalar, n, block, upcast, bool_mode)
        return
    g3 = (grid0, 1, 1)
    compiled = _mul_scalar_kernel[g3](
        av,
        scalar,
        n,
        BLOCK=block,
        UPCAST=upcast,
        BOOL=bool_mode,
        num_warps=_NUM_WARPS,
    )
    _LAUNCHERS[key] = compiled[g3]


def _run_strided(
    key, grid0, a, b, inner, tiles, S, AS, BS, ndim, block, upcast, bool_mode
):
    runner = _LAUNCHERS.get(key)
    if runner is not None:
        runner(a, b, inner, tiles, S, AS, BS, ndim, block, upcast, bool_mode)
        return
    g3 = (grid0, 1, 1)
    compiled = _mul_strided_kernel[g3](
        a,
        b,
        inner,
        tiles,
        S=S,
        AS=AS,
        BS=BS,
        NDIM=ndim,
        BLOCK=block,
        UPCAST=upcast,
        BOOL=bool_mode,
        num_warps=_NUM_WARPS,
    )
    _LAUNCHERS[key] = compiled[g3]


def _run_strided_scalar(
    key, grid0, a, scalar, inner, tiles, S, AS, ndim, block, upcast, bool_mode
):
    runner = _LAUNCHERS.get(key)
    if runner is not None:
        runner(a, scalar, inner, tiles, S, AS, ndim, block, upcast, bool_mode)
        return
    g3 = (grid0, 1, 1)
    compiled = _mul_strided_scalar_kernel[g3](
        a,
        scalar,
        inner,
        tiles,
        S=S,
        AS=AS,
        NDIM=ndim,
        BLOCK=block,
        UPCAST=upcast,
        BOOL=bool_mode,
        num_warps=_NUM_WARPS,
    )
    _LAUNCHERS[key] = compiled[g3]


def _aligned(t):
    return (t.data_ptr() & 15) == 0


def _launch_strided(a, b, scalar, upcast, bool_mode, itemsize):
    nd = a.dim()
    ashape = tuple(a.shape)
    astride = tuple(a.stride())
    if b is not None:
        pad = nd - b.dim()
        bshape = (1,) * pad + tuple(b.shape)
        bstride = (0,) * pad + tuple(b.stride())
        bstride = tuple(0 if bshape[i] == 1 else bstride[i] for i in range(nd))
    else:
        bstride = (0,) * nd

    dims = [i for i in range(nd) if ashape[i] != 1]
    sh = [ashape[i] for i in dims]
    ast = [astride[i] for i in dims]
    bst = [bstride[i] for i in dims]
    if not sh:
        sh = [1]
        ast = [1]
        bst = [0]

    if len(sh) > 1:
        vidx = -1
        for i, s in enumerate(ast):
            if s == 1:
                vidx = i
                break
        if vidx < 0:
            best = None
            for i, s in enumerate(ast):
                if s == 0:
                    continue
                size = s if s > 0 else -s
                if best is None or size < best[0]:
                    best = (size, i)
            vidx = best[1] if best is not None else len(sh) - 1
        if vidx != len(sh) - 1:
            order = [i for i in range(len(sh)) if i != vidx] + [vidx]
            sh = [sh[i] for i in order]
            ast = [ast[i] for i in order]
            bst = [bst[i] for i in order]

    cs = [sh[0]]
    ca = [ast[0]]
    cb = [bst[0]]
    for i in range(1, len(sh)):
        if ca[-1] == ast[i] * sh[i] and cb[-1] == bst[i] * sh[i]:
            cs[-1] = cs[-1] * sh[i]
        else:
            cs.append(sh[i])
            ca.append(ast[i])
            cb.append(bst[i])

    inner = int(cs[-1])
    outer = 1
    for s in cs[:-1]:
        outer *= int(s)
    ndim = len(cs)
    block = _pick_block(inner, itemsize)
    tiles = (inner + block - 1) // block
    grid0 = tiles * outer
    S = tuple(cs)
    AS = tuple(ca)
    BS = tuple(cb)

    if b is None:
        key = (
            3,
            S,
            AS,
            ndim,
            block,
            upcast,
            bool_mode,
            a.dtype,
            inner,
            tiles,
            _aligned(a),
        )
        _run_strided_scalar(
            key, grid0, a, scalar, inner, tiles, S, AS, ndim, block, upcast, bool_mode
        )
    else:
        key = (
            4,
            S,
            AS,
            BS,
            ndim,
            block,
            upcast,
            bool_mode,
            a.dtype,
            b.dtype,
            inner,
            tiles,
            _aligned(a),
            _aligned(b),
        )
        _run_strided(
            key,
            grid0,
            a,
            b,
            inner,
            tiles,
            S,
            AS,
            BS,
            ndim,
            block,
            upcast,
            bool_mode,
        )


def _launch(a, b, scalar):
    n = a.numel()
    if n == 0:
        return
    adt = a.dtype
    bool_mode = adt is torch.bool
    upcast = adt is torch.float16 or adt is torch.bfloat16
    av = a
    bv = b
    if bool_mode:
        av = a.view(torch.int8)
        if b is not None and b.dtype is torch.bool:
            bv = b.view(torch.int8)
    itemsize = _itemsize_of(adt)

    if a.is_contiguous():
        block = _pick_block(n, itemsize)
        grid0 = (n + block - 1) // block
        if b is None:
            key = (1, n, block, upcast, bool_mode, adt, scalar, _aligned(av))
            _run_scalar(key, grid0, av, scalar, n, block, upcast, bool_mode)
            return
        if b.numel() == 1:
            key = (
                2,
                n,
                block,
                upcast,
                bool_mode,
                adt,
                bv.dtype,
                _aligned(av),
                _aligned(bv),
            )
            _run_bcast(key, grid0, av, bv, n, block, upcast, bool_mode)
            return
        if b.shape == a.shape and b.is_contiguous():
            key = (
                5,
                n,
                block,
                upcast,
                bool_mode,
                adt,
                bv.dtype,
                _aligned(av),
                _aligned(bv),
            )
            _run_flat(key, grid0, av, bv, n, block, upcast, bool_mode)
            return
        _launch_strided(av, bv, None, upcast, bool_mode, itemsize)
        return
    _launch_strided(av, bv, scalar, upcast, bool_mode, itemsize)


class _MulInplace(torch.autograd.Function):
    """Autograd-aware wrapper used only when a gradient is actually required."""

    @staticmethod
    def forward(ctx, self_t, other_t, scalar):
        pre = None
        if other_t is not None and other_t.requires_grad:
            pre = self_t.detach().clone()
        _launch(self_t, other_t, scalar)
        ctx.mark_dirty(self_t)
        if other_t is not None:
            ctx.save_for_backward(other_t)
        ctx.pre_value = pre
        ctx.scalar = scalar
        return self_t

    @staticmethod
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        if saved:
            other = saved[0]
            gself = grad * other
            gother = None
            if ctx.pre_value is not None:
                gother = grad * ctx.pre_value
        else:
            gself = grad * ctx.scalar
            gother = None
        return gself, gother, None


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _use_triton_kernel_pair(self, other):
    if not _use_triton_kernel(self):
        return False
    if isinstance(other, torch.Tensor):
        return _use_triton_kernel(other)
    return True


def multiply_(self, other):
    logger.debug("GEMS_MTHREADS MULTIPLY_")

    if not _use_triton_kernel_pair(self, other):
        return default_multiply_(self, other)

    with torch_device_fn.device(self.device):
        return _launch_multiply_(self, other)


def _launch_multiply_(self, other):
    if not isinstance(self, torch.Tensor):
        raise TypeError("multiply_ expects a tensor as the first argument")

    ot = other if isinstance(other, torch.Tensor) else None
    scalar = None

    if ot is not None:
        if ot.dtype is not self.dtype:
            result_dtype = torch.result_type(self, ot)
            if not torch.can_cast(result_dtype, self.dtype):
                raise RuntimeError(
                    "result type {} can't be cast to the desired output type {}".format(
                        result_dtype, self.dtype
                    )
                )
        elif self.is_complex():
            raise RuntimeError("multiply_ does not support complex dtypes")
        if ot.shape != self.shape:
            out_shape = None
            try:
                out_shape = tuple(
                    torch.broadcast_shapes(tuple(self.shape), tuple(ot.shape))
                )
            except RuntimeError:
                out_shape = None
            if out_shape is None or out_shape != tuple(self.shape):
                raise RuntimeError(
                    "output with shape {} doesn't match the broadcast shape {}".format(
                        list(self.shape),
                        list(out_shape) if out_shape is not None else list(ot.shape),
                    )
                )
    else:
        if not isinstance(other, numbers.Number) or isinstance(other, complex):
            raise TypeError(
                "multiply_ expects a tensor or a real number as the second argument"
            )
        if not self.dtype.is_floating_point:
            result_dtype = torch.result_type(self, other)
            if not torch.can_cast(result_dtype, self.dtype):
                raise RuntimeError(
                    "result type {} can't be cast to the desired output type {}".format(
                        result_dtype, self.dtype
                    )
                )
        if self.dtype is torch.float64:
            ot = torch.tensor(float(other), dtype=torch.float64, device=self.device)
        elif self.dtype.is_floating_point:
            scalar = float(other)
        else:
            scalar = int(other)

    if torch.is_grad_enabled() and (
        self.requires_grad or (ot is not None and ot.requires_grad)
    ):
        if not torch.is_inference_mode_enabled():
            return _MulInplace.apply(self, ot, scalar)

    _launch(self, ot, scalar)
    return self
