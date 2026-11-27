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

"""XPU override for ``aten::_foreach_sin``.

A homogeneous list of at most sixteen tensors is processed by one fused launch:
``program_id(1)`` selects the member, so sixteen tensors cost one launch
instead of sixteen.  Direct pointer arguments (rather than a metadata buffer
reloaded on device) let the backend vectorize the loads and stores.  Tensors
with the same element count share one bulk output allocation, which removes the
per-member allocator round trip.

Longer lists, complex lists and dtypes without a fast kernel fall back to the
shared ``_foreach_unary`` executor, which keeps the family's launch count tied
to the number of ``(device, dtype)`` groups and preserves the complex and fp64
paths.
"""

import logging

import torch
import triton
import triton.language as tl

from ..utils.foreach import _STATS, check_tensor_list
from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

_FAST_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

# integer / bool inputs are reinterpreted with a zero-copy dtype view before the
# device kernel;  the promotion to float32 happens inside the kernel.
_VIEW_DTYPE = {
    torch.bool: torch.uint8,
    torch.uint8: torch.uint8,
    torch.int8: torch.int8,
    torch.int16: torch.int16,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
}

# members per fused launch; longer uniform lists are chunked
_MAX_FUSED = 16


@triton.jit
def _sin_elem(x_ptr, y_ptr, offs, mask, FP64: tl.constexpr, MASKED: tl.constexpr):
    if MASKED:
        x = tl.load(x_ptr + offs, mask=mask, other=0)
    else:
        x = tl.load(x_ptr + offs)
    if FP64:
        y = tl.sin(x)
    else:
        y = tl.sin(x.to(tl.float32))
    y = y.to(y_ptr.dtype.element_ty)
    if MASKED:
        tl.store(y_ptr + offs, y, mask=mask)
    else:
        tl.store(y_ptr + offs, y)


@triton.jit
def _sin_flat(
    x_ptr, y_ptr, n, FP64: tl.constexpr, MASKED: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    _sin_elem(x_ptr, y_ptr, offs, offs < n, FP64, MASKED)


@triton.jit
def _sin_fused(
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
    n,
    FP64: tl.constexpr,
    NM: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """All members share numel n; program_id(1) selects the member."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    m = tl.program_id(1)
    if NM > 1:
        if m == 0:
            _sin_elem(x0, y0, offs, mask, FP64, MASKED)
        elif m == 1:
            _sin_elem(x1, y1, offs, mask, FP64, MASKED)
        elif m == 2:
            _sin_elem(x2, y2, offs, mask, FP64, MASKED)
        elif m == 3:
            _sin_elem(x3, y3, offs, mask, FP64, MASKED)
        elif m == 4:
            _sin_elem(x4, y4, offs, mask, FP64, MASKED)
        elif m == 5:
            _sin_elem(x5, y5, offs, mask, FP64, MASKED)
        elif m == 6:
            _sin_elem(x6, y6, offs, mask, FP64, MASKED)
        elif m == 7:
            _sin_elem(x7, y7, offs, mask, FP64, MASKED)
        elif m == 8:
            _sin_elem(x8, y8, offs, mask, FP64, MASKED)
        elif m == 9:
            _sin_elem(x9, y9, offs, mask, FP64, MASKED)
        elif m == 10:
            _sin_elem(x10, y10, offs, mask, FP64, MASKED)
        elif m == 11:
            _sin_elem(x11, y11, offs, mask, FP64, MASKED)
        elif m == 12:
            _sin_elem(x12, y12, offs, mask, FP64, MASKED)
        elif m == 13:
            _sin_elem(x13, y13, offs, mask, FP64, MASKED)
        elif m == 14:
            _sin_elem(x14, y14, offs, mask, FP64, MASKED)
        else:
            _sin_elem(x15, y15, offs, mask, FP64, MASKED)
    else:
        _sin_elem(x0, y0, offs, mask, FP64, MASKED)


@triton.jit
def _sin_strided(
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
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    FP64: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = tl.where(mask, offs, 0).to(tl.int64)
    xo = tl.zeros([BLOCK], dtype=tl.int64)
    yo = tl.zeros([BLOCK], dtype=tl.int64)
    # mixed-radix decode, innermost dim first: dims are ordered slowest-first, so
    # the last entry carries the smallest input stride.
    if RANK > 7:
        q = rem % d7
        rem = rem // d7
        xo += q * i7
        yo += q * o7
    if RANK > 6:
        q = rem % d6
        rem = rem // d6
        xo += q * i6
        yo += q * o6
    if RANK > 5:
        q = rem % d5
        rem = rem // d5
        xo += q * i5
        yo += q * o5
    if RANK > 4:
        q = rem % d4
        rem = rem // d4
        xo += q * i4
        yo += q * o4
    if RANK > 3:
        q = rem % d3
        rem = rem // d3
        xo += q * i3
        yo += q * o3
    if RANK > 2:
        q = rem % d2
        rem = rem // d2
        xo += q * i2
        yo += q * o2
    if RANK > 1:
        q = rem % d1
        rem = rem // d1
        xo += q * i1
        yo += q * o1
    if RANK > 0:
        q = rem % d0
        rem = rem // d0
        xo += q * i0
        yo += q * o0
    x = tl.load(x_ptr + xo, mask=mask, other=0)
    if FP64:
        y = tl.sin(x)
    else:
        y = tl.sin(x.to(tl.float32))
    tl.store(y_ptr + yo, y.to(y_ptr.dtype.element_ty), mask=mask)


def _plan(t, out):
    """Per-dim (shape, in_stride, out_stride) tuples ordered slowest-first (so
    the innermost / fastest-varying dim carries the smallest input stride and
    loads stay coalesced) and collapsed when adjacent dims are unit-compatible
    for both the input and the output."""
    shape = t.shape
    ist = t.stride()
    ost = out.stride()
    dims = []
    for k in range(t.dim()):
        s = shape[k]
        if s == 1:
            continue
        dims.append((s, ist[k], ost[k]))
    dims.sort(key=lambda r: -r[1])
    merged = []
    for d in dims:
        if merged:
            p = merged[-1]
            if p[1] == d[0] * d[1] and p[2] == d[0] * d[2]:
                merged[-1] = (p[0] * d[0], d[1], d[2])
                continue
        merged.append(d)
    return merged


# Compiled kernels keyed by launch signature.  Re-launching a cached
# CompiledKernel skips the JIT signature binding and key construction that
# otherwise dominate a 5-6 us single-member call.
_LAUNCH_CACHE = {}
_NOOP_HOOK = lambda *args, **kwargs: None


def _flat_geometry(n):
    """(BLOCK, num_warps) for a flat run of n elements.  Large runs use a wide
    tile with 8 warps: this transcendental is a long dependent chain per
    element, so independent elements per thread and resident warps per SM
    matter more than the extra masked lanes cost."""
    if n >= 65536:
        return 8192, 8
    return 1024, 4


def _launch_strided(xarg, out, n, plan, fp64):
    d = [1] * 8
    i = [0] * 8
    o = [0] * 8
    for k, (s, si, so) in enumerate(plan):
        d[k] = s
        i[k] = si
        o[k] = so
    block, warps = _flat_geometry(n)
    _sin_strided[((n + block - 1) // block,)](
        xarg,
        out,
        n,
        *d,
        *i,
        *o,
        FP64=fp64,
        RANK=len(plan),
        BLOCK=block,
        num_warps=warps,
    )


def run(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "foreach_sin(): argument 'self' must be a sequence of Tensors, "
            "not " + type(self).__name__
        )
    if not self:
        raise RuntimeError("Tensor list must have at least one tensor.")
    count = len(self)
    infos = []
    infos_append = infos.append
    key = None
    uniform = True
    for t in self:
        try:
            dt = t.dtype
        except AttributeError:
            raise TypeError(
                "expected Tensor as element in argument 0, but got " + type(t).__name__
            ) from None
        if dt.is_floating_point:
            odt = dt
            is_fp = True
        else:
            odt = torch.float32
            is_fp = False
        n = t.numel()
        xarg = t if is_fp else t.view(_VIEW_DTYPE[dt])
        pre = None
        # contiguous members need no stride plan at all
        if t.is_contiguous():
            plan = None
        else:
            # ``empty_like`` reproduces PyTorch's foreach output layout: a
            # dense transposed member keeps its strides, a gappy member gets a
            # dense output.  The stride plan below maps both cases.
            pre = torch.empty_like(t, dtype=odt)
            plan = _plan(t, pre)
            if len(plan) == 0 or (
                len(plan) == 1 and plan[0][1] == 1 and plan[0][2] == 1
            ):
                plan = None
            else:
                uniform = False
        if uniform:
            if n == 0:
                uniform = False
            elif key is None:
                key = (dt, odt, n, t.device, t.shape)
            elif (
                key[0] is not dt
                or key[1] is not odt
                or key[2] != n
                or key[3] != t.device
                or key[4] != t.shape
            ):
                uniform = False
        infos_append((t, xarg, odt, n, plan, pre))
    if uniform and count > 1:
        # One bulk allocation plus unbind gives every member its own disjoint
        # output view, which removes count-1 allocator round trips per call.
        first = infos[0]
        outs = list(
            torch.empty(
                (count,) + tuple(first[0].shape),
                dtype=first[2],
                device=first[0].device,
            ).unbind(0)
        )
        n = first[3]
        block, warps = _flat_geometry(n)
        blocks = (n + block - 1) // block
        masked = (n % block) != 0
        fp64 = first[2] is torch.float64
        stream = torch.cuda.current_stream().cuda_stream
        x0, y0 = first[1], outs[0]
        for start in range(0, count, _MAX_FUSED):
            stop = start + _MAX_FUSED
            if stop > count:
                stop = count
            members = stop - start
            args = []
            for i in range(start, stop):
                args.append(infos[i][1])
                args.append(outs[i])
            for _ in range(_MAX_FUSED - members):
                args.append(x0)
                args.append(y0)
            lkey = (1, first[0].dtype, first[2], n, block, warps, masked, members)
            ck = _LAUNCH_CACHE.get(lkey)
            if ck is None:
                ck = _sin_fused.run(
                    *args,
                    n,
                    FP64=fp64,
                    NM=members,
                    MASKED=masked,
                    BLOCK=block,
                    num_warps=warps,
                    grid=(blocks, members),
                    warmup=True,
                )
                _LAUNCH_CACHE[lkey] = ck
            if ck is None:
                _sin_fused[(blocks, members)](
                    *args,
                    n,
                    FP64=fp64,
                    NM=members,
                    MASKED=masked,
                    BLOCK=block,
                    num_warps=warps,
                )
            else:
                ck.run(
                    blocks,
                    members,
                    1,
                    stream,
                    ck.function,
                    ck.packed_metadata,
                    None,
                    _NOOP_HOOK,
                    _NOOP_HOOK,
                    *args,
                    n,
                )
            _STATS.executor_launches += 1
        _STATS.groups += 1
        return outs
    outs = []
    outs_append = outs.append
    stream = None
    for t, xarg, odt, n, plan, pre in infos:
        out = pre if pre is not None else torch.empty_like(t, dtype=odt)
        outs_append(out)
        if n == 0:
            continue
        if plan is None:
            block, warps = _flat_geometry(n)
            blocks = (n + block - 1) // block
            masked = (n % block) != 0
            fp64 = odt is torch.float64
            if stream is None:
                stream = torch.cuda.current_stream().cuda_stream
            lkey = (0, xarg.dtype, odt, n, block, warps, masked)
            ck = _LAUNCH_CACHE.get(lkey)
            if ck is None:
                ck = _sin_flat.run(
                    xarg,
                    out,
                    n,
                    FP64=fp64,
                    MASKED=masked,
                    BLOCK=block,
                    num_warps=warps,
                    grid=(blocks,),
                    warmup=True,
                )
                _LAUNCH_CACHE[lkey] = ck
            if ck is None:
                _sin_flat[(blocks,)](
                    xarg,
                    out,
                    n,
                    FP64=fp64,
                    MASKED=masked,
                    BLOCK=block,
                    num_warps=warps,
                )
            else:
                ck.run(
                    blocks,
                    1,
                    1,
                    stream,
                    ck.function,
                    ck.packed_metadata,
                    None,
                    _NOOP_HOOK,
                    _NOOP_HOOK,
                    xarg,
                    out,
                    n,
                )
        else:
            _launch_strided(xarg, out, n, plan, odt is torch.float64)
        _STATS.executor_launches += 1
    _STATS.groups += 1
    return outs


def _foreach_sin(self):
    """Functional ``aten::_foreach_sin``.

    The fused kernel covers the benchmarked shape: one non-complex dtype, at
    most ``_MAX_FUSED`` members.  Everything else -- complex (handled by the
    complex kernel in the shared executor), fp64, and longer lists whose launch
    count must stay group-bounded -- goes through the existing executor.
    """
    logger.debug("GEMS _FOREACH_SIN")
    tensors = check_tensor_list(self)
    if len(tensors) > _MAX_FUSED or any(t.dtype not in _FAST_DTYPES for t in tensors):
        return apply_unary("sin", tensors, inplace=False)
    _STATS.reset()
    return run(tensors)
