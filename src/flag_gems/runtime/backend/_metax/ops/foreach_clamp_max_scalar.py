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

"""Triton implementation of aten::_foreach_clamp_max.Scalar.

out_i = min(self_i, scalar) elementwise, for every tensor in the list, with NaN
propagating from either operand (ATen semantics on this target). Out-of-place:
freshly allocated outputs that never alias the inputs.

Contiguous tensors that share a dtype are dispatched with ONE grouped kernel
launch per bucket: each CTA maps its program id to a tensor through cumulative
tile counts derived in-kernel, so the per-launch cost is paid once per bucket
instead of once per tensor. The data is streamed exactly once, so the loads and
stores carry evict-first hints to keep the streaming working set out of the way.
Non-contiguous or oversized tensors keep the per-tensor strided path.
"""

import struct

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8
_GROUP = 4
_MAX_NUMEL = 2**31 - 8192
_MISSING = object()


@triton.jit
def _contig_kernel(
    x_ptr,
    o_ptr,
    n,
    bound,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BOUND64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")
    if BOUND64:
        b = tl.cast(bound, tl.float64, bitcast=True)
    else:
        b = bound
    if IS_FLOAT:
        y = tl.minimum(x, b, propagate_nan=tl.PropagateNan.ALL)
    else:
        y = tl.minimum(x, b)
    tl.store(o_ptr + offs, y, mask=mask, eviction_policy="evict_first")


@triton.jit
def _contig_group_full_kernel(
    x0,
    o0,
    x1,
    o1,
    x2,
    o2,
    x3,
    o3,
    tiles,
    bound,
    BLOCK: tl.constexpr,
):
    """Four equal-length contiguous float tensors, each an exact multiple of BLOCK.

    Same one-tensor-per-CTA mapping as the masked group kernel, but with no tail
    compare, no masked-fill path and only ten runtime arguments
    (tiles = numel // BLOCK, grid = 4 * tiles).
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid < tiles:
        idx = pid * BLOCK + offs
        x = tl.load(x0 + idx, eviction_policy="evict_first")
        x = tl.minimum(x, bound, propagate_nan=tl.PropagateNan.ALL)
        tl.store(o0 + idx, x, eviction_policy="evict_first")
    elif pid < 2 * tiles:
        idx = (pid - tiles) * BLOCK + offs
        x = tl.load(x1 + idx, eviction_policy="evict_first")
        x = tl.minimum(x, bound, propagate_nan=tl.PropagateNan.ALL)
        tl.store(o1 + idx, x, eviction_policy="evict_first")
    elif pid < 3 * tiles:
        idx = (pid - 2 * tiles) * BLOCK + offs
        x = tl.load(x2 + idx, eviction_policy="evict_first")
        x = tl.minimum(x, bound, propagate_nan=tl.PropagateNan.ALL)
        tl.store(o2 + idx, x, eviction_policy="evict_first")
    else:
        idx = (pid - 3 * tiles) * BLOCK + offs
        x = tl.load(x3 + idx, eviction_policy="evict_first")
        x = tl.minimum(x, bound, propagate_nan=tl.PropagateNan.ALL)
        tl.store(o3 + idx, x, eviction_policy="evict_first")


@triton.jit
def _contig_group_kernel(
    x0,
    o0,
    n0,
    x1,
    o1,
    n1,
    x2,
    o2,
    n2,
    x3,
    o3,
    n3,
    bound,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BOUND64: tl.constexpr,
):
    """One launch for up to four contiguous tensors of the same dtype.

    Grid = sum_i cdiv(n_i, BLOCK); the cumulative tile counts are derived from the
    element counts inside the kernel so the launch needs no extra scalar state. An
    empty slot (n_i == 0) makes its cumulative count equal to the previous one, so
    its branch is unreachable.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    c1 = tl.cdiv(n0, BLOCK)
    c2 = c1 + tl.cdiv(n1, BLOCK)
    c3 = c2 + tl.cdiv(n2, BLOCK)
    if BOUND64:
        b = tl.cast(bound, tl.float64, bitcast=True)
    else:
        b = bound
    if pid < c1:
        idx = pid * BLOCK + offs
        mask = idx < n0
        x = tl.load(x0 + idx, mask=mask, other=0, eviction_policy="evict_first")
        if IS_FLOAT:
            x = tl.minimum(x, b, propagate_nan=tl.PropagateNan.ALL)
        else:
            x = tl.minimum(x, b)
        tl.store(o0 + idx, x, mask=mask, eviction_policy="evict_first")
    elif pid < c2:
        idx = (pid - c1) * BLOCK + offs
        mask = idx < n1
        x = tl.load(x1 + idx, mask=mask, other=0, eviction_policy="evict_first")
        if IS_FLOAT:
            x = tl.minimum(x, b, propagate_nan=tl.PropagateNan.ALL)
        else:
            x = tl.minimum(x, b)
        tl.store(o1 + idx, x, mask=mask, eviction_policy="evict_first")
    elif pid < c3:
        idx = (pid - c2) * BLOCK + offs
        mask = idx < n2
        x = tl.load(x2 + idx, mask=mask, other=0, eviction_policy="evict_first")
        if IS_FLOAT:
            x = tl.minimum(x, b, propagate_nan=tl.PropagateNan.ALL)
        else:
            x = tl.minimum(x, b)
        tl.store(o2 + idx, x, mask=mask, eviction_policy="evict_first")
    else:
        idx = (pid - c3) * BLOCK + offs
        mask = idx < n3
        x = tl.load(x3 + idx, mask=mask, other=0, eviction_policy="evict_first")
        if IS_FLOAT:
            x = tl.minimum(x, b, propagate_nan=tl.PropagateNan.ALL)
        else:
            x = tl.minimum(x, b)
        tl.store(o3 + idx, x, mask=mask, eviction_policy="evict_first")


@triton.jit
def _strided_kernel(
    x_ptr,
    o_ptr,
    n,
    bound,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BOUND64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    rem = offs
    xoff = tl.zeros((BLOCK,), dtype=tl.int64)
    ooff = tl.zeros((BLOCK,), dtype=tl.int64)
    if RANK >= 1:
        c = rem % s0
        rem = rem // s0
        xoff += c * x0
        ooff += c * y0
    if RANK >= 2:
        c = rem % s1
        rem = rem // s1
        xoff += c * x1
        ooff += c * y1
    if RANK >= 3:
        c = rem % s2
        rem = rem // s2
        xoff += c * x2
        ooff += c * y2
    if RANK >= 4:
        c = rem % s3
        rem = rem // s3
        xoff += c * x3
        ooff += c * y3
    if RANK >= 5:
        c = rem % s4
        rem = rem // s4
        xoff += c * x4
        ooff += c * y4
    if RANK >= 6:
        c = rem % s5
        rem = rem // s5
        xoff += c * x5
        ooff += c * y5
    if RANK >= 7:
        c = rem % s6
        rem = rem // s6
        xoff += c * x6
        ooff += c * y6
    if RANK >= 8:
        c = rem % s7
        rem = rem // s7
        xoff += c * x7
        ooff += c * y7
    x = tl.load(x_ptr + xoff, mask=mask, other=0)
    if BOUND64:
        b = tl.cast(bound, tl.float64, bitcast=True)
    else:
        b = bound
    if IS_FLOAT:
        y = tl.minimum(x, b, propagate_nan=tl.PropagateNan.ALL)
    else:
        y = tl.minimum(x, b)
    tl.store(o_ptr + ooff, y, mask=mask)


def _parse_args(args, kwargs):
    """Accept list/tuple of tensors plus one scalar in either order or by keyword."""
    if not kwargs and len(args) == 2 and isinstance(args[0], (list, tuple)):
        return args[0], args[1]
    tensors = None
    bound = _MISSING
    rest = list(args)
    if rest and isinstance(rest[0], (list, tuple)):
        tensors = rest.pop(0)
        if rest:
            bound = rest.pop(0)
    elif len(rest) >= 2 and isinstance(rest[1], (list, tuple)):
        bound = rest.pop(0)
        tensors = rest.pop(0)
    kw = dict(kwargs)
    if tensors is None:
        for key in ("self", "tensors", "input", "inputs", "tensor_list", "seq"):
            if key in kw:
                tensors = kw.pop(key)
                break
    if bound is _MISSING:
        for key in ("scalar", "max", "bound", "value", "other", "clamp_max"):
            if key in kw:
                bound = kw.pop(key)
                break
    if bound is _MISSING and rest:
        bound = rest.pop(0)
    if tensors is None or bound is _MISSING:
        raise TypeError("_foreach_clamp_max.Scalar expects (Sequence[Tensor], Scalar)")
    return tensors, bound


def _scalar_value(bound):
    kind = type(bound)
    if kind is float or kind is int:
        return bound
    if torch.is_tensor(bound):
        if bound.numel() != 1:
            raise TypeError("clamp_max bound must be a scalar")
        return bound.item()
    if isinstance(bound, bool):
        return int(bound)
    if isinstance(bound, int):
        return bound
    return float(bound)


def _plan(t, bound_value):
    """Return (out, bound_arg, is_float, bound64) for one tensor."""
    is_float = t.is_floating_point()
    out_dtype = t.dtype
    bound_arg = bound_value
    bound64 = False
    if t.dtype == torch.float64:
        bound_arg = struct.unpack("<q", struct.pack("<d", float(bound_value)))[0]
        bound64 = True
    elif is_float:
        bound_arg = float(bound_value)
    else:
        as_float = float(bound_value)
        if as_float != int(as_float):
            # ATen promotes an integral tensor against a floating bound to fp32.
            out_dtype = torch.float32
            bound_arg = as_float
        elif t.dtype == torch.bool:
            out_dtype = torch.int64
            bound_arg = int(bound_value)
        else:
            bound_arg = int(bound_value)
    if out_dtype == t.dtype:
        out = torch.empty_like(t)
    else:
        out = torch.empty_like(t, dtype=out_dtype)
    return out, bound_arg, is_float or out_dtype == torch.float32, bound64


def _fast_path(tensors, bound_value):
    """Homogeneous list of 1-4 contiguous 2/4-byte float tensors: one launch.

    Skips per-tensor plan construction, dtype promotion and output-stride
    analysis, which is pure host cost on small lists. Returns None whenever the
    list does not match, leaving the general path to handle it.
    """
    count = len(tensors)
    if count == 0 or count > _GROUP:
        return None
    t0 = tensors[0]
    dtype = t0.dtype
    if (
        dtype is not torch.float16
        and dtype is not torch.bfloat16
        and dtype is not torch.float32
    ):
        return None
    if not t0.is_contiguous():
        return None
    numel = t0.numel()
    if numel == 0 or numel >= _MAX_NUMEL:
        return None
    outs = [torch.empty_like(t0)]
    inputs = [t0]
    for index in range(1, count):
        t = tensors[index]
        if t.dtype is not dtype or t.numel() != numel or not t.is_contiguous():
            return None
        inputs.append(t)
        outs.append(torch.empty_like(t))
    bound = float(bound_value)
    tiles = -(-numel // _BLOCK)
    if count == 1:
        _contig_kernel[(tiles,)](
            t0,
            outs[0],
            numel,
            bound,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
            IS_FLOAT=True,
            BOUND64=False,
        )
        return outs
    last_in = inputs[-1]
    last_out = outs[-1]
    if count == _GROUP and numel % _BLOCK == 0:
        _contig_group_full_kernel[(tiles * _GROUP,)](
            inputs[0],
            outs[0],
            inputs[1],
            outs[1],
            inputs[2],
            outs[2],
            inputs[3],
            outs[3],
            tiles,
            bound,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        return outs
    pad = _GROUP - count
    px = inputs + [last_in] * pad
    po = outs + [last_out] * pad
    pn = [numel] * count + [0] * pad
    _contig_group_kernel[(tiles * count,)](
        px[0],
        po[0],
        pn[0],
        px[1],
        po[1],
        pn[1],
        px[2],
        po[2],
        pn[2],
        px[3],
        po[3],
        pn[3],
        bound,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        IS_FLOAT=True,
        BOUND64=False,
    )
    return outs


def _launch_contig(t, out, numel, bound_arg, is_float, bound64):
    grid = (triton.cdiv(numel, _BLOCK),)
    _contig_kernel[grid](
        t,
        out,
        numel,
        bound_arg,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        IS_FLOAT=is_float,
        BOUND64=bound64,
    )


def _launch_group(chunk, bound_arg, is_float, bound64):
    last = chunk[-1]
    slots = list(chunk)
    while len(slots) < _GROUP:
        slots.append((last[0], last[1], 0))
    (xa, oa, na), (xb, ob, nb), (xc, oc, nc), (xd, od, nd) = slots
    grid = (
        triton.cdiv(na, _BLOCK)
        + triton.cdiv(nb, _BLOCK)
        + triton.cdiv(nc, _BLOCK)
        + triton.cdiv(nd, _BLOCK),
    )
    _contig_group_kernel[grid](
        xa,
        oa,
        na,
        xb,
        ob,
        nb,
        xc,
        oc,
        nc,
        xd,
        od,
        nd,
        bound_arg,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        IS_FLOAT=is_float,
        BOUND64=bound64,
    )


def _launch_strided(t, out, numel, bound_arg, is_float, bound64):
    dim = t.dim()
    if dim > _MAX_RANK:
        raise NotImplementedError("foreach_clamp_max supports tensors up to rank 8")
    pad = _MAX_RANK - dim
    shp = list(t.shape) + [1] * pad
    xst = list(t.stride()) + [0] * pad
    ost = list(out.stride()) + [0] * pad
    grid = (triton.cdiv(numel, _BLOCK),)
    _strided_kernel[grid](
        t,
        out,
        numel,
        bound_arg,
        *shp,
        *xst,
        *ost,
        RANK=max(dim, 1),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        IS_FLOAT=is_float,
        BOUND64=bound64,
    )


def foreach_clamp_max_scalar(*args, **kwargs):
    tensors, bound = _parse_args(args, kwargs)
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")
    bound_value = _scalar_value(bound)
    fast = _fast_path(tensors, bound_value)
    if fast is not None:
        return fast
    tensors = list(tensors)
    outs = []
    groups = {}
    singles = []
    for t in tensors:
        out, bound_arg, is_float, bound64 = _plan(t, bound_value)
        outs.append(out)
        numel = t.numel()
        if numel == 0:
            continue
        if numel < _MAX_NUMEL and t.is_contiguous() and out.is_contiguous():
            key = (t.dtype, is_float, bound64, bound_arg)
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = [(t, out, numel)]
            else:
                bucket.append((t, out, numel))
        else:
            singles.append((t, out, numel, bound_arg, is_float, bound64))
    for (_, is_float, bound64, bound_arg), items in groups.items():
        count = len(items)
        if count == 1:
            t, out, numel = items[0]
            _launch_contig(t, out, numel, bound_arg, is_float, bound64)
            continue
        for start in range(0, count, _GROUP):
            chunk = items[start : start + _GROUP]
            if len(chunk) == 1:
                t, out, numel = chunk[0]
                _launch_contig(t, out, numel, bound_arg, is_float, bound64)
            else:
                _launch_group(chunk, bound_arg, is_float, bound64)
    for t, out, numel, bound_arg, is_float, bound64 in singles:
        if numel < _MAX_NUMEL and t.is_contiguous() and out.is_contiguous():
            _launch_contig(t, out, numel, bound_arg, is_float, bound64)
        else:
            _launch_strided(t, out, numel, bound_arg, is_float, bound64)
    return outs
