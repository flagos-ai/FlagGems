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

"""Triton implementation of aten::_foreach_div.ScalarList -> Tensor[]."""

import struct

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_WARPS = 4
_GROUP_MAX = 8
# Flat packing keeps every element offset inside int32, and large tensors are
# cheaper as one lean launch each anyway.
_GROUP_LIMIT = 1 << 30
_FAST_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _f64_bits(value):
    """Bit pattern of a python float, so an fp64 divisor reaches the kernel exactly."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


@triton.jit
def _div_body(in_ptr, out_ptr, elem_offs, s_f32):
    x = tl.load(in_ptr + elem_offs)
    y = x.to(tl.float32) / s_f32
    tl.store(out_ptr + elem_offs, y.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["s_bits"])
def _div_linear_kernel(
    in_ptr, out_ptr, s_f32, s_bits, IS_F64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    if IS_F64:
        s = tl.cast(s_bits, tl.float64, bitcast=True)
        y = x / s
    else:
        y = x.to(tl.float32) / s_f32
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n", "s_bits"])
def _div_linear_masked_kernel(
    in_ptr, out_ptr, n, s_f32, s_bits, IS_F64: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask, other=0)
    if IS_F64:
        s = tl.cast(s_bits, tl.float64, bitcast=True)
        y = x / s
    else:
        y = x.to(tl.float32) / s_f32
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["b0", "b1", "b2", "b3"])
def _div_group4_kernel(
    in0, in1, in2, in3, out_ptr, s0, s1, s2, s3, b0, b1, b2, b3, BLOCK: tl.constexpr
):
    """One launch for up to four same-dtype tensors.

    Outputs are packed head-to-tail in ``out_ptr``, so a program's store offset
    is just its global element offset; ``b0..b3`` are the per-tensor start block
    indices, and unused slots carry the total block count so their branches are
    unreachable.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid < b1:
        _div_body(in0, out_ptr, offs - b0 * BLOCK, s0)
    elif pid < b2:
        _div_body(in1, out_ptr, offs - b1 * BLOCK, s1)
    elif pid < b3:
        _div_body(in2, out_ptr, offs - b2 * BLOCK, s2)
    else:
        _div_body(in3, out_ptr, offs - b3 * BLOCK, s3)


@triton.jit(do_not_specialize=["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"])
def _div_group8_kernel(
    in0,
    in1,
    in2,
    in3,
    in4,
    in5,
    in6,
    in7,
    out_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    BLOCK: tl.constexpr,
):
    """Same packing scheme as ``_div_group4_kernel`` for five to eight tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid < b1:
        _div_body(in0, out_ptr, offs - b0 * BLOCK, s0)
    elif pid < b2:
        _div_body(in1, out_ptr, offs - b1 * BLOCK, s1)
    elif pid < b3:
        _div_body(in2, out_ptr, offs - b2 * BLOCK, s2)
    elif pid < b4:
        _div_body(in3, out_ptr, offs - b3 * BLOCK, s3)
    elif pid < b5:
        _div_body(in4, out_ptr, offs - b4 * BLOCK, s4)
    elif pid < b6:
        _div_body(in5, out_ptr, offs - b5 * BLOCK, s5)
    elif pid < b7:
        _div_body(in6, out_ptr, offs - b6 * BLOCK, s6)
    else:
        _div_body(in7, out_ptr, offs - b7 * BLOCK, s7)


@triton.jit
def _div_strided_kernel(
    in_ptr,
    out_ptr,
    n,
    s_f32,
    s_bits,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    k0,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    k7,
    RANK: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs.to(tl.int64)
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        off += (rem % d0) * k0
        rem = rem // d0
    if RANK > 1:
        off += (rem % d1) * k1
        rem = rem // d1
    if RANK > 2:
        off += (rem % d2) * k2
        rem = rem // d2
    if RANK > 3:
        off += (rem % d3) * k3
        rem = rem // d3
    if RANK > 4:
        off += (rem % d4) * k4
        rem = rem // d4
    if RANK > 5:
        off += (rem % d5) * k5
        rem = rem // d5
    if RANK > 6:
        off += (rem % d6) * k6
        rem = rem // d6
    if RANK > 7:
        off += (rem % d7) * k7
        rem = rem // d7
    x = tl.load(in_ptr + off, mask=mask, other=0)
    if IS_F64:
        s = tl.cast(s_bits, tl.float64, bitcast=True)
        y = x / s
    else:
        y = x.to(tl.float32) / s_f32
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_linear(src, dst, divisor):
    numel = src.numel()
    is_f64 = src.dtype == torch.float64
    if numel % _BLOCK == 0:
        grid = (numel // _BLOCK,)
        if is_f64:
            _div_linear_kernel[grid](
                src,
                dst,
                0.0,
                _f64_bits(divisor),
                IS_F64=True,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        else:
            _div_linear_kernel[grid](
                src,
                dst,
                float(divisor),
                0,
                IS_F64=False,
                BLOCK=_BLOCK,
                num_warps=_WARPS,
            )
        return
    grid = ((numel + _BLOCK - 1) // _BLOCK,)
    if is_f64:
        _div_linear_masked_kernel[grid](
            src,
            dst,
            numel,
            0.0,
            _f64_bits(divisor),
            IS_F64=True,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    else:
        _div_linear_masked_kernel[grid](
            src,
            dst,
            numel,
            float(divisor),
            0,
            IS_F64=False,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )


def _launch_single(item, outs):
    idx, view, numel, shape, inv, s = item
    out = torch.empty(view.shape, dtype=view.dtype, device=view.device)
    _launch_linear(view, out, s)
    outs[idx] = out if inv is None else out.permute(inv)


def _launch_group(items, outs):
    """items: list of (index, src_view, numel, out_shape, inv_perm, divisor)."""
    total = 0
    for it in items:
        total += it[2]
    flat = torch.empty(total, dtype=items[0][1].dtype, device=items[0][1].device)
    starts = []
    off = 0
    for it in items:
        starts.append(off // _BLOCK)
        off += it[2]
    total_blocks = off // _BLOCK
    scal = [float(it[5]) for it in items]
    ptrs = [it[1] for it in items]
    if len(items) <= 4:
        while len(ptrs) < 4:
            ptrs.append(ptrs[0])
            scal.append(scal[0])
        while len(starts) < 4:
            starts.append(total_blocks)
        _div_group4_kernel[(total_blocks,)](
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            flat,
            scal[0],
            scal[1],
            scal[2],
            scal[3],
            starts[0],
            starts[1],
            starts[2],
            starts[3],
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    else:
        while len(ptrs) < _GROUP_MAX:
            ptrs.append(ptrs[0])
            scal.append(scal[0])
        while len(starts) < _GROUP_MAX:
            starts.append(total_blocks)
        _div_group8_kernel[(total_blocks,)](
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            ptrs[4],
            ptrs[5],
            ptrs[6],
            ptrs[7],
            flat,
            scal[0],
            scal[1],
            scal[2],
            scal[3],
            scal[4],
            scal[5],
            scal[6],
            scal[7],
            starts[0],
            starts[1],
            starts[2],
            starts[3],
            starts[4],
            starts[5],
            starts[6],
            starts[7],
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    shape0 = items[0][3]
    uniform = True
    for it in items:
        if it[4] is not None or it[3] != shape0:
            uniform = False
            break
    if uniform:
        parts = flat.view(len(items), *shape0).unbind(0)
        for it, part in zip(items, parts):
            outs[it[0]] = part
    else:
        sizes = []
        for it in items:
            sizes.append(it[2])
        parts = flat.split(sizes)
        for it, part in zip(items, parts):
            out = part.view(it[3])
            if it[4] is not None:
                out = out.permute(it[4])
            outs[it[0]] = out


def _launch_strided(src, dst, divisor):
    numel = src.numel()
    dims = [1] * 8
    strides = [0] * 8
    for i in range(src.dim()):
        dims[i] = src.shape[i]
        strides[i] = src.stride(i)
    grid = ((numel + _BLOCK - 1) // _BLOCK,)
    if src.dtype == torch.float64:
        bits = _f64_bits(divisor)
        s32 = 0.0
    else:
        bits = 0
        s32 = float(divisor)
    _div_strided_kernel[grid](
        src,
        dst,
        numel,
        s32,
        bits,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        dims[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        RANK=src.dim(),
        IS_F64=(src.dtype == torch.float64),
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _permute_to_contig(t):
    strides = t.stride()
    perm = sorted(range(t.dim()), key=lambda d: -strides[d])
    view = t.permute(perm)
    if not view.is_contiguous():
        return None
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return view, inv


def _result_dtype(t):
    dt = t.dtype
    if dt.is_floating_point or dt.is_complex:
        return dt
    return torch.get_default_dtype()


def _is_scalar_like(value):
    if isinstance(value, (int, float, bool)):
        return True
    return torch.is_tensor(value) and value.dim() == 0


def foreach_div_scalar_list(self, scalars):
    tensors = self if isinstance(self, (list, tuple)) else [self]
    count = len(tensors)
    if count == 0:
        return []
    if _is_scalar_like(scalars):
        divisors = [scalars] * count
    else:
        divisors = list(scalars)
        if len(divisors) != count:
            raise RuntimeError("tensor list and scalar list lengths must match")

    # Fast path: a homogeneous list of contiguous tensors of one fast dtype is
    # served by a single packed launch.
    if 2 <= count <= _GROUP_MAX:
        first = tensors[0]
        dtype = first.dtype
        if dtype in _FAST_DTYPES and first.is_contiguous():
            device = first.device
            total = 0
            ok = True
            for t in tensors:
                numel = t.numel()
                if (
                    numel <= 0
                    or numel % _BLOCK
                    or t.dtype != dtype
                    or not t.is_contiguous()
                    or t.device != device
                ):
                    ok = False
                    break
                total += numel
            if ok and total <= _GROUP_LIMIT:
                outs = [None] * count
                items = []
                i = 0
                for t, s in zip(tensors, divisors):
                    items.append((i, t, t.numel(), t.shape, None, s))
                    i += 1
                _launch_group(items, outs)
                return outs

    outs = [None] * count
    groupable = []
    for i in range(count):
        t = tensors[i]
        s = divisors[i]
        numel = t.numel()
        if numel == 0:
            outs[i] = torch.empty(t.shape, dtype=_result_dtype(t), device=t.device)
            continue
        if t.is_contiguous():
            view, inv = t, None
        else:
            conv = _permute_to_contig(t)
            if conv is None:
                out = torch.empty(t.shape, dtype=_result_dtype(t), device=t.device)
                _launch_strided(t, out, s)
                outs[i] = out
                continue
            view, inv = conv
        if numel % _BLOCK == 0 and view.dtype in _FAST_DTYPES:
            groupable.append((i, view, numel, view.shape, inv, s))
            continue
        out = torch.empty(view.shape, dtype=_result_dtype(t), device=t.device)
        _launch_linear(view, out, s)
        outs[i] = out if inv is None else out.permute(inv)

    if groupable:
        buckets = {}
        order = []
        for it in groupable:
            key = (it[1].dtype, it[1].device)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = []
                order.append(key)
            bucket.append(it)
        for key in order:
            chunk = []
            chunk_numel = 0
            for it in bucket:
                if chunk and (
                    len(chunk) >= _GROUP_MAX or chunk_numel + it[2] > _GROUP_LIMIT
                ):
                    if len(chunk) == 1:
                        _launch_single(chunk[0], outs)
                    else:
                        _launch_group(chunk, outs)
                    chunk = []
                    chunk_numel = 0
                chunk.append(it)
                chunk_numel += it[2]
            if chunk:
                if len(chunk) == 1:
                    _launch_single(chunk[0], outs)
                else:
                    _launch_group(chunk, outs)
    return outs
