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

"""In-place foreach acos: ``self[i] = acos(self[i])`` for every tensor in the list.

Architecture: the measured per-call cost on this target is dominated by
host-side launch overhead (~14.5us per Python-side Triton launch against ~0.7us
for ATen's C++ launch), so flat-addressable tensors are grouped by
(dtype, numel) and served by one launch whose grid axis 0 selects the tensor and
axis 1 the chunk.  Gappy views, which cannot be addressed by a flat index, keep
their own rank-specialised strided launch.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_WARPS = 4
_CAP = 16
_FUSE_BYTES = 4 << 20

# A 16-bit element carries half the bytes, so at equal elements per thread the
# 16-bit kernel issues one 128-bit access where fp32 issues two and tops out near
# 1.1 TB/s against fp32's 1.5 TB/s.  Doubling the block for 16-bit dtypes
# restores the fp32 per-thread byte footprint (two 128-bit accesses).
_BLOCK_OF = {
    torch.float16: 2048,
    torch.bfloat16: 2048,
    torch.float32: 1024,
    torch.float64: 1024,
}

_ITEMSIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

_TL_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def _acos_inplace_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = libdevice.acos(x)
    else:
        y = libdevice.acos(x.to(tl.float32)).to(x.dtype)
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit(
    do_not_specialize=[
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13",
        "p14",
        "p15",
    ]
)
def _acos_fused_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    n,
    BLOCK: tl.constexpr,
    DT: tl.constexpr,
):
    """One launch for up to ``_CAP`` same-dtype, same-numel tensors.

    Grid axis 0 selects the tensor, so a partial group costs no extra programs;
    axis 1 is the chunk inside the shared element count.  Addresses arrive as
    int64 scalars rather than tensors: the per-argument tensor specialization and
    pointer extraction in the launcher cost ~1.4us each, which is the dominant
    term once the group is launch-bound, and this path only serves groups whose
    GPU work is shorter than a launch, so the lost alignment assumption is free.
    """
    t = tl.program_id(0)
    p = p0
    p = tl.where(t == 1, p1, p)
    p = tl.where(t == 2, p2, p)
    p = tl.where(t == 3, p3, p)
    p = tl.where(t == 4, p4, p)
    p = tl.where(t == 5, p5, p)
    p = tl.where(t == 6, p6, p)
    p = tl.where(t == 7, p7, p)
    p = tl.where(t == 8, p8, p)
    p = tl.where(t == 9, p9, p)
    p = tl.where(t == 10, p10, p)
    p = tl.where(t == 11, p11, p)
    p = tl.where(t == 12, p12, p)
    p = tl.where(t == 13, p13, p)
    p = tl.where(t == 14, p14, p)
    p = tl.where(t == 15, p15, p)

    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    ptr = p.to(tl.pointer_type(DT))
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    if DT == tl.float64:
        y = libdevice.acos(x)
    else:
        y = libdevice.acos(x.to(tl.float32)).to(DT)
    tl.store(ptr + offs, y, mask=mask)


@triton.jit
def _acos_inplace_strided_kernel(
    x_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = idx < n
    rem = idx
    off = tl.zeros_like(idx)
    if RANK >= 1:
        off += (rem % d0) * s0
        rem = rem // d0
    if RANK >= 2:
        off += (rem % d1) * s1
        rem = rem // d1
    if RANK >= 3:
        off += (rem % d2) * s2
        rem = rem // d2
    if RANK >= 4:
        off += (rem % d3) * s3
        rem = rem // d3
    if RANK >= 5:
        off += (rem % d4) * s4
        rem = rem // d4
    x = tl.load(x_ptr + off, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = libdevice.acos(x)
    else:
        y = libdevice.acos(x.to(tl.float32)).to(x.dtype)
    tl.store(x_ptr + off, y, mask=mask)


def _flat_addressable(t):
    """True when [data_ptr, data_ptr + numel) covers exactly this tensor."""
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _layout(t):
    """Collapse a gappy view into (sizes, strides) with unit-named dims dropped."""
    shape = []
    strides = []
    for size, stride in zip(t.shape, t.stride()):
        if size == 1:
            continue
        if shape and stride * size == strides[-1]:
            # dims merge when the outer stride equals inner_size * inner_stride;
            # the merged dim then carries the *inner* stride.
            shape[-1] *= size
            strides[-1] = stride
        else:
            shape.append(size)
            strides.append(stride)
    return shape, strides


_DTYPE_NAMES = {
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.uint8: "Byte",
    torch.bool: "Bool",
}


def _check_dtype(dtype):
    if dtype.is_floating_point:
        return
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type "
            + str(dtype).replace("torch.", "").capitalize()
        )
    raise RuntimeError(
        "result type Float can't be cast to the desired output type " + name
    )


def _launch_uniform(tensors, count, dtype, n):
    """Launch for ``count`` flat-addressable tensors of one dtype and numel.

    Fusing trades ~23us of extra per-launch host cost (16 pointer arguments plus
    the address select) for 15 avoided launches, and it costs ~35% of element
    throughput on 16-bit dtypes.  That only pays while one tensor's slice of GPU
    work is shorter than a launch itself; above the crossover a per-tensor launch
    is both cheaper per element and hidden behind the GPU.
    """
    block = _BLOCK_OF[dtype]
    chunks = -(-n // block)
    if count == 1:
        _acos_inplace_kernel[(chunks,)](tensors[0], n, BLOCK=block, num_warps=_WARPS)
        return
    if n * _ITEMSIZE[dtype] > _FUSE_BYTES:
        for t in tensors:
            _acos_inplace_kernel[(chunks,)](t, n, BLOCK=block, num_warps=_WARPS)
        return
    dt = _TL_DT[dtype]
    i = 0
    while i < count:
        part = tensors[i : i + _CAP]
        k = len(part)
        ptrs = [x.data_ptr() for x in part]
        if k < _CAP:
            ptrs += [ptrs[0]] * (_CAP - k)
        _acos_fused_kernel[(k, chunks)](*ptrs, n, BLOCK=block, DT=dt, num_warps=_WARPS)
        i += _CAP


def _run_general(tensors):
    """Everything the uniform fast path cannot take: mixed dtypes, sizes or views."""
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        _check_dtype(t.dtype)

    groups = {}
    gappy = []
    for t in tensors:
        n = t.numel()
        if n == 0:
            continue
        if _flat_addressable(t):
            key = (t.dtype, n)
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = [t]
            else:
                bucket.append(t)
        else:
            gappy.append(t)

    for (dtype, n), members in groups.items():
        _launch_uniform(members, len(members), dtype, n)

    for t in gappy:
        n = t.numel()
        if torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        shape, strides = _layout(t)
        rank = len(shape)
        if rank > 5:
            raise RuntimeError("foreach acos_: view rank too large")
        shape = shape + [1] * (5 - rank)
        strides = strides + [0] * (5 - rank)
        block = _BLOCK_OF[t.dtype]
        _acos_inplace_strided_kernel[(-(-n // block),)](
            t,
            n,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            RANK=rank,
            BLOCK=block,
            num_warps=_WARPS,
        )


def foreach_acos_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(self)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Fast path: the benchmark and most call sites pass one dtype, one numel and
    # contiguous storage, so a single cheap scan replaces the per-tensor grouping
    # dict and the second validation pass.  Everything else (mixed dtypes, mixed
    # sizes, views, non-Tensor elements, rejected dtypes) falls through.
    first = self[0]
    if isinstance(first, torch.Tensor):
        dtype = first.dtype
        if dtype.is_floating_point and first.is_contiguous():
            n = first.numel()
            uniform = True
            for i in range(1, count):
                t = self[i]
                if (
                    type(t) is not torch.Tensor
                    or t.dtype is not dtype
                    or t.numel() != n
                    or not t.is_contiguous()
                ):
                    uniform = False
                    break
            if uniform:
                if n:
                    _launch_uniform(self, count, dtype, n)
                return None

    _run_general(self)
    return None
