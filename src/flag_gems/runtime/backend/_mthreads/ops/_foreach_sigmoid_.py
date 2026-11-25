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
"""In-place foreach sigmoid, Triton implementation.

Semantics reproduced from ``aten::_foreach_sigmoid_(Tensor(a!)[] self) -> ()``:

* every list element is mutated in place (storage, strides, dtype, shape kept),
* the call returns ``None``,
* integral / bool inputs are rejected exactly like ATen's in-place promotion
  refusal (``result type Float can't be cast to the desired output type ...``),
* tensors whose elements alias a single memory location are rejected,
* empty tensors are skipped entirely (no dummy launches).

Half precision inputs are upcast to fp32, evaluated and rounded once on store,
matching ATen's float opmath for the CUDA foreach kernels.

Launch strategy: ATen's CUDA foreach kernels process a whole TensorList with one
``multi_tensor_apply`` launch, so a per-tensor Python loop loses badly on short
lists.  Whenever the list is homogeneous (same dtype, same numel, all
contiguous, disjoint storages) *and* its tensors are small, all tensors are
handled by a single launch whose per-tensor base addresses are passed as
``int64`` kernel arguments.  That fused kernel only reaches ~0.45 TB/s, so
larger tensors keep the per-tensor launches that saturate the memory system.

The small-tensor rows are host-launch bound, so repeated launches replay a
cached ``CompiledKernel`` runner instead of going through ``JITFunction.run``
(binder, specialization refinement, MUSA pointer-alias spec, cache-key
computation, globals check).  The replay is only used for the exact
specialization that was compiled (same dtype, same numel, 16-byte aligned
pointers); anything else falls back to the ordinary JIT path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import _foreach_sigmoid_ as default__foreach_sigmoid_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_TL_ACC = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_TL_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_ISIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}

_BLOCK = 1024
_MAX_RANK = 8
_MULTI = 16
# Fusing a list into one launch removes ~20 us of enqueue cost per tensor, but
# the address-selecting kernel only sustains ~0.45 TB/s against the ~1.2 TB/s
# the plain per-tensor kernel reaches.  Fuse only while the saved launches
# dominate the per-byte penalty, i.e. for small tensors.
_FUSE_MAX_BYTES = 1 << 20

# (dtype, numel[, nt]) -> cached CompiledKernel runner for an already compiled
# specialization.  Keyed on the exact values Triton specializes on, so a
# replayed launch always matches the compiled assumptions.
_RUNNERS = {}


@triton.jit
def _sigmoid_flat_kernel(x_ptr, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    """Sigmoid over the flat storage span of one tensor (contiguous layouts)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(ACC)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sigmoid_multi_kernel(
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
    ACC: tl.constexpr,
    DT: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for up to ``_MULTI`` homogeneous contiguous tensors.

    Axis 0 selects the tensor, axis 1 the chunk inside it; every tensor of the
    group has the same element count, so a single ``n`` covers all of them.
    Base addresses arrive as ``int64`` scalars (no device metadata buffer, no
    host-to-device copy on the critical path).
    """
    t = tl.program_id(0)
    off = tl.program_id(1) * BLOCK
    p = p0
    if NT > 1:
        p = tl.where(t == 1, p1, p)
    if NT > 2:
        p = tl.where(t == 2, p2, p)
        p = tl.where(t == 3, p3, p)
    if NT > 4:
        p = tl.where(t == 4, p4, p)
        p = tl.where(t == 5, p5, p)
        p = tl.where(t == 6, p6, p)
        p = tl.where(t == 7, p7, p)
    if NT > 8:
        p = tl.where(t == 8, p8, p)
        p = tl.where(t == 9, p9, p)
        p = tl.where(t == 10, p10, p)
        p = tl.where(t == 11, p11, p)
        p = tl.where(t == 12, p12, p)
        p = tl.where(t == 13, p13, p)
        p = tl.where(t == 14, p14, p)
        p = tl.where(t == 15, p15, p)
    idx = off + tl.arange(0, BLOCK)
    mask = idx < n
    xp = p.to(tl.pointer_type(DT))
    x = tl.load(xp + idx, mask=mask, other=0.0).to(ACC)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(xp + idx, y.to(DT), mask=mask)


@triton.jit
def _sigmoid_strided_kernel(
    x_ptr,
    n,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    ACC: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sigmoid over a collapsed (shape, stride) descriptor, in place.

    The flat logical index is decoded dimension by dimension so that the real
    (possibly gappy) access pattern is preserved.  Nothing is staged through a
    contiguous copy: the mutation must land in the original storage.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    q = idx.to(tl.int64)
    o = (q % d0) * s0
    q = q // d0
    if RANK > 1:
        o += (q % d1) * s1
        q = q // d1
    if RANK > 2:
        o += (q % d2) * s2
        q = q // d2
    if RANK > 3:
        o += (q % d3) * s3
        q = q // d3
    if RANK > 4:
        o += (q % d4) * s4
        q = q // d4
    if RANK > 5:
        o += (q % d5) * s5
        q = q // d5
    if RANK > 6:
        o += (q % d6) * s6
        q = q // d6
    if RANK > 7:
        o += (q % d7) * s7
    x = tl.load(x_ptr + o, mask=mask, other=0.0).to(ACC)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(x_ptr + o, y.to(x_ptr.dtype.element_ty), mask=mask)


def _collapse(t):
    """Drop size-1 dims and merge adjacent dims that are already contiguous.

    Two neighbours (outer ``Sz, So`` and inner ``sz, si``) describe one flat run
    when ``So == si * sz``; the merged dimension then has stride ``si`` (the
    inner stride is the step between consecutive logical elements).
    """
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    if not dims:
        return [(1, 1)]
    merged = [list(dims[0])]
    for sz, st in dims[1:]:
        last = merged[-1]
        if last[1] == st * sz:
            last[0] *= sz
            last[1] = st
        else:
            merged.append([sz, st])
    return merged


def _launch_flat(t, n):
    dtype = t.dtype
    acc = _TL_ACC[dtype]
    g0 = (n + _BLOCK - 1) // _BLOCK
    if t.data_ptr() & 15:
        # Not 16-byte aligned: let Triton resolve (and cache) the right
        # specialization instead of replaying an aligned kernel.
        _sigmoid_flat_kernel[(g0,)](t, n, acc, _BLOCK, num_warps=4)
        return
    key = (dtype, n)
    runner = _RUNNERS.get(key)
    if runner is None:
        compiled = _sigmoid_flat_kernel.warmup(
            t, n, acc, _BLOCK, grid=(g0, 1, 1), num_warps=4
        )
        runner = _RUNNERS[key] = compiled[(g0, 1, 1)]
    runner(t, n, acc, _BLOCK)


def _launch_multi(ptrs, n, dtype):
    nt = len(ptrs)
    args = ptrs + [ptrs[0]] * (_MULTI - nt)
    acc = _TL_ACC[dtype]
    tldt = _TL_DT[dtype]
    g0 = (n + _BLOCK - 1) // _BLOCK
    key = (dtype, n, nt)
    runner = _RUNNERS.get(key)
    if runner is not None:
        runner(*args, n, acc, tldt, nt, _BLOCK)
        return
    for p in ptrs:
        if p & 15:
            _sigmoid_multi_kernel[(nt, g0)](
                *args, n, acc, tldt, nt, _BLOCK, num_warps=4
            )
            return
    compiled = _sigmoid_multi_kernel.warmup(
        *args, n, acc, tldt, nt, _BLOCK, grid=(nt, g0, 1), num_warps=4
    )
    runner = _RUNNERS[key] = compiled[(nt, g0, 1)]
    runner(*args, n, acc, tldt, nt, _BLOCK)


def _launch_strided(t, n):
    dims = _collapse(t)
    rank = len(dims)
    if rank > _MAX_RANK:
        raise NotImplementedError("foreach_sigmoid_: tensor rank too high")
    sizes = [d[0] for d in dims] + [1] * (_MAX_RANK - rank)
    strides = [d[1] for d in dims] + [0] * (_MAX_RANK - rank)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _sigmoid_strided_kernel[grid](
        t, n, *sizes, *strides, _TL_ACC[t.dtype], rank, _BLOCK, num_warps=4
    )


def _foreach_sigmoid_(self):  # noqa: A002 - mirrors the ATen argument name
    logger.debug("GEMS_MTHREADS _FOREACH_SIGMOID_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_sigmoid_(self)
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    count = len(tensors)
    if count == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    if count == 1:
        # A single-tensor call is pure per-call overhead; skip the list
        # bookkeeping the multi-tensor paths need.
        t = tensors[0]
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        if _TL_ACC.get(t.dtype) is None:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + str(t.dtype).replace("torch.", "").capitalize()
            )
        n = t.numel()
        if n == 0:
            return None
        if n == 1 or t.is_contiguous():
            _launch_flat(t, n)
        else:
            if torch._debug_has_internal_overlap(t) == 1:
                raise RuntimeError(
                    "unsupported operation: more than one element of the"
                    " written-to tensor refers to a single memory location."
                    " Please clone() the tensor before performing the"
                    " operation."
                )
            _launch_strided(t, n)
        return None

    numels = []
    contig = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if _TL_ACC.get(dtype) is None:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                + str(dtype).replace("torch.", "").capitalize()
            )
        c = t.is_contiguous()
        if not c and torch._debug_has_internal_overlap(t) == 1:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to"
                " tensor refers to a single memory location. Please clone() the"
                " tensor before performing the operation."
            )
        numels.append(t.numel())
        contig.append(c)

    dtype = tensors[0].dtype
    n = numels[0]
    if (
        n != 0
        and n * _ISIZE[dtype] <= _FUSE_MAX_BYTES
        and numels.count(n) == count
        and contig.count(True) == count
        and len({t.dtype for t in tensors}) == 1
    ):
        ptrs = [t.data_ptr() for t in tensors]
        if len(set(ptrs)) == count:
            for start in range(0, count, _MULTI):
                _launch_multi(ptrs[start : start + _MULTI], n, dtype)
            return None

    for i, t in enumerate(tensors):
        n = numels[i]
        if n == 0:
            continue
        if n == 1 or contig[i]:
            _launch_flat(t, n)
        else:
            _launch_strided(t, n)
    return None
