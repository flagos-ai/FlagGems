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

"""In-place foreach add of a scalar to a list of tensors.

Semantics: for every tensor T in the list, T[i] <- cast(T[i] + scalar) in fp32
opmath, written back into T's existing storage.  ``run`` returns None.

Layout: tensors are grouped by (device, dtype).  Each group of contiguous
tensors is executed with ONE fused kernel that indexes the concatenation of the
padded per-tensor element ranges and resolves the owning tensor from a scalar
where-chain, so a list of N tensors costs ceil(N/4) launches instead of N.
Non-contiguous tensors keep their physical strides and are handled by a
per-tensor stride-decomposition kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_add__Scalar as default_foreach_add__Scalar,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK_32BIT = 1024
_BLOCK_16BIT = 2048
_BLOCK_STRIDED = 512
_MAX_DIMS = 8
_FUSE_SLOTS = 4


def _block_for(itemsize):
    """Elements per program chosen so each thread moves ~32 bytes per program."""
    return _BLOCK_16BIT if itemsize <= 2 else _BLOCK_32BIT


@triton.jit(do_not_specialize=["n", "sc"])
def _add_scalar_flat_kernel(ptr, n, sc, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(ptr + offs)
        tl.store(ptr + offs, x + sc)
    else:
        mask = offs < n
        x = tl.load(ptr + offs, mask=mask, other=0)
        tl.store(ptr + offs, x + sc, mask=mask)


@triton.jit(do_not_specialize=["sc", "n0", "n1", "n2", "n3", "q1", "q2", "q3"])
def _fused_add_scalar_kernel(
    p0,
    p1,
    p2,
    p3,
    sc,
    n0,
    n1,
    n2,
    n3,
    q1,
    q2,
    q3,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """One launch for up to four contiguous same-dtype tensors.

    Program ids are laid out as padded block ranges; q1..q3 are the cumulative
    block counts in program units.  Pointers are real kernel arguments (so their
    alignment is known) and the per-program offset is the runtime block index
    multiplied by the constant BLOCK, which keeps the access pattern vectorizable.
    """
    pid = tl.program_id(0)
    p = p0
    sel = pid
    nn = n0.to(tl.int32)
    if pid >= q1:
        p = p1
        sel = pid - q1
        nn = n1.to(tl.int32)
    if pid >= q2:
        p = p2
        sel = pid - q2
        nn = n2.to(tl.int32)
    if pid >= q3:
        p = p3
        sel = pid - q3
        nn = n3.to(tl.int32)
    offs = tl.max_contiguous(
        tl.multiple_of(sel * BLOCK + tl.arange(0, BLOCK), BLOCK), BLOCK
    )
    if EVEN:
        x = tl.load(p + offs)
        tl.store(p + offs, x + sc)
    else:
        mask = offs < nn
        x = tl.load(p + offs, mask=mask, other=0)
        tl.store(p + offs, x + sc, mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "sc",
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
    ]
)
def _add_scalar_strided_kernel(
    ptr,
    n,
    sc,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    BLOCK: tl.constexpr,
    NDIM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = offs
    addr = tl.zeros([BLOCK], dtype=tl.int64)
    if NDIM > 0:
        i0 = rem % d0
        rem = rem // d0
        addr += i0 * t0
    if NDIM > 1:
        i1 = rem % d1
        rem = rem // d1
        addr += i1 * t1
    if NDIM > 2:
        i2 = rem % d2
        rem = rem // d2
        addr += i2 * t2
    if NDIM > 3:
        i3 = rem % d3
        rem = rem // d3
        addr += i3 * t3
    if NDIM > 4:
        i4 = rem % d4
        rem = rem // d4
        addr += i4 * t4
    if NDIM > 5:
        i5 = rem % d5
        rem = rem // d5
        addr += i5 * t5
    if NDIM > 6:
        i6 = rem % d6
        rem = rem // d6
        addr += i6 * t6
    if NDIM > 7:
        i7 = rem % d7
        rem = rem // d7
        addr += i7 * t7
    x = tl.load(ptr + addr, mask=mask, other=0)
    tl.store(ptr + addr, x + sc, mask=mask)


def _collapse_dims(shape, strides):
    """Return (size, stride) pairs in fastest-varying-first order.

    Size-1 dimensions are dropped and runs of contiguous dimensions are merged,
    which keeps the number of address-decomposition steps small.
    """
    dims = []
    for i in range(len(shape) - 1, -1, -1):
        sz = int(shape[i])
        st = int(strides[i])
        if sz == 1:
            continue
        if dims:
            psz, pst = dims[-1]
            if pst * psz == st:
                dims[-1] = (psz * sz, pst)
                continue
        dims.append((sz, st))
    return dims


def _check_no_overlap(shape, strides):
    for sz, st in zip(shape, strides):
        if int(sz) > 1 and int(st) == 0:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to "
                "tensor refers to a single memory location. Please clone() the "
                "tensor before performing the operation."
            )


def _scalar_value(t, scalar):
    if t.dtype.is_floating_point:
        return float(scalar)
    if isinstance(scalar, float):
        raise RuntimeError("result type Float can't be cast to the desired output type")
    return int(scalar)


def _launch_flat(t, numel, sval):
    blk = _block_for(t.element_size())
    grid = ((numel + blk - 1) // blk,)
    even = (numel % blk) == 0
    _add_scalar_flat_kernel[grid](t, numel, sval, BLOCK=blk, EVEN=even, num_warps=4)


def _launch_fused(chunk, numels, sval):
    nt = len(chunk)
    ptrs = [t.data_ptr() for t in chunk]
    if len(set(ptrs)) != nt or max(numels) >= 2147483648:
        return False
    isz = chunk[0].element_size()
    blk = _block_for(isz)
    ppts = [((numels[k] + blk - 1) // blk) for k in range(nt)]
    grid_n = sum(ppts)
    qs = []
    acc = 0
    even = True
    for k in range(3):
        if k < nt - 1:
            acc += ppts[k]
            qs.append(acc)
        else:
            qs.append(grid_n)
    for k in range(nt):
        if numels[k] % blk:
            even = False
    ps = [chunk[k] if k < nt else chunk[0] for k in range(_FUSE_SLOTS)]
    ns = [numels[k] if k < nt else 0 for k in range(_FUSE_SLOTS)]
    _fused_add_scalar_kernel[(grid_n,)](
        ps[0],
        ps[1],
        ps[2],
        ps[3],
        sval,
        ns[0],
        ns[1],
        ns[2],
        ns[3],
        qs[0],
        qs[1],
        qs[2],
        BLOCK=blk,
        EVEN=even,
        num_warps=4,
    )
    return True


def _launch_strided(t, numel, sval, dims):
    ds = [1] * _MAX_DIMS
    ts = [0] * _MAX_DIMS
    for i, (sz, st) in enumerate(dims):
        ds[i] = sz
        ts[i] = st
    grid = ((numel + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)
    _add_scalar_strided_kernel[grid](
        t,
        numel,
        sval,
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        ts[0],
        ts[1],
        ts[2],
        ts[3],
        ts[4],
        ts[5],
        ts[6],
        ts[7],
        BLOCK=_BLOCK_STRIDED,
        NDIM=len(dims),
        num_warps=4,
    )


def _add_strided_one(t, scalar):
    numel = t.numel()
    if numel == 0:
        return
    if t.is_complex():
        raise RuntimeError("foreach_add_: complex tensors are not supported")
    sval = _scalar_value(t, scalar)
    shape = t.shape
    strides = t.stride()
    _check_no_overlap(shape, strides)
    dims = _collapse_dims(shape, strides)
    if len(dims) <= _MAX_DIMS:
        _launch_strided(t, numel, sval, dims)
        return
    # Rare: too many surviving dimensions.  Peel the outermost dimension and
    # recurse on zero-copy sub-views.
    outer = int(shape[0])
    for i in range(outer):
        _add_strided_one(t[i], scalar)


def _run_group(group, scalar):
    contig = []
    numels = []
    others = []
    for t in group:
        if not isinstance(t, torch.Tensor):
            raise TypeError("_foreach_add_: expected Tensor")
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            if t.is_complex():
                raise RuntimeError("foreach_add_: complex tensors are not supported")
            contig.append(t)
            numels.append(n)
        else:
            others.append(t)
    if contig:
        sval = _scalar_value(contig[0], scalar)
        nc = len(contig)
        if nc == 1:
            _launch_flat(contig[0], numels[0], sval)
        else:
            for i in range(0, nc, _FUSE_SLOTS):
                chunk = contig[i : i + _FUSE_SLOTS]
                cnumels = numels[i : i + _FUSE_SLOTS]
                if len(chunk) == 1:
                    _launch_flat(chunk[0], cnumels[0], sval)
                elif not _launch_fused(chunk, cnumels, sval):
                    for j in range(len(chunk)):
                        _launch_flat(chunk[j], cnumels[j], sval)
    for t in others:
        _add_strided_one(t, scalar)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_add__Scalar(self, scalar):
    logger.debug("GEMS_MTHREADS _FOREACH_ADD__SCALAR")

    if not _list_supported(self):
        return default_foreach_add__Scalar(self, scalar)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_add__Scalar(self, scalar)


def _launch_foreach_add__Scalar(self, scalar):
    tensors = self
    if tensors is None:
        return None
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    try:
        ntensor = len(tensors)
    except TypeError:
        tensors = (tensors,)
        ntensor = 1
    if ntensor == 0:
        return None
    if isinstance(scalar, torch.Tensor):
        if scalar.numel() != 1:
            raise RuntimeError(
                "_foreach_add_: tensor-list overload is not supported here"
            )
        scalar = scalar.item()
    if isinstance(scalar, (list, tuple)):
        if len(scalar) != ntensor:
            raise RuntimeError(
                "_foreach_add_: Length of tensors and scalars must match"
            )
        for i in range(ntensor):
            t = tensors[i]
            if not isinstance(t, torch.Tensor):
                raise TypeError("_foreach_add_: expected Tensor")
            if t.numel() == 0:
                continue
            if t.is_contiguous():
                if t.is_complex():
                    raise RuntimeError(
                        "foreach_add_: complex tensors are not supported"
                    )
                _launch_flat(t, t.numel(), _scalar_value(t, scalar[i]))
            else:
                _add_strided_one(t, scalar[i])
        return None

    first = tensors[0]
    same = isinstance(first, torch.Tensor)
    if same:
        dtype = first.dtype
        dev = first.device
        for i in range(1, ntensor):
            t = tensors[i]
            if not isinstance(t, torch.Tensor):
                same = False
                break
            if t.dtype is not dtype or t.device != dev:
                same = False
                break
    if same:
        _run_group(tensors, scalar)
    else:
        groups = {}
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise TypeError("_foreach_add_: expected Tensor")
            groups.setdefault((t.device, t.dtype), []).append(t)
        for g in groups.values():
            _run_group(g, scalar)
    return None
