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

"""Triton implementation of ``aten::_foreach_abs_`` (in-place multi-tensor abs).

Semantics (verified against the target build, torch 2.9 / ROCm 6.3):
  * every tensor in the list is mutated in place, storage/stride/list identity
    unchanged, and the call returns ``None``;
  * accepted dtypes: fp16, bf16, fp32, fp64, int8, int16, int32, int64, uint8,
    bool.  ``abs`` on bool/uint8 is the identity, signed ints wrap two's
    complement for INT_MIN, floats clear the sign bit exactly (-0.0 -> +0.0,
    NaN stays NaN);
  * complex input raises ``RuntimeError("In-place abs is not supported for
    complex tensors.")``; float8/uint16/uint32/uint64 raise
    ``NotImplementedError`` from the ATen kernel;
  * internal overlap (e.g. ``expand``) raises the ATen overlap error.

Execution shape: dense tensors are grouped by (dtype, device).  A group of small
equal-sized dense tensors is handled by ONE unmasked launch (one pointer slot per
tensor), large groups get one unmasked launch per tensor.  ``BLOCK`` is chosen so
every lane moves 16 bytes, and masks are dropped whenever the extent is an exact
multiple of ``BLOCK``: masked Triton loads lower to per-element
``buffer_load_ushort``/``buffer_load_dword`` on this target, so dropping the mask
is what unlocks wider accesses.  Gappy views use a memory-order strided kernel.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_FLAT_WARPS = 4
_STRIDED_BLOCK = 512
_STRIDED_WARPS = 4
_MAX_DIMS = 8
_FUSED_SLOTS = 16
_FUSE_MAX_BYTES = 8 << 20
_SUPPORTED = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    )
)
_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor refers "
    "to a single memory location. Please clone() the tensor before performing "
    "the operation."
)
_has_internal_overlap = getattr(torch, "_debug_has_internal_overlap", None)
_BLOCK_CACHE = {}
_DTYPE_INFO = {}
for _dt in _SUPPORTED:
    _isize = _dt.itemsize
    _DTYPE_INFO[_dt] = (_isize, max(256, 4096 // _isize))
del _dt, _isize


def _flat_block(dtype):
    return _DTYPE_INFO[dtype][1]


def _aten_dtype_name(dtype):
    name = str(dtype)
    if name.startswith("torch."):
        name = name[6:]
    if name == "bool":
        return "Bool"
    if name == "bfloat16":
        return "BFloat16"
    if name.startswith("uint"):
        return "UInt" + name[4:]
    if name.startswith("int"):
        return "Int" + name[3:]
    return name[:1].upper() + name[1:]


def _collapse(shape, stride):
    """Memory-order dims: drop size-1 axes, sort by stride, merge adjacent."""
    dims = [[sz, st] for (sz, st) in zip(shape, stride) if sz != 1]
    if not dims:
        return [[1, 1]]
    if len(dims) > 1:
        dims.sort(key=lambda d: d[1])
        out = [dims[0]]
        for sz, st in dims[1:]:
            prev = out[-1]
            if st == prev[1] * prev[0]:
                prev[0] = prev[0] * sz
            else:
                out.append([sz, st])
        dims = out
    return dims


@triton.jit(do_not_specialize=["n"])
def _abs_flat_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    val = tl.load(x_ptr + idx, mask=mask)
    tl.store(x_ptr + idx, tl.abs(val), mask=mask)


@triton.jit
def _abs_flat_full_kernel(x_ptr, BLOCK: tl.constexpr):
    """Unmasked: the grid covers exactly ``BLOCK`` elements per program."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + idx, tl.abs(tl.load(x_ptr + idx)))


@triton.jit
def _abs_fused_full_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch for ``NT`` equal-sized same-dtype dense tensors."""
    slot = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if slot == 0:
        tl.store(a0 + idx, tl.abs(tl.load(a0 + idx)))
    elif slot == 1:
        tl.store(a1 + idx, tl.abs(tl.load(a1 + idx)))
    elif slot == 2:
        tl.store(a2 + idx, tl.abs(tl.load(a2 + idx)))
    elif slot == 3:
        tl.store(a3 + idx, tl.abs(tl.load(a3 + idx)))
    elif slot == 4:
        tl.store(a4 + idx, tl.abs(tl.load(a4 + idx)))
    elif slot == 5:
        tl.store(a5 + idx, tl.abs(tl.load(a5 + idx)))
    elif slot == 6:
        tl.store(a6 + idx, tl.abs(tl.load(a6 + idx)))
    elif slot == 7:
        tl.store(a7 + idx, tl.abs(tl.load(a7 + idx)))
    elif slot == 8:
        tl.store(a8 + idx, tl.abs(tl.load(a8 + idx)))
    elif slot == 9:
        tl.store(a9 + idx, tl.abs(tl.load(a9 + idx)))
    elif slot == 10:
        tl.store(a10 + idx, tl.abs(tl.load(a10 + idx)))
    elif slot == 11:
        tl.store(a11 + idx, tl.abs(tl.load(a11 + idx)))
    elif slot == 12:
        tl.store(a12 + idx, tl.abs(tl.load(a12 + idx)))
    elif slot == 13:
        tl.store(a13 + idx, tl.abs(tl.load(a13 + idx)))
    elif slot == 14:
        tl.store(a14 + idx, tl.abs(tl.load(a14 + idx)))
    elif slot == 15:
        tl.store(a15 + idx, tl.abs(tl.load(a15 + idx)))


@triton.jit(
    do_not_specialize=[
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
        "n7",
        "n8",
        "n9",
        "n10",
        "n11",
        "n12",
        "n13",
        "n14",
        "n15",
    ]
)
def _abs_fused_kernel(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Masked variant for small groups whose extents differ."""
    slot = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if slot == 0:
        mask = idx < n0
        tl.store(a0 + idx, tl.abs(tl.load(a0 + idx, mask=mask)), mask=mask)
    elif slot == 1:
        mask = idx < n1
        tl.store(a1 + idx, tl.abs(tl.load(a1 + idx, mask=mask)), mask=mask)
    elif slot == 2:
        mask = idx < n2
        tl.store(a2 + idx, tl.abs(tl.load(a2 + idx, mask=mask)), mask=mask)
    elif slot == 3:
        mask = idx < n3
        tl.store(a3 + idx, tl.abs(tl.load(a3 + idx, mask=mask)), mask=mask)
    elif slot == 4:
        mask = idx < n4
        tl.store(a4 + idx, tl.abs(tl.load(a4 + idx, mask=mask)), mask=mask)
    elif slot == 5:
        mask = idx < n5
        tl.store(a5 + idx, tl.abs(tl.load(a5 + idx, mask=mask)), mask=mask)
    elif slot == 6:
        mask = idx < n6
        tl.store(a6 + idx, tl.abs(tl.load(a6 + idx, mask=mask)), mask=mask)
    elif slot == 7:
        mask = idx < n7
        tl.store(a7 + idx, tl.abs(tl.load(a7 + idx, mask=mask)), mask=mask)
    elif slot == 8:
        mask = idx < n8
        tl.store(a8 + idx, tl.abs(tl.load(a8 + idx, mask=mask)), mask=mask)
    elif slot == 9:
        mask = idx < n9
        tl.store(a9 + idx, tl.abs(tl.load(a9 + idx, mask=mask)), mask=mask)
    elif slot == 10:
        mask = idx < n10
        tl.store(a10 + idx, tl.abs(tl.load(a10 + idx, mask=mask)), mask=mask)
    elif slot == 11:
        mask = idx < n11
        tl.store(a11 + idx, tl.abs(tl.load(a11 + idx, mask=mask)), mask=mask)
    elif slot == 12:
        mask = idx < n12
        tl.store(a12 + idx, tl.abs(tl.load(a12 + idx, mask=mask)), mask=mask)
    elif slot == 13:
        mask = idx < n13
        tl.store(a13 + idx, tl.abs(tl.load(a13 + idx, mask=mask)), mask=mask)
    elif slot == 14:
        mask = idx < n14
        tl.store(a14 + idx, tl.abs(tl.load(a14 + idx, mask=mask)), mask=mask)
    elif slot == 15:
        mask = idx < n15
        tl.store(a15 + idx, tl.abs(tl.load(a15 + idx, mask=mask)), mask=mask)


@triton.jit(
    do_not_specialize=[
        "s0",
        "t0",
        "s1",
        "t1",
        "s2",
        "t2",
        "s3",
        "t3",
        "s4",
        "t4",
        "s5",
        "t5",
        "s6",
        "t6",
        "s7",
        "t7",
    ]
)
def _abs_strided_kernel(
    x_ptr,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    s7,
    t7,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (outer row, fast-axis segment) in memory order."""
    r = tl.program_id(0).to(tl.int64)
    base = r - r
    if R > 1:
        base += r % s1 * t1
        r = r // s1
    if R > 2:
        base += r % s2 * t2
        r = r // s2
    if R > 3:
        base += r % s3 * t3
        r = r // s3
    if R > 4:
        base += r % s4 * t4
        r = r // s4
    if R > 5:
        base += r % s5 * t5
        r = r // s5
    if R > 6:
        base += r % s6 * t6
        r = r // s6
    if R > 7:
        base += r % s7 * t7
        r = r // s7
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < s0
    ptr = x_ptr + base + off.to(tl.int64) * t0
    val = tl.load(ptr, mask=mask)
    tl.store(ptr, tl.abs(val), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n",
        "s0",
        "t0",
        "s1",
        "t1",
        "s2",
        "t2",
        "s3",
        "t3",
        "s4",
        "t4",
        "s5",
        "t5",
        "s6",
        "t6",
        "s7",
        "t7",
    ]
)
def _abs_generic_kernel(
    x_ptr,
    n,
    s0,
    t0,
    s1,
    t1,
    s2,
    t2,
    s3,
    t3,
    s4,
    t4,
    s5,
    t5,
    s6,
    t6,
    s7,
    t7,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Waste-free fallback: linear index, offset rebuilt from the collapsed dims."""
    pid = tl.program_id(0).to(tl.int64)
    flat = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = flat < n
    r = flat
    off = flat - flat
    off += r % s0 * t0
    r = r // s0
    if R > 1:
        off += r % s1 * t1
        r = r // s1
    if R > 2:
        off += r % s2 * t2
        r = r // s2
    if R > 3:
        off += r % s3 * t3
        r = r // s3
    if R > 4:
        off += r % s4 * t4
        r = r // s4
    if R > 5:
        off += r % s5 * t5
        r = r // s5
    if R > 6:
        off += r % s6 * t6
        r = r // s6
    if R > 7:
        off += r % s7 * t7
        r = r // s7
    ptr = x_ptr + off
    val = tl.load(ptr, mask=mask)
    tl.store(ptr, tl.abs(val), mask=mask)


def _launch_flat(tensor, numel, block):
    if numel % block == 0:
        _abs_flat_full_kernel[numel // block,](
            tensor, BLOCK=block, num_warps=_FLAT_WARPS
        )
    else:
        _abs_flat_kernel[(numel + block - 1) // block,](
            tensor, numel, BLOCK=block, num_warps=_FLAT_WARPS
        )


def _launch_uniform(seq, count, numel, block, itemsize):
    """All tensors dense, same size and dtype: fuse when that is the cheaper path."""
    if count > _FUSED_SLOTS:
        for start in range(0, count, _FUSED_SLOTS):
            chunk = seq[start : start + _FUSED_SLOTS]
            _launch_uniform(chunk, len(chunk), numel, block, itemsize)
        return
    if count == 1:
        _launch_flat(seq[0], numel, block)
        return
    if numel * itemsize > _FUSE_MAX_BYTES:
        for tensor in seq:
            _launch_flat(tensor, numel, block)
        return
    ptrs = list(seq)
    pad = _FUSED_SLOTS - count
    if numel % block == 0:
        if pad:
            ptrs.extend([ptrs[-1]] * pad)
        _abs_fused_full_kernel[count, numel // block](
            *ptrs, count, BLOCK=block, num_warps=_FLAT_WARPS
        )
        return
    sizes = [numel] * count
    if pad:
        ptrs.extend([ptrs[-1]] * pad)
        sizes.extend([0] * pad)
    _abs_fused_kernel[count, (numel + block - 1) // block](
        *ptrs, *sizes, count, BLOCK=block, num_warps=_FLAT_WARPS
    )


def _launch_group(items, block, itemsize):
    """Small same-dtype dense group: one launch, or per-tensor when large."""
    count = len(items)
    max_numel = items[0][1]
    min_numel = max_numel
    for _, numel in items[1:]:
        if numel > max_numel:
            max_numel = numel
        if numel < min_numel:
            min_numel = numel
    if min_numel == max_numel:
        _launch_uniform(
            [tensor for (tensor, _) in items], count, max_numel, block, itemsize
        )
        return
    if max_numel * itemsize > _FUSE_MAX_BYTES:
        for tensor, numel in items:
            _launch_flat(tensor, numel, block)
        return
    ptrs = [tensor for (tensor, _) in items]
    sizes = [numel for (_, numel) in items]
    pad = _FUSED_SLOTS - count
    if pad:
        ptrs.extend([ptrs[-1]] * pad)
        sizes.extend([0] * pad)
    grid = (count, (max_numel + block - 1) // block)
    _abs_fused_kernel[grid](*ptrs, *sizes, count, BLOCK=block, num_warps=_FLAT_WARPS)


def _launch_strided(tensor, dims):
    rank = len(dims)
    if rank > _MAX_DIMS:
        raise NotImplementedError(
            "foreach_abs_: strided tensor with %d collapsed dimensions" % rank
        )
    s0, t0 = dims[0]
    outer = 1
    for sz, _ in dims[1:]:
        outer *= sz
    args = [tensor, s0, t0]
    for k in range(1, _MAX_DIMS):
        if k < rank:
            args.extend(dims[k])
        else:
            args.extend((1, 0))
    if s0 >= 16:
        grid = (outer, (s0 + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK)
        _abs_strided_kernel[grid](
            *args, rank, BLOCK=_STRIDED_BLOCK, num_warps=_STRIDED_WARPS
        )
    else:
        numel = s0 * outer
        grid = ((numel + _STRIDED_BLOCK - 1) // _STRIDED_BLOCK,)
        _abs_generic_kernel[grid](
            tensor,
            numel,
            *args[1:],
            rank,
            BLOCK=_STRIDED_BLOCK,
            num_warps=_STRIDED_WARPS,
        )


def _foreach_abs_(self):
    logger.debug("GEMS_HYGON _FOREACH_ABS_ (GEMS _FOREACH_ABS_)")
    if not isinstance(self, (list, tuple)):
        raise TypeError(
            "_foreach_abs_(): argument 'self' (position 1) must be tuple of Tensors, not %s"
            % type(self).__name__
        )
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    uniform = False
    try:
        first = self[0]
        dtype = first.dtype
        info = _DTYPE_INFO.get(dtype)
        if info is not None and first.is_contiguous():
            numel = first.numel()
            dev = first.get_device()
            count = len(self)
            i = 1
            while i < count:
                t = self[i]
                if (
                    t.dtype is not dtype
                    or t.numel() != numel
                    or t.get_device() != dev
                    or (not t.is_contiguous())
                ):
                    break
                i += 1
            uniform = i == count
    except Exception:
        uniform = False
    if uniform:
        if numel:
            _launch_uniform(self, count, numel, info[1], info[0])
        return None
    groups = {}
    strided_items = []
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("_foreach_abs_(): argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype.is_complex:
            raise RuntimeError("In-place abs is not supported for complex tensors.")
        if dtype not in _SUPPORTED:
            raise NotImplementedError(
                "\"foreach_unary_op_cuda\" not implemented for '%s'"
                % _aten_dtype_name(dtype)
            )
        numel = t.numel()
        if numel == 0:
            continue
        if t.is_contiguous():
            dense = True
            dims = None
        else:
            if _has_internal_overlap is not None and _has_internal_overlap(t) == 1:
                raise RuntimeError(_OVERLAP_MSG)
            dims = _collapse(t.shape, t.stride())
            dense = len(dims) == 1 and dims[0][1] == 1
        if dense:
            key = (dtype, t.get_device())
            try:
                groups[key].append((t, numel))
            except KeyError:
                groups[key] = [(t, numel)]
        else:
            strided_items.append((t, dims))
    for key, items in groups.items():
        dtype, _dev = key
        block = _flat_block(dtype)
        if len(items) == 1:
            _launch_flat(items[0][0], items[0][1], block)
        elif len(items) <= _FUSED_SLOTS:
            _launch_group(items, block, dtype.itemsize)
        else:
            itemsize = dtype.itemsize
            for i in range(0, len(items), _FUSED_SLOTS):
                chunk = items[i : i + _FUSED_SLOTS]
                if len(chunk) == 1:
                    _launch_flat(chunk[0][0], chunk[0][1], block)
                else:
                    _launch_group(chunk, block, itemsize)
    for tensor, dims in strided_items:
        _launch_strided(tensor, dims)
    return None
