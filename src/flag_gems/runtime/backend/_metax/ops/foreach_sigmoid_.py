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

"""In-place Triton implementation of ``aten::_foreach_sigmoid_``.

Semantics reproduced from ATen:
  * ``self`` must be a non-empty list/tuple of tensors, else the matching
    exception is raised before any mutation happens.
  * ``sigmoid`` promotes integral/bool inputs to float32, so the in-place
    variant rejects them ("result type Float32 can't be cast ...").
  * every element is written back to its own storage slot; shape, stride,
    storage_offset and dtype are untouched and the call returns ``None``.

Execution strategy: all flat-addressable tensors of one ``(device, dtype)``
group are processed by a *single* launch, with their base pointers and element
counts passed as int64 scalar arguments.  No device-side descriptor buffer is
built: a host->device copy from pageable memory costs a blocking stream sync
here, which is far more expensive than the launch it would save.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

# Per-dtype launch table: (kind, triton dtype, opmath dtype, use fast math).
# kind 1 = real floating (in place), 2 = complex.  Looked up once per tensor,
# which keeps the hot path to a single dict hit plus a tuple unpack.
_INFO = {
    torch.float16: (1, tl.float16, tl.float32, True),
    torch.bfloat16: (1, tl.bfloat16, tl.float32, True),
    torch.float32: (1, tl.float32, tl.float32, False),
    torch.float64: (1, tl.float64, tl.float64, False),
    torch.complex64: (2, tl.float32, tl.float32, False),
    torch.complex128: (2, tl.float64, tl.float64, False),
}

# Integral / bool inputs: ATen promotes to float32 and then refuses the
# in-place cast back into the input dtype.
_CAST_ERROR = {
    torch.int8: "Int8",
    torch.int16: "Int16",
    torch.int32: "Int32",
    torch.int64: "Int64",
    torch.uint8: "Uint8",
    torch.bool: "Bool",
}


def _dtype_info(dt):
    info = _INFO.get(dt)
    if info is not None:
        return info
    name = _CAST_ERROR.get(dt)
    if name is not None:
        raise RuntimeError(
            "result type Float32 can't be cast to the desired output type %s" % name
        )
    raise RuntimeError(
        "\"foreach\" not implemented for '%s'" % str(dt).replace("torch.", "")
    )


_FLAT_BLOCK = 1024
_STRIDED_BLOCK = 512
_MAX_RANK = 8
_MAX_BATCH = 16
# Group traffic (read + write) above which per-tensor launches beat fusion.
_FUSE_BYTES = 16 << 20

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sig_flat_kernel(
    x_ptr,
    n_elements,
    COMPLEX: tl.constexpr,
    FAST: tl.constexpr,
    ACC_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Element-wise in-place sigmoid over one flat-addressable tensor."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    if COMPLEX:
        re = tl.load(x_ptr + 2 * offs, mask=mask, other=0.0).to(ACC_DT)
        im = tl.load(x_ptr + 2 * offs + 1, mask=mask, other=0.0).to(ACC_DT)
        ea = tl.exp(-re)
        cb = tl.cos(im)
        sb = tl.sin(im)
        dr = 1.0 + ea * cb
        di = -ea * sb
        den = dr * dr + di * di
        tl.store(x_ptr + 2 * offs, (dr / den).to(IN_DT), mask=mask)
        tl.store(x_ptr + 2 * offs + 1, (-di / den).to(IN_DT), mask=mask)
    else:
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(ACC_DT)
        if FAST:
            # 16-bit results round away far more than this costs, so use the
            # cheap transcendental: 1/(1+exp(-x)) = 1/(1+exp2(-x*log2(e))).
            y = tl.fdiv(
                1.0,
                1.0 + tl.exp2(-x * 1.4426950408889634),
                ieee_rounding=False,
            )
        else:
            y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(x_ptr + offs, y.to(IN_DT), mask=mask)


@triton.jit
def _sig_multi_kernel(
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
    FAST: tl.constexpr,
    ACC_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Up to 16 flat tensors in one launch, with no device-side metadata.

    Axis 0 selects the tensor (so unused slots cost no programs at all), axis 1
    the chunk inside it.  Pointers arrive as int64 scalar arguments and are
    selected with a ``tl.where`` chain: one uniform scalar select per program,
    which avoids the host->device copy (and its blocking stream sync) that a
    device descriptor buffer would need.
    """
    s = tl.program_id(0)
    p = p0
    n = n0
    p = tl.where(s == 1, p1, p)
    n = tl.where(s == 1, n1, n)
    p = tl.where(s == 2, p2, p)
    n = tl.where(s == 2, n2, n)
    p = tl.where(s == 3, p3, p)
    n = tl.where(s == 3, n3, n)
    p = tl.where(s == 4, p4, p)
    n = tl.where(s == 4, n4, n)
    p = tl.where(s == 5, p5, p)
    n = tl.where(s == 5, n5, n)
    p = tl.where(s == 6, p6, p)
    n = tl.where(s == 6, n6, n)
    p = tl.where(s == 7, p7, p)
    n = tl.where(s == 7, n7, n)
    p = tl.where(s == 8, p8, p)
    n = tl.where(s == 8, n8, n)
    p = tl.where(s == 9, p9, p)
    n = tl.where(s == 9, n9, n)
    p = tl.where(s == 10, p10, p)
    n = tl.where(s == 10, n10, n)
    p = tl.where(s == 11, p11, p)
    n = tl.where(s == 11, n11, n)
    p = tl.where(s == 12, p12, p)
    n = tl.where(s == 12, n12, n)
    p = tl.where(s == 13, p13, p)
    n = tl.where(s == 13, n13, n)
    p = tl.where(s == 14, p14, p)
    n = tl.where(s == 14, n14, n)
    p = tl.where(s == 15, p15, p)
    n = tl.where(s == 15, n15, n)
    ptr = p.to(tl.pointer_type(IN_DT))
    idx = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    x = tl.load(ptr + idx, mask=mask, other=0.0).to(ACC_DT)
    if FAST:
        y = tl.fdiv(
            1.0,
            1.0 + tl.exp2(-x * 1.4426950408889634),
            ieee_rounding=False,
        )
    else:
        y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(ptr + idx, y.to(IN_DT), mask=mask)


@triton.jit
def _sig_multi_eq_kernel(
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
    n_elements,
    FAST: tl.constexpr,
    ACC_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """As :func:`_sig_multi_kernel` but all slots share one element count.

    Passing the shared count once instead of sixteen times removes fifteen
    int64 arguments from the launch.  On this target each scalar argument costs
    roughly a microsecond of host binding time, which is large next to the
    ~1 us of GPU work the tiny-tensor workloads actually need.
    """
    s = tl.program_id(0)
    p = p0
    p = tl.where(s == 1, p1, p)
    p = tl.where(s == 2, p2, p)
    p = tl.where(s == 3, p3, p)
    p = tl.where(s == 4, p4, p)
    p = tl.where(s == 5, p5, p)
    p = tl.where(s == 6, p6, p)
    p = tl.where(s == 7, p7, p)
    p = tl.where(s == 8, p8, p)
    p = tl.where(s == 9, p9, p)
    p = tl.where(s == 10, p10, p)
    p = tl.where(s == 11, p11, p)
    p = tl.where(s == 12, p12, p)
    p = tl.where(s == 13, p13, p)
    p = tl.where(s == 14, p14, p)
    p = tl.where(s == 15, p15, p)
    ptr = p.to(tl.pointer_type(IN_DT))
    idx = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n_elements
    x = tl.load(ptr + idx, mask=mask, other=0.0).to(ACC_DT)
    if FAST:
        y = tl.fdiv(
            1.0,
            1.0 + tl.exp2(-x * 1.4426950408889634),
            ieee_rounding=False,
        )
    else:
        y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(ptr + idx, y.to(IN_DT), mask=mask)


@triton.jit
def _sig_strided_kernel(
    x_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    RANK: tl.constexpr,
    COMPLEX: tl.constexpr,
    ACC_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Element-wise in-place sigmoid over a gappy (but non-overlapping) view."""
    pid = tl.program_id(0)
    rem = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = rem < n_elements
    off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        off += (rem % s0) * t0
        rem = rem // s0
    if RANK > 1:
        off += (rem % s1) * t1
        rem = rem // s1
    if RANK > 2:
        off += (rem % s2) * t2
        rem = rem // s2
    if RANK > 3:
        off += (rem % s3) * t3
        rem = rem // s3
    if RANK > 4:
        off += (rem % s4) * t4
        rem = rem // s4
    if RANK > 5:
        off += (rem % s5) * t5
        rem = rem // s5
    if RANK > 6:
        off += (rem % s6) * t6
        rem = rem // s6
    if RANK > 7:
        off += (rem % s7) * t7
        rem = rem // s7
    if COMPLEX:
        re = tl.load(x_ptr + 2 * off, mask=mask, other=0.0).to(ACC_DT)
        im = tl.load(x_ptr + 2 * off + 1, mask=mask, other=0.0).to(ACC_DT)
        ea = tl.exp(-re)
        cb = tl.cos(im)
        sb = tl.sin(im)
        dr = 1.0 + ea * cb
        di = -ea * sb
        den = dr * dr + di * di
        tl.store(x_ptr + 2 * off, (dr / den).to(IN_DT), mask=mask)
        tl.store(x_ptr + 2 * off + 1, (-di / den).to(IN_DT), mask=mask)
    else:
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(ACC_DT)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(x_ptr + off, y.to(IN_DT), mask=mask)


# ---------------------------------------------------------------------------
# layout helpers (host side, metadata only)
# ---------------------------------------------------------------------------


def _is_flat_addressable(t: torch.Tensor) -> bool:
    """``[data_ptr, data_ptr + numel)`` covers exactly the tensor's elements.

    True for contiguous tensors and for any non-overlapping dense view
    (transposes, permutations): flat traversal then visits every element
    exactly once, which is all an element-wise op needs.
    """
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


def _strided_meta(t: torch.Tensor):
    """Coalesced (sizes, strides) of ``t`` padded to ``_MAX_RANK`` slots."""
    dims = [(st, sz) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    dims.sort()
    merged = []
    for st, sz in dims:
        if merged and st == merged[-1][0] * merged[-1][1]:
            merged[-1] = (merged[-1][0], merged[-1][1] * sz)
        else:
            merged.append((st, sz))
    sizes = [sz for _st, sz in merged]
    strides = [st for st, _sz in merged]
    rank = len(sizes)
    while len(sizes) < _MAX_RANK:
        sizes.append(1)
        strides.append(0)
    return rank, sizes, strides


def _launch_one(t, n, acc, in_tl, is_complex, fast):
    _sig_flat_kernel[(triton.cdiv(n, _FLAT_BLOCK),)](
        t,
        n,
        is_complex,
        fast,
        acc,
        in_tl,
        _FLAT_BLOCK,
        num_warps=4,
        num_stages=1,
    )


def _launch_many(items, acc, in_tl, fast):
    """One launch for up to ``_MAX_BATCH`` flat tensors of the same dtype."""
    k = len(items)
    ptrs = [it[0] for it in items]
    n0 = items[0][1]
    same = all(it[1] == n0 for it in items)
    if k < _MAX_BATCH:
        pad = _MAX_BATCH - k
        ptrs += [ptrs[0]] * pad
    if same:
        # Uniform shapes: hand the shared element count over once, which keeps
        # fifteen int64 arguments out of the launch.
        _sig_multi_eq_kernel[(k, triton.cdiv(n0, _FLAT_BLOCK))](
            *ptrs,
            n0,
            fast,
            acc,
            in_tl,
            _FLAT_BLOCK,
            num_warps=4,
            num_stages=1,
        )
        return
    ns = [it[1] for it in items]
    if k < _MAX_BATCH:
        ns += [0] * (_MAX_BATCH - k)
    grid = (k, triton.cdiv(max(ns), _FLAT_BLOCK))
    _sig_multi_kernel[grid](
        *ptrs,
        *ns,
        fast,
        acc,
        in_tl,
        _FLAT_BLOCK,
        num_warps=4,
        num_stages=1,
    )


def _launch_strided(t, n, acc, in_tl, is_complex):
    rank, sizes, strides = _strided_meta(t)
    if rank > _MAX_RANK:
        raise NotImplementedError(
            "foreach_sigmoid_: rank %d view exceeds the %d-dim fast path"
            % (rank, _MAX_RANK)
        )
    _sig_strided_kernel[(triton.cdiv(n, _STRIDED_BLOCK),)](
        t,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        rank,
        is_complex,
        acc,
        in_tl,
        _STRIDED_BLOCK,
        num_warps=4,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def foreach_sigmoid_(self):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(self) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    # Common case: one dense tensor.  Skip the grouping bookkeeping entirely.
    if len(self) == 1:
        t = self[0]
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        kind, in_tl, acc, fast = _dtype_info(t.dtype)
        n = t.numel()
        if n and t.is_contiguous():
            _launch_one(t, n, acc, in_tl, kind == 2, fast)
            return None

    flat_groups = {}  # (device, dtype) -> [(ptr, numel, tensor)]
    gappy = None  # [(tensor, numel)] for gappy views

    # Validate the whole list (ATen check order) before mutating anything,
    # collecting launch metadata in the same pass.
    for t in self:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        kind, in_tl, acc, fast = _dtype_info(t.dtype)
        n = t.numel()
        if n == 0:
            continue
        if not t.is_contiguous() and n > 1 and not _is_flat_addressable(t):
            if torch._debug_has_internal_overlap(t) == 1:
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if gappy is None:
                gappy = [(t, n)]
            else:
                gappy.append((t, n))
            continue
        key = (t.device, t.dtype)
        grp = flat_groups.get(key)
        item = (t.data_ptr(), n, t)
        if grp is None:
            flat_groups[key] = [item]
        else:
            grp.append(item)

    # Grouped launches.  Fusing a group into one launch only pays while the
    # launches themselves are the cost: once a group moves enough bytes for the
    # GPU work to hide them, per-tensor launches win (they hand Triton a
    # pointer-typed argument, which keeps 128-bit vectorization).
    for (_dev, dt), items in flat_groups.items():
        kind, in_tl, acc, fast = _INFO[dt]
        k = len(items)
        if kind == 2:
            for _ptr, n, t in items:
                _launch_one(t, n, acc, in_tl, True, False)
            continue
        max_n = items[0][1]
        for _ptr, n, _t in items:
            if n > max_n:
                max_n = n
        if k == 1 or max_n * k * dt.itemsize * 2 > _FUSE_BYTES:
            for _ptr, n, t in items:
                _launch_one(t, n, acc, in_tl, False, fast)
            continue
        for i in range(0, k, _MAX_BATCH):
            _launch_many(items[i : i + _MAX_BATCH], acc, in_tl, fast)

    for t, n in gappy or ():
        kind, in_tl, acc, fast = _INFO[t.dtype]
        _launch_strided(t, n, acc, in_tl, kind == 2)
    return None
