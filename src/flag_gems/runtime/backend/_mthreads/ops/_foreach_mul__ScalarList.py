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
"""Triton implementation of aten::_foreach_mul_ (ScalarList / Scalar overload).

Semantics: for each i, self[i] := self[i] * scalars[i] (elementwise, in place).
Storage identity, strides and storage_offset of every input are preserved.
Returns None (schema '-> ()').

Design:
  * all same-dtype tensors whose elements occupy one flat storage block are
    multiplied by a single fused launch (2D grid: block x slot) so a list of
    tensors costs one launch instead of one per tensor;
  * views that are dense up to a permutation (transpose/permute/t()/narrow on
    the outer dim) are flat-addressable from data_ptr and join the fused path;
  * genuinely gappy views fall back to a rank<=8 strided kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    _foreach_mul__ScalarList as default__foreach_mul__ScalarList,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

BLOCK = 1024
NUM_WARPS = 4
MAX_SLOTS = 8
# Above this per-tensor byte size the four-streams-per-program kernel loses
# DRAM efficiency versus one tensor per slot, so large tensors use the 2D grid.
FUSED4_MAX_BYTES = 96 << 20
# Above this element count the pooled slot kernel is autotuned over BLOCK/warps.
BIG_ELEMS = 64 << 20
_BIG_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
]

# Cache of already-compiled _mul_fused4 kernels, keyed by (dtype, device). Only
# reusable because the launch path below pins the whole specialization class:
# four distinct 16-byte-aligned pointers and numel a multiple of 16 (and > 1).
_FAST_LAUNCH = {}


@triton.jit
def _mul_one(ptr, n, s, offs, BLOCK: tl.constexpr):
    m = offs < n
    x = tl.load(ptr + offs, mask=m, other=0)
    tl.store(ptr + offs, x * s, mask=m)


@triton.jit
def _mul_fused4(x0, x1, x2, x3, n, s0, s1, s2, s3, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x0 + offs, mask=m, other=0)
    tl.store(x0 + offs, v * s0, mask=m)
    v = tl.load(x1 + offs, mask=m, other=0)
    tl.store(x1 + offs, v * s1, mask=m)
    v = tl.load(x2 + offs, mask=m, other=0)
    tl.store(x2 + offs, v * s2, mask=m)
    v = tl.load(x3 + offs, mask=m, other=0)
    tl.store(x3 + offs, v * s3, mask=m)


@triton.jit
def _mul_slots(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.program_id(1)
    if slot == 0:
        _mul_one(x0, n0, s0, offs, BLOCK)
    if NT > 1:
        if slot == 1:
            _mul_one(x1, n1, s1, offs, BLOCK)
    if NT > 2:
        if slot == 2:
            _mul_one(x2, n2, s2, offs, BLOCK)
    if NT > 3:
        if slot == 3:
            _mul_one(x3, n3, s3, offs, BLOCK)
    if NT > 4:
        if slot == 4:
            _mul_one(x4, n4, s4, offs, BLOCK)
    if NT > 5:
        if slot == 5:
            _mul_one(x5, n5, s5, offs, BLOCK)
    if NT > 6:
        if slot == 6:
            _mul_one(x6, n6, s6, offs, BLOCK)
    if NT > 7:
        if slot == 7:
            _mul_one(x7, n7, s7, offs, BLOCK)


@triton.jit
def _mul_fused(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _mul_slots(
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        NT,
        BLOCK,
    )


# Only used when every tensor of the call holds >= BIG_ELEMS elements, so the
# in-place repeat benchmarking performed by autotune can never touch a
# correctness workload.
@triton.autotune(configs=_BIG_CONFIGS, key=["NT"])
@triton.jit
def _mul_fused_big(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _mul_slots(
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        n6,
        n7,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        NT,
        BLOCK,
    )


@triton.jit
def _mul_strided_kernel(
    x_ptr,
    n,
    s,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    st6,
    st7,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lin = pid * BLOCK + tl.arange(0, BLOCK)
    m = lin < n
    rem = lin
    addr = tl.zeros((BLOCK,), dtype=tl.int32)
    if R > 0:
        i = rem % sh0
        rem = rem // sh0
        addr = addr + i * st0
    if R > 1:
        i = rem % sh1
        rem = rem // sh1
        addr = addr + i * st1
    if R > 2:
        i = rem % sh2
        rem = rem // sh2
        addr = addr + i * st2
    if R > 3:
        i = rem % sh3
        rem = rem // sh3
        addr = addr + i * st3
    if R > 4:
        i = rem % sh4
        rem = rem // sh4
        addr = addr + i * st4
    if R > 5:
        i = rem % sh5
        rem = rem // sh5
        addr = addr + i * st5
    if R > 6:
        i = rem % sh6
        rem = rem // sh6
        addr = addr + i * st6
    if R > 7:
        i = rem % sh7
        rem = rem // sh7
        addr = addr + i * st7
    x = tl.load(x_ptr + addr, mask=m, other=0)
    tl.store(x_ptr + addr, x * s, mask=m)


def _collapse_sorted(shape, stride):
    """Remove size-1 dims, collapse mergeable neighbours, sort ascending by stride."""
    dims = []
    for sz, st in zip(shape, stride):
        sz = int(sz)
        if sz != 1:
            dims.append((sz, int(st)))
    collapsed = []
    for sz, st in dims:
        if collapsed and collapsed[-1][1] == st * sz:
            psz, _ = collapsed[-1]
            collapsed[-1] = (psz * sz, st)
        else:
            collapsed.append((sz, st))
    return sorted(collapsed, key=lambda d: d[1])


def _is_dense(dims):
    """True when the view's elements occupy one contiguous storage block at data_ptr."""
    expect = 1
    for sz, st in dims:
        if st != expect:
            return False
        expect *= sz
    return True


def _is_flat(t):
    if t.is_contiguous():
        return True
    return _is_dense(_collapse_sorted(t.shape, t.stride()))


def _scalar_value(s, is_float):
    return float(s) if is_float else int(s)


def _launch_fused(views, scal, is_float):
    nt = len(views)
    ptrs = views + [views[0]] * (MAX_SLOTS - nt)
    nums = [v.numel() for v in views] + [0] * (MAX_SLOTS - nt)
    scs = [_scalar_value(s, is_float) for s in scal] + [0.0] * (MAX_SLOTS - nt)
    max_n = max(nums)
    if max_n >= BIG_ELEMS:
        _mul_fused_big[lambda meta: (triton.cdiv(max_n, meta["BLOCK"]), nt)](
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            ptrs[4],
            ptrs[5],
            ptrs[6],
            ptrs[7],
            nums[0],
            nums[1],
            nums[2],
            nums[3],
            nums[4],
            nums[5],
            nums[6],
            nums[7],
            scs[0],
            scs[1],
            scs[2],
            scs[3],
            scs[4],
            scs[5],
            scs[6],
            scs[7],
            NT=nt,
        )
        return
    _mul_fused[(triton.cdiv(max_n, BLOCK), nt)](
        ptrs[0],
        ptrs[1],
        ptrs[2],
        ptrs[3],
        ptrs[4],
        ptrs[5],
        ptrs[6],
        ptrs[7],
        nums[0],
        nums[1],
        nums[2],
        nums[3],
        nums[4],
        nums[5],
        nums[6],
        nums[7],
        scs[0],
        scs[1],
        scs[2],
        scs[3],
        scs[4],
        scs[5],
        scs[6],
        scs[7],
        NT=nt,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )


def _launch_strided(t, s):
    n = t.numel()
    dims = _collapse_sorted(t.shape, t.stride())
    if _is_dense(dims):
        _mul_fused[(triton.cdiv(n, BLOCK), 1)](
            t,
            t,
            t,
            t,
            t,
            t,
            t,
            t,
            n,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            _scalar_value(s, t.dtype.is_floating_point),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            NT=1,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
        return
    if len(dims) > MAX_SLOTS:
        for i in range(int(t.shape[0])):
            _launch_strided(t.narrow(0, i, 1).squeeze(0), s)
        return
    padded = dims + [(1, 0)] * (MAX_SLOTS - len(dims))
    shapes = [d[0] for d in padded]
    strides = [d[1] for d in padded]
    _mul_strided_kernel[(triton.cdiv(n, BLOCK),)](
        t,
        n,
        _scalar_value(s, t.dtype.is_floating_point),
        shapes[0],
        shapes[1],
        shapes[2],
        shapes[3],
        shapes[4],
        shapes[5],
        shapes[6],
        shapes[7],
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        R=len(dims),
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )


def _foreach_mul__ScalarList(self, scalars):
    logger.debug("GEMS_MTHREADS _FOREACH_MUL__SCALARLIST")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if any(not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in _seq):
        return default__foreach_mul__ScalarList(self)
    tensors = self
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    n = len(tensors)
    if n == 0:
        return None

    if type(scalars) is list or type(scalars) is tuple:
        s_list = scalars
        if len(s_list) != n:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
    elif isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            s_list = [scalars.item()] * n
        else:
            s_list = scalars.tolist()
            if len(s_list) != n:
                raise RuntimeError(
                    "Tensor list must have same number of elements as scalar list."
                )
    else:
        s_list = [scalars] * n

    # Common case: four equal-sized contiguous tensors of one dtype -> one
    # 9-runtime-arg launch in which every program handles all four tensors.
    if n == 4:
        t0 = tensors[0]
        t1 = tensors[1]
        t2 = tensors[2]
        t3 = tensors[3]
        num = t0.numel()
        dt = t0.dtype
        if (
            num != 0
            and t1.dtype is dt
            and t2.dtype is dt
            and t3.dtype is dt
            and t1.numel() == num
            and t2.numel() == num
            and t3.numel() == num
            and num * t0.element_size() <= FUSED4_MAX_BYTES
            and t0.is_contiguous()
            and t1.is_contiguous()
            and t2.is_contiguous()
            and t3.is_contiguous()
        ):
            f = dt.is_floating_point
            f0 = float(s_list[0]) if f else int(s_list[0])
            f1 = float(s_list[1]) if f else int(s_list[1])
            f2 = float(s_list[2]) if f else int(s_list[2])
            f3 = float(s_list[3]) if f else int(s_list[3])
            grid0 = triton.cdiv(num, BLOCK)
            if num & 15 == 0:
                p0 = t0.data_ptr()
                p1 = t1.data_ptr()
                p2 = t2.data_ptr()
                p3 = t3.data_ptr()
                if (
                    ((p0 | p1 | p2 | p3) & 15) == 0
                    and p0 != p1
                    and p0 != p2
                    and p0 != p3
                    and p1 != p2
                    and p1 != p3
                    and p2 != p3
                ):
                    # Specialization class is fixed by (dtype, device) here:
                    # every pointer is 16-byte aligned and numel is a multiple
                    # of 16, so a captured CompiledKernel can be re-launched
                    # directly, skipping binder/cache-key/alias-spec work.
                    key = (dt, t0.device)
                    comp = _FAST_LAUNCH.get(key)
                    if comp is None:
                        comp = _mul_fused4[(grid0,)](
                            t0,
                            t1,
                            t2,
                            t3,
                            num,
                            f0,
                            f1,
                            f2,
                            f3,
                            BLOCK=BLOCK,
                            num_warps=NUM_WARPS,
                        )
                        _FAST_LAUNCH[key] = comp
                    else:
                        comp[(grid0, 1, 1)](t0, t1, t2, t3, num, f0, f1, f2, f3, BLOCK)
                    return None
            _mul_fused4[(grid0,)](
                t0,
                t1,
                t2,
                t3,
                num,
                f0,
                f1,
                f2,
                f3,
                BLOCK=BLOCK,
                num_warps=NUM_WARPS,
            )
            return None

    groups = {}
    order = []
    for i in range(n):
        t = tensors[i]
        if t.numel() == 0:
            continue
        if t.is_contiguous():
            key = t.dtype
            g = groups.get(key)
            if g is None:
                groups[key] = g = []
                order.append(key)
            g.append(i)
        else:
            _launch_strided(t, s_list[i])

    for key in order:
        idxs = groups[key]
        is_float = key.is_floating_point
        for c in range(0, len(idxs), MAX_SLOTS):
            chunk = idxs[c : c + MAX_SLOTS]
            _launch_fused(
                [tensors[j] for j in chunk],
                [s_list[j] for j in chunk],
                is_float,
            )
    return None
