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

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tuned launch parameters (fixed; verified through formal evaluation rounds)
# ---------------------------------------------------------------------------
_BLOCK = 2048
# 64*num_warps = 256 threads, so these blocks give every thread 8 bytes of
# traffic per access, which measured fastest for each element size.
_BLOCK_OF = {torch.float32: 512, torch.float16: 1024, torch.bfloat16: 1024}
_WARPS = 4
# offsets computed with pid * BLOCK stay inside int32 while n is below this
_I64_LIMIT = (1 << 31) - 65536

# Cache of prepared launch handles: (dtype, device index) -> plan tuple
_FAST_PLAN = {}
_FAST_OK = True
_current_stream = torch.cuda.current_stream


@triton.jit
def _max_nan(a, b):
    # torch.maximum semantics: NaN propagates from either operand, otherwise a > b.
    return tl.where(a != a, a, tl.where(b != b, b, tl.where(a > b, a, b)))


@triton.jit
def _flat_tile(a_ptr, b_ptr, offs, n):
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(a_ptr + offs, _max_nan(a, b), mask=mask)


@triton.jit
def _maximum_flat_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr, I64: tl.constexpr):
    pid = tl.program_id(0)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    _flat_tile(a_ptr, b_ptr, base + tl.arange(0, BLOCK), n)


@triton.jit
def _maximum_fused4_kernel(
    a0,
    b0,
    a1,
    b1,
    a2,
    b2,
    a3,
    b3,
    n,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    which = tl.program_id(1)
    if I64:
        base = pid.to(tl.int64) * BLOCK
    else:
        base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    if which == 0:
        _flat_tile(a0, b0, offs, n)
    elif which == 1:
        _flat_tile(a1, b1, offs, n)
    elif which == 2:
        _flat_tile(a2, b2, offs, n)
    else:
        _flat_tile(a3, b3, offs, n)


@triton.jit
def _maximum_strided_kernel(
    a_ptr,
    b_ptr,
    n,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    as0,
    as1,
    as2,
    as3,
    as4,
    as5,
    as6,
    as7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    rem = offs
    c = rem % n7
    rem = rem // n7
    oa = c * as7
    ob = c * bs7
    c = rem % n6
    rem = rem // n6
    oa += c * as6
    ob += c * bs6
    c = rem % n5
    rem = rem // n5
    oa += c * as5
    ob += c * bs5
    c = rem % n4
    rem = rem // n4
    oa += c * as4
    ob += c * bs4
    c = rem % n3
    rem = rem // n3
    oa += c * as3
    ob += c * bs3
    c = rem % n2
    rem = rem // n2
    oa += c * as2
    ob += c * bs2
    c = rem % n1
    rem = rem // n1
    oa += c * as1
    ob += c * bs1
    oa += rem * as0
    ob += rem * bs0

    a = tl.load(a_ptr + oa, mask=mask, other=0.0)
    b = tl.load(b_ptr + ob, mask=mask, other=0.0)
    tl.store(a_ptr + oa, _max_nan(a, b), mask=mask)


def _strided_launch(a, b):
    """Stride-aware in-place maximum for a pair that is not flat-compatible."""
    n = a.numel()
    if n == 0:
        return
    ashape = a.shape
    rank = len(ashape)
    if rank > 8:
        # Extremely rare: peel the leading dimension and recurse (views only).
        for i in range(ashape[0]):
            av = a[i]
            bd = len(b.shape) - rank
            if bd >= 0:
                bv = b[0] if b.shape[0] == 1 else b[i]
            else:
                bv = b
            _strided_launch(av, bv)
        return

    astride = a.stride()
    bshape = b.shape
    bstride = b.stride()
    off = len(bshape) - rank
    if off > 0:
        # b has extra leading dims; in-place is only valid when they are all 1
        for d in range(off):
            if bshape[d] != 1:
                raise RuntimeError(
                    "output tensor is not large enough for the broadcast result"
                )
        off = 0

    sizes = [1, 1, 1, 1, 1, 1, 1, 1]
    asst = [0, 0, 0, 0, 0, 0, 0, 0]
    bsst = [0, 0, 0, 0, 0, 0, 0, 0]
    start = 8 - rank
    for d in range(rank):
        s = ashape[d]
        bd = len(bshape) - rank + d
        bs = bshape[bd] if bd >= 0 else 1
        if bs != s and bs != 1:
            raise RuntimeError(
                "output tensor is not large enough for the broadcast result"
            )
        sizes[start + d] = s
        asst[start + d] = astride[d]
        bsst[start + d] = bstride[bd] if bs != 1 else 0

    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _maximum_strided_kernel[grid](
        a,
        b,
        n,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        sizes[6],
        sizes[7],
        asst[0],
        asst[1],
        asst[2],
        asst[3],
        asst[4],
        asst[5],
        asst[6],
        asst[7],
        bsst[0],
        bsst[1],
        bsst[2],
        bsst[3],
        bsst[4],
        bsst[5],
        bsst[6],
        bsst[7],
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _pair(a, b):
    n = a.numel()
    if n == 0:
        return
    if a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK,)
        _maximum_flat_kernel[grid](
            a, b, n, BLOCK=_BLOCK, I64=n > _I64_LIMIT, num_warps=_WARPS
        )
        return
    _strided_launch(a, b)


def _fused_plan(a0, b0, a1, b1, a2, b2, a3, b3, n, key, dev, block):
    """Prepared launch handles for the fused kernel, built once per key."""
    kern = _maximum_fused4_kernel.warmup(
        a0,
        b0,
        a1,
        b1,
        a2,
        b2,
        a3,
        b3,
        n,
        grid=(1, 1, 1),
        BLOCK=block,
        I64=False,
        num_warps=_WARPS,
    )
    run = kern.run
    fn = kern.function
    meta = kern.packed_metadata
    direct = (
        getattr(run, "global_scratch_size", 1) == 0
        and getattr(run, "profile_scratch_size", 1) == 0
        and hasattr(run, "launch")
    )
    plan = (run.launch if direct else run, fn, meta, direct, dev)
    _FAST_PLAN[key] = plan
    return plan


def foreach_maximum_list_(self, other):
    global _FAST_OK
    n_pairs = len(self)
    if len(other) != n_pairs:
        raise RuntimeError(
            "_foreach_maximum_: self and other must have the same number of tensors"
        )
    if n_pairs == 0:
        return None

    if n_pairs == 4:
        a0 = self[0]
        b0 = other[0]
        if b0.shape == a0.shape and a0.is_contiguous() and b0.is_contiguous():
            shp = a0.shape
            dt = a0.dtype
            n = a0.numel()
            a1 = self[1]
            b1 = other[1]
            a2 = self[2]
            b2 = other[2]
            a3 = self[3]
            b3 = other[3]
            if (
                n != 0
                and a1.dtype == dt
                and a2.dtype == dt
                and a3.dtype == dt
                and b1.dtype == dt
                and b2.dtype == dt
                and b3.dtype == dt
                and a1.shape == shp
                and a2.shape == shp
                and a3.shape == shp
                and b1.shape == shp
                and b2.shape == shp
                and b3.shape == shp
                and a1.is_contiguous()
                and a2.is_contiguous()
                and a3.is_contiguous()
                and b1.is_contiguous()
                and b2.is_contiguous()
                and b3.is_contiguous()
            ):
                block = _BLOCK_OF.get(dt, 1024)
                gx = (n + block - 1) // block
                p0 = a0.data_ptr()
                q0 = b0.data_ptr()
                p1 = a1.data_ptr()
                q1 = b1.data_ptr()
                p2 = a2.data_ptr()
                q2 = b2.data_ptr()
                p3 = a3.data_ptr()
                q3 = b3.data_ptr()
                if (
                    _FAST_OK
                    and n % 16 == 0
                    and n <= _I64_LIMIT
                    and (p0 | q0 | p1 | q1 | p2 | q2 | p3 | q3) & 15 == 0
                ):
                    # Relaunch the already-compiled kernel straight through the
                    # metax C launcher with pre-extracted integer pointers: this
                    # skips JIT binding/cache-key/metadata work and the per-tensor
                    # data_ptr() calls the launcher would do.  The guards keep the
                    # compiled specialization valid (n divisible by 16, pointers
                    # 16-byte aligned, int32 offsets); any mismatch or failure
                    # falls back to the ordinary launch below.
                    dev = a0.device.index
                    key = (dt, dev, block)
                    plan = _FAST_PLAN.get(key)
                    if plan is None:
                        try:
                            plan = _fused_plan(
                                a0, b0, a1, b1, a2, b2, a3, b3, n, key, dev, block
                            )
                        except Exception:
                            _FAST_OK = False
                            plan = None
                    if plan is not None:
                        launch, fn, meta, direct, pdev = plan
                        try:
                            if direct:
                                launch(
                                    gx,
                                    4,
                                    1,
                                    _current_stream(pdev).cuda_stream,
                                    fn,
                                    None,
                                    None,
                                    meta,
                                    None,
                                    None,
                                    None,
                                    p0,
                                    q0,
                                    p1,
                                    q1,
                                    p2,
                                    q2,
                                    p3,
                                    q3,
                                    n,
                                    block,
                                    False,
                                )
                            else:
                                launch(
                                    gx,
                                    4,
                                    1,
                                    _current_stream(pdev).cuda_stream,
                                    fn,
                                    meta,
                                    None,
                                    None,
                                    None,
                                    p0,
                                    q0,
                                    p1,
                                    q1,
                                    p2,
                                    q2,
                                    p3,
                                    q3,
                                    n,
                                    block,
                                    False,
                                )
                            return None
                        except Exception:
                            _FAST_OK = False
                _maximum_fused4_kernel[(gx, 4)](
                    a0,
                    b0,
                    a1,
                    b1,
                    a2,
                    b2,
                    a3,
                    b3,
                    n,
                    BLOCK=block,
                    I64=n > _I64_LIMIT,
                    num_warps=_WARPS,
                )
                return None

    for a, b in zip(self, other):
        _pair(a, b)
    return None
