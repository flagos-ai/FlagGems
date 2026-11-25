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
"""In-place foreach subtraction with a per-tensor scalar (ScalarList overload).

Semantics (matches ATen ``_foreach_sub_.ScalarList``): for every i,
``self[i] -= scalars[i]`` in place, storage/stride preserving, returns None.
Floating inputs are computed in float32 opmath (ATen's gpu_kernel_with_scalars
behaviour) and stored back with the storage dtype; integral inputs stay in the
integral path.

Dispatch keeps the per-call host cost minimal (the benchmark times a single
call on an idle device, so Python launch overhead is on the critical path):

* contiguous float lists of 2..4 pairwise-disjoint tensors whose largest tensor
  is small are served by one fused launch, grid = (blocks, list_len, 1);
* anything larger uses one launch per tensor with the plain 1-D kernel;
* non-contiguous views use a rank<=5 divmod strided kernel.

Every launch goes through a cached ``CompiledKernel`` runner (built once per
(dtype, grid, numel, alignment) signature via ``JITFunction.warmup``), which
skips Triton's per-call argument binding, specialization refinement and cache
key computation.  Reused runners are keyed on every input property that can
change the compiled specialization, so a different dtype, numel, grid or
pointer-alignment class always compiles/uses its own kernel.
"""

import logging
import numbers

import torch
import triton
import triton.language as tl
from triton import knobs

from flag_gems.ops._foreach_binary import (
    _foreach_sub__ScalarList as default__foreach_sub__ScalarList,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_BLOCK = 1024
_WARPS = 4
_FUSE_MAX_NUMEL = 262144

_ITEMSIZE = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}

# 16 bytes per thread: 8 half elements or 4 float elements per lane.
_WARPS_DTYPE = {torch.float16: 4, torch.bfloat16: 4, torch.float32: 8}

_RUNNERS = {}


@triton.jit
def _sub_flat_f32(x_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
    y = x.to(tl.float32) - s
    tl.store(x_ptr + offs, y, mask=mask, eviction_policy="evict_first")


@triton.jit
def _sub_flat_int(x_ptr, n, s, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = x.to(tl.int64) - s
    tl.store(x_ptr + offs, y, mask=mask)


@triton.jit
def _body_f32(ptr, n, s, offs):
    mask = offs < n
    x = tl.load(ptr + offs, mask=mask, other=0.0)
    tl.store(ptr + offs, x.to(tl.float32) - s, mask=mask)


@triton.jit
def _sub_fused2(p0, p1, n0, n1, s0, s1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _body_f32(p0, n0, s0, offs)
    if tid == 1:
        _body_f32(p1, n1, s1, offs)


@triton.jit
def _sub_fused3(p0, p1, p2, n0, n1, n2, s0, s1, s2, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _body_f32(p0, n0, s0, offs)
    if tid == 1:
        _body_f32(p1, n1, s1, offs)
    if tid == 2:
        _body_f32(p2, n2, s2, offs)


@triton.jit
def _sub_fused4(p0, p1, p2, p3, n0, n1, n2, n3, s0, s1, s2, s3, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if tid == 0:
        _body_f32(p0, n0, s0, offs)
    if tid == 1:
        _body_f32(p1, n1, s1, offs)
    if tid == 2:
        _body_f32(p2, n2, s2, offs)
    if tid == 3:
        _body_f32(p3, n3, s3, offs)


@triton.jit
def _sub_strided(
    x_ptr,
    n,
    s,
    sz0,
    sz1,
    sz2,
    sz3,
    sz4,
    st0,
    st1,
    st2,
    st3,
    st4,
    RANK: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        q = idx // sz0
        acc += (idx - q * sz0) * st0
        idx = q
    if RANK > 1:
        q = idx // sz1
        acc += (idx - q * sz1) * st1
        idx = q
    if RANK > 2:
        q = idx // sz2
        acc += (idx - q * sz2) * st2
        idx = q
    if RANK > 3:
        q = idx // sz3
        acc += (idx - q * sz3) * st3
        idx = q
    if RANK > 4:
        q = idx // sz4
        acc += (idx - q * sz4) * st4
        idx = q
    x = tl.load(x_ptr + acc, mask=mask, other=0)
    if IS_FLOAT:
        y = x.to(tl.float32) - s
    else:
        y = x.to(tl.int64) - s
    tl.store(x_ptr + acc, y, mask=mask)


def _collapse_dims(shape, strides):
    """Collapse adjacent dims whose strides compose into one linear dim."""
    dims = []
    for sz, st in zip(shape, strides):
        sz = int(sz)
        st = int(st)
        if sz == 1:
            continue
        if sz == 0:
            return []
        if dims and dims[-1][1] == st * sz:
            dims[-1] = (dims[-1][0] * sz, st)
        else:
            dims.append((sz, st))
    return dims


def _stream():
    return torch.musa.current_stream().musa_stream


def _hooks_quiet():
    """True when no launch hook is installed on either chain (cheap check)."""
    for hc in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
        if hc is None:
            continue
        calls = getattr(hc, "calls", None)
        if calls is None or len(calls):
            return False
    return True


def _make_runner(jit_fn, grid, key, sample, warps=_WARPS):
    """Compile once per signature and cache the argument-pre-bound launcher."""
    kernel = jit_fn.warmup(*sample, BLOCK=_BLOCK, num_warps=warps, grid=grid)
    launcher = kernel.run
    raw = getattr(launcher, "launch", None)
    if (
        raw is not None
        and not getattr(launcher, "_debug_enabled", True)
        and getattr(launcher, "global_scratch_size", 1) == 0
    ):
        fn = kernel.function
        pm = kernel.packed_metadata
        g0, g1, g2 = grid
        fallback = kernel[grid]

        def runner(*args):
            # Skips the runner's device/stream lookup, the per-call LazyDict
            # launch metadata and the launcher's python wrapper.
            if _hooks_quiet():
                raw(g0, g1, g2, _stream(), fn, pm, None, None, None, *args)
            else:
                fallback(*args)

        _RUNNERS[key] = runner
        return runner
    runner = kernel[grid]
    _RUNNERS[key] = runner
    return runner


def _try_fused(tensors, scalars):
    """One fused launch for small contiguous float lists; True when handled."""
    n = len(tensors)
    dtype = tensors[0].dtype
    isize = _ITEMSIZE.get(dtype)
    if isize is None:
        return False
    ptrs = []
    nums = []
    max_n = 0
    for t in tensors:
        if t.dtype is not dtype or not t.is_contiguous():
            return False
        m = t.numel()
        if m == 0 or m > _FUSE_MAX_NUMEL:
            return False
        if m > max_n:
            max_n = m
        ptrs.append(t.data_ptr())
        nums.append(m)
    unaligned = ptrs[0] & 15
    for i in range(1, n):
        unaligned |= ptrs[i] & 15
    for i in range(n):
        a = ptrs[i]
        b = a + nums[i] * isize
        for j in range(i + 1, n):
            c = ptrs[j]
            if a < c + nums[j] * isize and c < b:
                return False
    grid_x = (max_n + _BLOCK - 1) // _BLOCK
    key = (n, dtype, grid_x, tuple(nums), unaligned)
    runner = _RUNNERS.get(key)
    fs = [float(s) for s in scalars]
    if runner is None:
        grid = (grid_x, n, 1)
        jit_fn = _sub_fused4 if n == 4 else (_sub_fused3 if n == 3 else _sub_fused2)
        runner = _make_runner(
            jit_fn, grid, key, tuple(tensors) + tuple(nums) + tuple(fs)
        )
    runner(*tensors, *nums, *fs, _BLOCK)
    return True


def _launch_one(t, s):
    n = t.numel()
    if t.is_contiguous():
        grid = ((n + _BLOCK - 1) // _BLOCK, 1, 1)
        if t.is_floating_point():
            f = float(s)
            key = (1, t.dtype, grid, n, t.data_ptr() & 15)
            runner = _RUNNERS.get(key)
            if runner is None:
                runner = _make_runner(
                    _sub_flat_f32,
                    grid,
                    key,
                    (t, n, f),
                    _WARPS_DTYPE.get(t.dtype, _WARPS),
                )
            runner(t, n, f, _BLOCK)
            return
        _sub_flat_int[grid](t, n, int(s), BLOCK=_BLOCK, num_warps=_WARPS)
        return
    _launch_strided(t, s, n, ((n + _BLOCK - 1) // _BLOCK,))


def _launch_strided(t, s, n, grid):
    dims = _collapse_dims(t.shape, t.stride())
    if not dims:
        dims = [(1, 0)]
    if len(dims) > 5:
        raise RuntimeError("_foreach_sub__ScalarList: unsupported tensor rank")
    sizes = [1, 1, 1, 1, 1]
    sts = [0, 0, 0, 0, 0]
    for k, (sz, st) in enumerate(dims):
        sizes[k] = sz
        sts[k] = st
    is_float = t.is_floating_point()
    _sub_strided[grid](
        t,
        n,
        float(s) if is_float else int(s),
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sts[0],
        sts[1],
        sts[2],
        sts[3],
        sts[4],
        RANK=len(dims),
        IS_FLOAT=is_float,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _scalar_value(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, complex):
        raise TypeError("_foreach_sub__ScalarList: complex scalars are not supported")
    if isinstance(s, bool):
        return int(s)
    if not isinstance(s, numbers.Number):
        raise TypeError("_foreach_sub__ScalarList: scalars must be numbers")
    return s


def _foreach_sub__ScalarList(self, scalars, *args, **kwargs):
    logger.debug("GEMS_MTHREADS _FOREACH_SUB__SCALARLIST")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_sub__ScalarList(self, scalars)
    tensors = self
    n = len(tensors)
    if n == 0:
        return None

    if isinstance(scalars, (list, tuple)):
        slist = scalars
        if len(slist) != n:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )
    elif isinstance(scalars, torch.Tensor) or isinstance(scalars, numbers.Number):
        slist = [scalars] * n
    else:
        slist = list(scalars)
        if len(slist) != n:
            raise RuntimeError(
                "Tensor list must have same number of elements as scalar list."
            )

    # Fuse in chunks of four: on this target a Python-level launch costs far
    # more than the tiny kernel, so a sixteen-entry list goes out as four
    # launches instead of sixteen.  A chunk that cannot be fused (mixed dtype,
    # gapped view, overlapping spans) falls back to the per-tensor path.
    i = 0
    while i < n:
        chunk = tensors[i : i + 4]
        if len(chunk) > 1 and _try_fused(chunk, slist[i : i + 4]):
            i += len(chunk)
            continue
        t = tensors[i]
        if t.numel() == 0:
            i += 1
            continue
        if t.is_contiguous() and t.is_floating_point():
            _launch_one(t, slist[i])
        else:
            _launch_one(t, _scalar_value(slist[i]))
        i += 1
    return None
