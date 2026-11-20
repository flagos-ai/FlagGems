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
"""Triton implementation of ``aten::_foreach_log10_``.

Applies ``log10`` in place to every tensor of a ``Tensor[]`` argument.  Each
element keeps its storage, ``data_ptr``, shape and strides, the list order is
preserved and the call returns ``None`` -- exactly the reference semantics.

Design:
  * dense tensors of the same (device, dtype) are handled by a *single* launch
    (up to 8 tensors per launch) whose pointers and element counts are plain
    kernel scalar arguments -- no host->device metadata staging, which is what
    dominates the cost of the reference executor on this target;
  * a flat task space (``sum_i cdiv(n_i, BLOCK)`` programs) is resolved per
    program against the cumulative block boundaries;
  * genuinely non-dense views (stepped slices) go through an
    unflatten/scatter kernel instead of a flat traversal;
  * launches go through a pre-bound ``CompiledKernel`` launcher: the standard
    ``kernel[grid](...)`` path re-binds the arguments, re-refines the
    specialization, rebuilds the MUSA pointer-alias spec and re-hashes the
    cache key on every call, which on this target costs more than the kernel
    itself for small tensors (measured 46us per 64x64 launch against ATen's
    22us for the whole list).
"""

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_unary import _foreach_log10_ as default__foreach_log10_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _needs_fallback(seq):
    """True when the list cannot use the MUSA kernels (device or dtype)."""
    for t in seq:
        if not isinstance(t, torch.Tensor):
            return True
        if t.device.type != "musa" or t.dtype not in _SUPPORTED_DTYPES:
            return True
    return False


_BLOCK = 1024
_NUM_WARPS = 4
_MAX_NT = 8

_ATEN_DTYPE_NAME = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
}


@triton.jit
def _log10_body(ptr, off, mask, IS_F64: tl.constexpr):
    x = tl.load(ptr + off, mask=mask)
    if IS_F64:
        y = tld.log10(x)
    else:
        y = tld.log10(x.to(tl.float32))
    tl.store(ptr + off, y.to(ptr.dtype.element_ty), mask=mask)


@triton.jit
def _log10_kernel(x_ptr, numel, IS_F64: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    _log10_body(x_ptr, offs, offs < numel, IS_F64)


@triton.jit
def _log10_multi(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    NT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    b1 = tl.cdiv(n0, BLOCK)
    b2 = b1 + tl.cdiv(n1, BLOCK)
    b3 = b2 + tl.cdiv(n2, BLOCK)
    b4 = b3 + tl.cdiv(n3, BLOCK)
    b5 = b4 + tl.cdiv(n4, BLOCK)
    b6 = b5 + tl.cdiv(n5, BLOCK)
    b7 = b6 + tl.cdiv(n6, BLOCK)

    ptr = p0
    local = pid
    n = n0
    if NT > 1:
        if pid >= b1:
            ptr = p1
            local = pid - b1
            n = n1
    if NT > 2:
        if pid >= b2:
            ptr = p2
            local = pid - b2
            n = n2
    if NT > 3:
        if pid >= b3:
            ptr = p3
            local = pid - b3
            n = n3
    if NT > 4:
        if pid >= b4:
            ptr = p4
            local = pid - b4
            n = n4
    if NT > 5:
        if pid >= b5:
            ptr = p5
            local = pid - b5
            n = n5
    if NT > 6:
        if pid >= b6:
            ptr = p6
            local = pid - b6
            n = n6
    if NT > 7:
        if pid >= b7:
            ptr = p7
            local = pid - b7
            n = n7

    offs = local * BLOCK + tl.arange(0, BLOCK)
    _log10_body(ptr, offs, offs < n, IS_F64)


@triton.jit
def _log10_strided(
    x_ptr,
    numel,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    st0,
    st1,
    st2,
    st3,
    st4,
    st5,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    rem = offs
    i5 = rem % sh5
    rem = rem // sh5
    i4 = rem % sh4
    rem = rem // sh4
    i3 = rem % sh3
    rem = rem // sh3
    i2 = rem % sh2
    rem = rem // sh2
    i1 = rem % sh1
    i0 = rem // sh1

    off = i0 * st0 + i1 * st1 + i2 * st2 + i3 * st3 + i4 * st4 + i5 * st5
    _log10_body(x_ptr, off, mask, IS_F64)


# ---------------------------------------------------------------------------
# pre-bound launcher
# ---------------------------------------------------------------------------
#
# ``JITFunction.run`` does, on every single call: hook dispatch, a device cache
# lookup, argument binding, ``refine_specialization``, the MUSA v3.1 pointer
# alias-spec construction and a cache-key hash -- and only then the launch.  For
# the tens-of-thousands-of-elements workloads that bookkeeping is the whole
# cost, so it is paid once per (kernel, dtype, shape-class) and cached here.
# ``CompiledKernel.run`` is the same callable entity Triton itself invokes from
# the tail of ``JITFunction.run``.

_launchers = {}
_no_fast = set()
_stream_fn = None


def _stream_of(device_index):
    global _stream_fn
    fn = _stream_fn
    if fn is None:

        fn = _stream_fn = _driver.active.get_current_stream
    return fn(device_index)


def _make_launcher(compiled):
    run_fn = compiled.run
    func = compiled.function
    pmeta = compiled.packed_metadata
    lmeta = compiled.launch_metadata
    from triton import knobs

    enter = knobs.runtime.launch_enter_hook
    exit_ = knobs.runtime.launch_exit_hook

    def launch(grid0, stream, args):
        grid = (grid0, 1, 1)
        run_fn(
            grid0,
            1,
            1,
            stream,
            func,
            pmeta,
            lmeta(grid, stream, *args),
            enter,
            exit_,
            *args,
        )

    return launch


def _launch(kernel, args, grid0, key, device_index):
    launcher = _launchers.get(key)
    if launcher is None:
        if key not in _no_fast:
            try:
                compiled = kernel.warmup(*args, grid=(1, 1, 1), num_warps=_NUM_WARPS)
                compiled._init_handles()
                launcher = _make_launcher(compiled)
            except Exception:
                _no_fast.add(key)
                launcher = None
        if launcher is None:
            kernel[(grid0,)](*args, num_warps=_NUM_WARPS)
            return
        _launchers[key] = launcher
    launcher(grid0, _stream_of(device_index), args)


def _spec(ptrs, sizes):
    """The argument classes Triton bakes into a compiled specialization."""
    return (
        tuple(p % 16 == 0 for p in ptrs),
        tuple((n % 16 == 0, n == 1) for n in sizes),
    )


def _is_flat_dense(t):
    """True when ``[data_ptr, data_ptr + numel)`` is exactly the tensor."""
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


def _launch_group(chunk, is_f64, device_index, dtype):
    """One launch for a list of at most ``_MAX_NT`` dense same-dtype tensors."""
    nt = len(chunk)
    numels = [t.numel() for t in chunk]
    dev_key = (device_index, dtype)

    if nt == 1:
        numel = numels[0]
        ptrs, sizes = (chunk[0].data_ptr(),), (numel,)
        _launch(
            _log10_kernel,
            (chunk[0], numel, is_f64, _BLOCK),
            (numel + _BLOCK - 1) // _BLOCK,
            (0, dev_key, is_f64, _spec(ptrs, sizes)),
            device_index,
        )
        return

    total = 0
    for s in numels:
        total += (s + _BLOCK - 1) // _BLOCK
    ptrs = list(chunk)
    first = chunk[0]
    while len(ptrs) < _MAX_NT:
        ptrs.append(first)
    sizes = numels + [0] * (_MAX_NT - nt)
    distinct = len({t.data_ptr() for t in chunk}) == nt
    _launch(
        _log10_multi,
        (
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            ptrs[4],
            ptrs[5],
            ptrs[6],
            ptrs[7],
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            sizes[5],
            sizes[6],
            sizes[7],
            nt,
            is_f64,
            _BLOCK,
        ),
        total,
        (
            1,
            dev_key,
            is_f64,
            nt,
            distinct,
            _spec([t.data_ptr() for t in ptrs], sizes),
        ),
        device_index,
    )


def _launch_strided(t, is_f64):
    if torch._debug_has_internal_overlap(t) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )
    dims = [(int(sz), int(st)) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    # Collapse dims that are contiguous with their inner neighbour.
    merged = []
    for sz, st in reversed(dims):
        if merged and merged[-1][1] * merged[-1][0] == st:
            merged[-1] = (merged[-1][0] * sz, merged[-1][1])
        else:
            merged.append((sz, st))
    merged.reverse()
    if len(merged) > 6:
        raise NotImplementedError(
            "_foreach_log10_: view with too many non-collapsible dimensions"
        )
    pairs = [(1, 0)] * (6 - len(merged)) + merged
    numel = t.numel()
    _log10_strided[(triton.cdiv(numel, _BLOCK),)](
        t,
        numel,
        pairs[0][0],
        pairs[1][0],
        pairs[2][0],
        pairs[3][0],
        pairs[4][0],
        pairs[5][0],
        pairs[0][1],
        pairs[1][1],
        pairs[2][1],
        pairs[3][1],
        pairs[4][1],
        pairs[5][1],
        is_f64,
        _BLOCK,
        num_warps=_NUM_WARPS,
    )


def _foreach_log10_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_LOG10_")
    _seq = self if isinstance(self, (list, tuple)) else [self]
    if _needs_fallback(_seq):
        return default__foreach_log10_(self)
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "_foreach_log10_(): argument 'self' (position 1) must be tuple of "
            f"Tensors, not {type(tensors).__name__}"
        )
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    keys = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                "expected Tensor as element 0 in argument 0, but got "
                f"{type(t).__name__}"
            )
        dtype = t.dtype
        if dtype.is_complex:
            raise NotImplementedError(
                "_foreach_log10_: complex tensors are not supported"
            )
        if not dtype.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type "
                f"{_ATEN_DTYPE_NAME.get(dtype, 'Float')}"
            )
        if t.numel() == 0:
            keys.append(None)
        elif _is_flat_dense(t):
            keys.append((t.device, dtype))
        else:
            keys.append(("strided", t.device, dtype))

    # Fast path: every element lands in the same dense (device, dtype) group,
    # which is what a benchmark list looks like.
    first_key = keys[0]
    if first_key is not None and first_key[0] != "strided":
        same = True
        for k in keys:
            if k != first_key:
                same = False
                break
        if same:
            device, dtype = first_key
            is_f64 = dtype is torch.float64
            total = len(tensors)
            for start in range(0, total, _MAX_NT):
                _launch_group(
                    tensors[start : start + _MAX_NT], is_f64, device.index, dtype
                )
            return None

    groups = {}
    strided = []
    for t, key in zip(tensors, keys):
        if key is None:
            continue
        if key[0] == "strided":
            strided.append(t)
            continue
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [t]
        else:
            bucket.append(t)

    for (device, dtype), bucket in groups.items():
        is_f64 = dtype is torch.float64
        for start in range(0, len(bucket), _MAX_NT):
            _launch_group(bucket[start : start + _MAX_NT], is_f64, device.index, dtype)
    for t in strided:
        _launch_strided(t, t.dtype is torch.float64)
    return None
