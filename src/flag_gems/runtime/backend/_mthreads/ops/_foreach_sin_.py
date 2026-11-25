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
"""Triton implementation of ``aten::_foreach_sin_`` (in-place multi-tensor sin).

Architecture
------------
Host side performs metadata-only work (dtype / aliasing validation, layout
classification, chunking).  Every element of the list is mutated in place:

* flat-addressable tensors (contiguous, or any non-overlapping-and-dense view)
  are processed by as few launches as possible: one launch for a single tensor,
  one 2-D launch (blocks x slots) for a chunk of up to 16 same-numel tensors,
  and a cumulative-bounds fused launch for a heterogeneous chunk.  Data
  pointers and element counts travel as kernel arguments, so no
  host-to-device metadata copy exists.
* gappy views use a rank-decomposed strided kernel, one launch per tensor.
* complex tensors are addressed as interleaved (re, im) real components.

Because a per-call Triton launch on this backend costs tens of microseconds
(more than ATen spends on the whole list), the flat paths reuse a compiled
kernel handle: the first call for a given specialization compiles through
``JITFunction.warmup`` and caches the ``CompiledKernel`` runner, which skips
per-argument binding on later calls.  The cached handle is keyed by everything
Triton specialises on (dtype, device, pointer alignment flags, scalar
divisibility/==1 flags, slot count); anything unusual falls back to the
ordinary JIT dispatch.

Flat addressing is element-wise correct for non-overlapping-and-dense views
because sin is applied in place: every element is read and written at the same
storage slot, so the logical index mapping is irrelevant.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

from flag_gems.ops._foreach_unary import _foreach_sin_ as default__foreach_sin_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_MAX_RANK = 8
_BLOCK = 1024
_NUM_WARPS = 4
_FUSED_SLOTS = 16

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}

_COMPLEX_TO_REAL = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

_INT_TYPE_NAMES = {
    torch.bool: "Bool",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
}

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)

try:
    from triton import knobs as _knobs

    _FAST_OK = True
except Exception:  # pragma: no cover - defensive
    _knobs = None
    _FAST_OK = False

_RUNNERS = {}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sin_block_body(
    p,
    n,
    start,
    pid,
    OUT_DT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Apply sin to one BLOCK-sized tile of the tensor based at ``p``."""
    idx = (pid - start).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    x = tl.load(p + idx, mask=mask, other=0.0)
    if IS_F64:
        y = tl.sin(x)
    else:
        y = tl.sin(x.to(tl.float32)).to(OUT_DT)
    tl.store(p + idx, y, mask=mask)


@triton.jit
def _sin_flat_kernel(
    p,
    n,
    OUT_DT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    x = tl.load(p + idx, mask=mask, other=0.0)
    if IS_F64:
        y = tl.sin(x)
    else:
        y = tl.sin(x.to(tl.float32)).to(OUT_DT)
    tl.store(p + idx, y, mask=mask)


@triton.jit
def _sin_flat_uniform_kernel(
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
    OUT_DT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Up to 16 same-numel flat tensors: grid = (blocks, len(list)).

    ``program_id(1)`` selects the tensor, so unused slots are simply never
    launched (no masked-out work) and every program has one base pointer.
    """
    slot = tl.program_id(1)
    pid = tl.program_id(0)
    if slot == 0:
        _sin_block_body(p0, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 1:
        _sin_block_body(p1, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 2:
        _sin_block_body(p2, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 3:
        _sin_block_body(p3, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 4:
        _sin_block_body(p4, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 5:
        _sin_block_body(p5, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 6:
        _sin_block_body(p6, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 7:
        _sin_block_body(p7, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 8:
        _sin_block_body(p8, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 9:
        _sin_block_body(p9, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 10:
        _sin_block_body(p10, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 11:
        _sin_block_body(p11, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 12:
        _sin_block_body(p12, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 13:
        _sin_block_body(p13, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif slot == 14:
        _sin_block_body(p14, n, 0, pid, OUT_DT, IS_F64, BLOCK)
    else:
        _sin_block_body(p15, n, 0, pid, OUT_DT, IS_F64, BLOCK)


@triton.jit
def _sin_flat_fused_kernel(
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
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,
    b9,
    b10,
    b11,
    b12,
    b13,
    b14,
    b15,
    OUT_DT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Heterogeneous chunk: 1-D grid, cumulative block bounds select the slot."""
    pid = tl.program_id(0)
    if pid < b1:
        _sin_block_body(p0, n0, 0, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b2:
        _sin_block_body(p1, n1, b1, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b3:
        _sin_block_body(p2, n2, b2, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b4:
        _sin_block_body(p3, n3, b3, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b5:
        _sin_block_body(p4, n4, b4, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b6:
        _sin_block_body(p5, n5, b5, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b7:
        _sin_block_body(p6, n6, b6, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b8:
        _sin_block_body(p7, n7, b7, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b9:
        _sin_block_body(p8, n8, b8, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b10:
        _sin_block_body(p9, n9, b9, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b11:
        _sin_block_body(p10, n10, b10, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b12:
        _sin_block_body(p11, n11, b11, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b13:
        _sin_block_body(p12, n12, b12, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b14:
        _sin_block_body(p13, n13, b13, pid, OUT_DT, IS_F64, BLOCK)
    elif pid < b15:
        _sin_block_body(p14, n14, b14, pid, OUT_DT, IS_F64, BLOCK)
    else:
        _sin_block_body(p15, n15, b15, pid, OUT_DT, IS_F64, BLOCK)


@triton.jit
def _strided_offsets(
    idx,
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
):
    """Row-major -> storage offset for a rank<=8 layout, innermost dim first."""
    rem = idx
    off = tl.zeros_like(idx)
    q = rem // d7
    off += (rem - q * d7) * s7
    rem = q
    q = rem // d6
    off += (rem - q * d6) * s6
    rem = q
    q = rem // d5
    off += (rem - q * d5) * s5
    rem = q
    q = rem // d4
    off += (rem - q * d4) * s4
    rem = q
    q = rem // d3
    off += (rem - q * d3) * s3
    rem = q
    q = rem // d2
    off += (rem - q * d2) * s2
    rem = q
    q = rem // d1
    off += (rem - q * d1) * s1
    rem = q
    q = rem // d0
    off += (rem - q * d0) * s0
    return off


@triton.jit
def _sin_strided_kernel(
    p,
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
    OUT_DT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    off = _strided_offsets(
        idx, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7
    )
    x = tl.load(p + off, mask=mask, other=0.0)
    if IS_F64:
        y = tl.sin(x)
    else:
        y = tl.sin(x.to(tl.float32)).to(OUT_DT)
    tl.store(p + off, y, mask=mask)


@triton.jit
def _sin_complex_kernel(
    p,
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
    R_DT: tl.constexpr,
    IS_F64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < n
    off = _strided_offsets(
        idx, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7
    )
    base = 2 * off
    re = tl.load(p + base, mask=mask, other=0.0)
    im = tl.load(p + base + 1, mask=mask, other=0.0)
    if IS_F64:
        a = re
        b = im
    else:
        a = re.to(tl.float32)
        b = im.to(tl.float32)
    sa = tl.sin(a)
    ca = tl.cos(a)
    eb = tl.exp(b)
    enb = tl.exp(-b)
    shb = (eb - enb) * 0.5
    chb = (eb + enb) * 0.5
    tl.store(p + base, (sa * chb).to(R_DT), mask=mask)
    tl.store(p + base + 1, (ca * shb).to(R_DT), mask=mask)


# ---------------------------------------------------------------------------
# cached compiled-kernel launch
# ---------------------------------------------------------------------------


def _make_runner(compiled):
    run = compiled.run
    function = compiled.function
    packed_metadata = compiled.packed_metadata
    launch_metadata = compiled.launch_metadata
    get_device = _driver.active.get_current_device
    get_stream = _driver.active.get_current_stream

    def runner(grid, args):
        device = get_device()
        stream = get_stream(device)
        meta = launch_metadata(grid, stream, *args)
        run(
            grid[0],
            grid[1],
            grid[2],
            stream,
            function,
            packed_metadata,
            meta,
            _knobs.runtime.launch_enter_hook,
            _knobs.runtime.launch_exit_hook,
            *args,
        )

    return runner


def _signature_ok(signature, pattern):
    """Check that the compiled launcher takes our exact argument list.

    ``pattern`` uses ``P`` for a pointer parameter, ``I`` for an integer
    parameter and ``C`` for a constexpr (which the launcher also receives).
    """
    values = list(signature.values())
    if len(values) != len(pattern):
        return False
    for value, kind in zip(values, pattern):
        text = str(value)
        is_pointer = text.startswith("*")
        is_constexpr = "constexpr" in text
        if kind == "P":
            if not is_pointer:
                return False
        elif kind == "I":
            if is_pointer or is_constexpr:
                return False
    return True


def _fast_launch(jit_fn, key, args, grid, pattern, num_warps):
    """Launch through a cached CompiledKernel when the specialization matches.

    ``args`` contains every kernel parameter in declaration order, including
    constexprs, exactly as ``JITFunction.run`` forwards them to the launcher.
    """
    global _FAST_OK
    if not _FAST_OK:
        return False
    runner = _RUNNERS.get(key)
    if runner is None:
        try:
            compiled = jit_fn.warmup(*args, grid=grid, num_warps=num_warps)
            if compiled is None:
                raise RuntimeError("warmup produced no kernel")
            if not _signature_ok(compiled.src.signature, pattern):
                raise RuntimeError("kernel signature mismatch")
            compiled._init_handles()
            runner = _make_runner(compiled)
            _RUNNERS[key] = runner
        except Exception:
            _RUNNERS.clear()
            _FAST_OK = False
            return False
    runner(grid, args)
    return True


# ---------------------------------------------------------------------------
# host-side metadata helpers
# ---------------------------------------------------------------------------


def _is_dense_view(t):
    """``is_non_overlapping_and_dense`` without a dispatcher round trip."""
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted((s, z) for s, z in zip(t.stride(), t.shape) if z != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _collapse(shape, stride):
    """Merge adjacent dims that form a contiguous run (outer -> inner order)."""
    dims = []
    for size, st in zip(shape, stride):
        if size == 1:
            continue
        if dims and dims[-1][1] == st * size:
            prev_size, _ = dims[-1]
            dims[-1] = (prev_size * size, st)
        else:
            dims.append((size, st))
    return dims


def _pad_rank(dims):
    dims = list(dims)
    rank = len(dims)
    if rank < _MAX_RANK:
        dims.extend([(1, 1)] * (_MAX_RANK - rank))
    return [d for d, _ in dims], [s for _, s in dims]


def _pointer_flags(tensors):
    """Alignment flags for the 16 slots plus whether the real slots are unique.

    Padded slots reuse ``tensors[0]``, so their flags repeat the first one; the
    aliasing Triton records for them is therefore fixed by ``len(tensors)``.
    """
    pointers = [t.data_ptr() for t in tensors]
    flags = tuple((p & 15) == 0 for p in pointers)
    flags += (flags[0],) * (_FUSED_SLOTS - len(pointers))
    return flags, len(set(pointers)) == len(pointers)


def _launch_single(t, n):
    dtype = t.dtype
    out_dt = _TL_DTYPE[dtype]
    is_f64 = dtype is torch.float64
    grid = ((n + _BLOCK - 1) // _BLOCK, 1, 1)
    key = (
        0,
        dtype,
        t.device,
        is_f64,
        (t.data_ptr() & 15) == 0,
        n & 15 == 0,
        n == 1,
    )
    args = (t, n, out_dt, is_f64, _BLOCK)
    if not _fast_launch(_sin_flat_kernel, key, args, grid, "PICCC", _NUM_WARPS):
        _sin_flat_kernel[grid](*args, num_warps=_NUM_WARPS)


def _launch_uniform(chunk):
    n = chunk[0].numel()
    dtype = chunk[0].dtype
    out_dt = _TL_DTYPE[dtype]
    is_f64 = dtype is torch.float64
    grid = ((n + _BLOCK - 1) // _BLOCK, len(chunk), 1)
    pointers = list(chunk) + [chunk[0]] * (_FUSED_SLOTS - len(chunk))
    args = (*pointers, n, out_dt, is_f64, _BLOCK)
    flags, unique = _pointer_flags(chunk)
    if not unique:
        # Duplicate user pointers make Triton's aliasing specialization
        # call-specific; use the ordinary dispatch for that rare case.
        _sin_flat_uniform_kernel[grid](*args, num_warps=_NUM_WARPS)
        return
    key = (
        1,
        dtype,
        chunk[0].device,
        is_f64,
        len(chunk),
        flags,
        n & 15 == 0,
        n == 1,
    )
    if not _fast_launch(
        _sin_flat_uniform_kernel,
        key,
        args,
        grid,
        "P" * _FUSED_SLOTS + "ICCC",
        _NUM_WARPS,
    ):
        _sin_flat_uniform_kernel[grid](*args, num_warps=_NUM_WARPS)


def _launch_hetero(chunk):
    numels = [t.numel() for t in chunk]
    pad = _FUSED_SLOTS - len(chunk)
    pointers = list(chunk) + [chunk[0]] * pad
    sizes = numels + [0] * pad
    bounds = []
    acc = 0
    for n in sizes[:-1]:
        acc += (n + _BLOCK - 1) // _BLOCK
        bounds.append(acc)
    total = acc + (sizes[-1] + _BLOCK - 1) // _BLOCK
    if total == 0:
        return
    dtype = chunk[0].dtype
    _sin_flat_fused_kernel[(total, 1, 1)](
        *pointers,
        *sizes,
        *bounds,
        OUT_DT=_TL_DTYPE[dtype],
        IS_F64=dtype is torch.float64,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_flat_group(tensors):
    count = len(tensors)
    if count == 1:
        t = tensors[0]
        _launch_single(t, t.numel())
        return
    for base in range(0, count, _FUSED_SLOTS):
        chunk = tensors[base : base + _FUSED_SLOTS]
        numels = [t.numel() for t in chunk]
        if min(numels) == max(numels):
            _launch_uniform(chunk)
        else:
            _launch_hetero(chunk)


def _launch_strided(t):
    n = t.numel()
    dims = _collapse(t.shape, t.stride())
    if len(dims) > _MAX_RANK:
        raise NotImplementedError(
            "foreach sin_ does not support this strided layout (rank > %d)" % _MAX_RANK
        )
    ds, ss = _pad_rank(dims)
    _sin_strided_kernel[((n + _BLOCK - 1) // _BLOCK, 1, 1)](
        t,
        n,
        *ds,
        *ss,
        OUT_DT=_TL_DTYPE[t.dtype],
        IS_F64=t.dtype is torch.float64,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_complex(t):
    n = t.numel()
    real_dt = _COMPLEX_TO_REAL[t.dtype]
    if t.dim() == 0:
        p = t.reshape(1).view(real_dt)
    else:
        p = t.view(real_dt)
    dims = _collapse(t.shape, t.stride())
    if len(dims) > _MAX_RANK:
        raise NotImplementedError(
            "foreach sin_ does not support this strided layout (rank > %d)" % _MAX_RANK
        )
    ds, ss = _pad_rank(dims)
    _sin_complex_kernel[((n + _BLOCK - 1) // _BLOCK, 1, 1)](
        p,
        n,
        *ds,
        *ss,
        R_DT=_TL_DTYPE[real_dt],
        IS_F64=real_dt is torch.float64,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _foreach_sin_(self):
    logger.debug("GEMS_MTHREADS _FOREACH_SIN_")
    if not isinstance(self, (list, tuple)) or any(
        not isinstance(t, torch.Tensor) or t.device.type != "musa" for t in self
    ):
        return default__foreach_sin_(self)
    tensors = self
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    flat_groups = {}
    strided = []
    complex_ts = []

    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        dtype = t.dtype
        if dtype not in _TL_DTYPE:
            if dtype in _COMPLEX_TO_REAL:
                if t.numel():
                    complex_ts.append(t)
                continue
            int_name = _INT_TYPE_NAMES.get(dtype)
            if int_name is not None:
                # In-place sin promotes to float, which cannot be cast back.
                raise RuntimeError(
                    "result type Float can't be cast to the desired output type "
                    + int_name
                )
            raise RuntimeError(
                "\"foreach\" not implemented for '%s'"
                % str(dtype).replace("torch.", "")
            )
        if t.numel() == 0:
            continue
        if t.is_contiguous():
            group_key = (dtype, t.device)
            group = flat_groups.get(group_key)
            if group is None:
                flat_groups[group_key] = [t]
            else:
                group.append(t)
            continue
        # Provable internal overlap (expanded / stride-0 dim) cannot be written
        # in place; ATen refuses it as well.
        for size, stride in zip(t.shape, t.stride()):
            if size > 1 and stride == 0:
                raise RuntimeError(_OVERLAP_MSG)
        if _is_dense_view(t):
            group_key = (dtype, t.device)
            group = flat_groups.get(group_key)
            if group is None:
                flat_groups[group_key] = [t]
            else:
                group.append(t)
        else:
            strided.append(t)

    for group in flat_groups.values():
        _launch_flat_group(group)
    for t in strided:
        _launch_strided(t)
    for t in complex_ts:
        _launch_complex(t)
    return None
