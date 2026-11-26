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

"""In-place foreach true division: aten::_foreach_div_.List.

foreach_div_list_(self, other) mutates every tensor of ``self`` in place by dividing it by the
paired element of ``other`` (a Tensor, a Number, or a per-position sequence),
and returns ``None`` -- exactly what the in-place ATen overload does.

Numerics follow ATen's CUDA ``div_true``: the division runs in opmath
(float32 for fp16/bf16/fp32, float64 when the operand is fp64) with IEEE
round-to-nearest, and the single rounding back into the destination dtype
happens on the store.

Launch strategy: an in-place foreach list of contiguous tensors that all have
the same element count is served by ONE kernel launch in which every program
applies all pairs at the same offset (a lane touches pair 0..L-1 at that
offset). That keeps one launch for the whole list without paying the
per-segment index arithmetic a concatenated index space would need. For such a
launch-bound call the compiled kernel captured for the same specialization is
invoked directly, skipping the per-call JIT dispatch, whenever every operand
keeps the alignment and element-count class it was compiled for. Everything
else (noncontiguous operands, broadcast or scalar divisors, mixed element
counts, other arities, mismatched classes) falls back to one lean launch per
pair.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_MAX_RANK = 8

_OVERLAP_MSG = (
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation."
)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def _pair_body(a_ptr, b_ptr, off, mask):
    """``a[off] = a[off] / b[off]`` with ATen's opmath for one pair."""
    if a_ptr.dtype.element_ty == tl.float64:
        x = tl.load(a_ptr + off, mask=mask, other=0.0)
        y = tl.load(b_ptr + off, mask=mask, other=1.0)
        tl.store(a_ptr + off, x / y, mask=mask)
    else:
        x = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(b_ptr + off, mask=mask, other=1.0).to(tl.float32)
        tl.store(a_ptr + off, (x / y).to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _div_uniform2(a0, b0, a1, b1, n, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    _pair_body(a0, b0, off, mask)
    _pair_body(a1, b1, off, mask)


@triton.jit
def _div_uniform3(a0, b0, a1, b1, a2, b2, n, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    _pair_body(a0, b0, off, mask)
    _pair_body(a1, b1, off, mask)
    _pair_body(a2, b2, off, mask)


@triton.jit
def _div_uniform4(a0, b0, a1, b1, a2, b2, a3, b3, n, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    _pair_body(a0, b0, off, mask)
    _pair_body(a1, b1, off, mask)
    _pair_body(a2, b2, off, mask)
    _pair_body(a3, b3, off, mask)


_UNIFORM = {2: _div_uniform2, 3: _div_uniform3, 4: _div_uniform4}


@triton.jit
def _div_uniform4_short(a0, b0, a1, b1, a2, b2, a3, b3, n, BLOCK: tl.constexpr):
    """Short-list variant: every load is issued before any store.

    Same math as ``_div_uniform4``, but the four pairs' memory round trips
    overlap instead of serializing behind each other. Only worth it (and only
    used) below the size where the call is latency- rather than
    bandwidth-bound; it keeps more values live, so it is not used for long
    lists.
    """
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x0 = tl.load(a0 + off, mask=mask, other=0.0)
    y0 = tl.load(b0 + off, mask=mask, other=1.0)
    x1 = tl.load(a1 + off, mask=mask, other=0.0)
    y1 = tl.load(b1 + off, mask=mask, other=1.0)
    x2 = tl.load(a2 + off, mask=mask, other=0.0)
    y2 = tl.load(b2 + off, mask=mask, other=1.0)
    x3 = tl.load(a3 + off, mask=mask, other=0.0)
    y3 = tl.load(b3 + off, mask=mask, other=1.0)
    if a0.dtype.element_ty == tl.float64:
        tl.store(a0 + off, x0 / y0, mask=mask)
        tl.store(a1 + off, x1 / y1, mask=mask)
        tl.store(a2 + off, x2 / y2, mask=mask)
        tl.store(a3 + off, x3 / y3, mask=mask)
    else:
        tl.store(
            a0 + off,
            (x0.to(tl.float32) / y0.to(tl.float32)).to(a0.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            a1 + off,
            (x1.to(tl.float32) / y1.to(tl.float32)).to(a1.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            a2 + off,
            (x2.to(tl.float32) / y2.to(tl.float32)).to(a2.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            a3 + off,
            (x3.to(tl.float32) / y3.to(tl.float32)).to(a3.dtype.element_ty),
            mask=mask,
        )


_SHORT = {4: _div_uniform4_short}
_SHORT_MAX = 1 << 16


@triton.jit
def _div_flat_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    _pair_body(a_ptr, b_ptr, off, mask)


@triton.jit
def _div_scalar_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    if a_ptr.dtype.element_ty == tl.float64:
        y = tl.load(b_ptr).to(tl.float64)
        x = tl.load(a_ptr + off, mask=mask, other=0.0)
        tl.store(a_ptr + off, x / y, mask=mask)
    else:
        y = tl.load(b_ptr).to(tl.float32)
        x = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(a_ptr + off, (x / y).to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _div_num_kernel(a_ptr, y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    if a_ptr.dtype.element_ty == tl.float64:
        d = y.to(tl.float64)
        x = tl.load(a_ptr + off, mask=mask, other=0.0)
        tl.store(a_ptr + off, x / d, mask=mask)
    else:
        d = y.to(tl.float32)
        x = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(a_ptr + off, (x / d).to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _unravel(
    idx,
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
    NDIM: tl.constexpr,
):
    off = idx * 0
    if NDIM > 0:
        q = idx // s0
        off += (idx - q * s0) * t0
        idx = q
    if NDIM > 1:
        q = idx // s1
        off += (idx - q * s1) * t1
        idx = q
    if NDIM > 2:
        q = idx // s2
        off += (idx - q * s2) * t2
        idx = q
    if NDIM > 3:
        q = idx // s3
        off += (idx - q * s3) * t3
        idx = q
    if NDIM > 4:
        q = idx // s4
        off += (idx - q * s4) * t4
        idx = q
    if NDIM > 5:
        q = idx // s5
        off += (idx - q * s5) * t5
        idx = q
    if NDIM > 6:
        q = idx // s6
        off += (idx - q * s6) * t6
        idx = q
    if NDIM > 7:
        q = idx // s7
        off += (idx - q * s7) * t7
    return off


@triton.jit
def _div_strided_kernel(
    a_ptr,
    b_ptr,
    n,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    aoff = _unravel(
        idx, s0, s1, s2, s3, s4, s5, s6, s7, a0, a1, a2, a3, a4, a5, a6, a7, NDIM
    )
    boff = _unravel(
        idx, s0, s1, s2, s3, s4, s5, s6, s7, b0, b1, b2, b3, b4, b5, b6, b7, NDIM
    )
    if a_ptr.dtype.element_ty == tl.float64:
        x = tl.load(a_ptr + aoff, mask=mask, other=0.0)
        y = tl.load(b_ptr + boff, mask=mask, other=1.0)
        tl.store(a_ptr + aoff, x / y, mask=mask)
    else:
        x = tl.load(a_ptr + aoff, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(b_ptr + boff, mask=mask, other=1.0).to(tl.float32)
        tl.store(a_ptr + aoff, (x / y).to(a_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "").capitalize()


def _check_pair_types(a, b):
    """Validate ATen's promotion rules for one in-place pair."""
    dtype = a.dtype
    if dtype.is_complex:
        raise NotImplementedError("foreach div has no complex kernel")
    if not dtype.is_floating_point:
        out_dtype = torch.float32
    elif isinstance(b, torch.Tensor):
        out_dtype = dtype if b.dtype is dtype else torch.result_type(a, b)
    else:
        out_dtype = dtype if type(b) is float else torch.result_type(a, b)
    if out_dtype != dtype:
        raise RuntimeError(
            f"result type {_dtype_name(out_dtype)} can't be cast to the "
            f"desired output type {_dtype_name(dtype)}"
        )


def _aligned(t, rank):
    """Shape/strides of ``t`` right-aligned into a rank-dim index space."""
    shape = [1] * rank
    stride = [0] * rank
    off = rank - t.dim()
    ts = t.shape
    tst = t.stride()
    for d in range(t.dim()):
        shape[off + d] = ts[d]
        stride[off + d] = tst[d]
    return shape, stride


def _launch_strided(a, b, rank, shape, astride, bstride):
    shape = list(shape) + [1] * (_MAX_RANK - rank)
    astride = list(astride) + [0] * (_MAX_RANK - rank)
    bstride = list(bstride) + [0] * (_MAX_RANK - rank)
    n = a.numel()
    _div_strided_kernel[(triton.cdiv(n, _BLOCK),)](
        a,
        b,
        n,
        *shape,
        *astride,
        *bstride,
        NDIM=rank,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )


def _launch_num(a, value):
    n = a.numel()
    _div_num_kernel[(triton.cdiv(n, _BLOCK),)](
        a, float(value), n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
    )


def _pair_strided(a, b):
    if not isinstance(b, torch.Tensor):
        _launch_num(a, b)
        return
    rank = max(a.dim(), b.dim())
    if rank > _MAX_RANK:
        raise NotImplementedError(
            f"foreach div supports tensors of rank <= {_MAX_RANK}"
        )
    ashape, astride = _aligned(a, rank)
    bshape, bstride = _aligned(b, rank)
    bnumel = b.numel()
    for d in range(rank):
        if bnumel == 1:
            bstride[d] = 0
            continue
        bs = bshape[d]
        if bs == ashape[d]:
            continue
        if bs == 1:
            bstride[d] = 0
        else:
            raise RuntimeError(
                f"The size of tensor a ({ashape[d]}) must match the size of "
                f"tensor b ({bs}) at non-singleton dimension "
                f"{d - (rank - a.dim())}"
            )
    _launch_strided(a, b, rank, ashape, astride, bstride)


def _pair(a, b):
    n = a.numel()
    if n == 0:
        return
    _check_pair_types(a, b)
    if a.is_contiguous():
        if not isinstance(b, torch.Tensor):
            _launch_num(a, b)
        elif b.numel() == 1:
            _div_scalar_kernel[(triton.cdiv(n, _BLOCK),)](
                a, b, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
        elif b.numel() == n and b.is_contiguous():
            _div_flat_kernel[(triton.cdiv(n, _BLOCK),)](
                a, b, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
            )
        else:
            _pair_strided(a, b)
    else:
        if torch._debug_has_internal_overlap(a) == 1:
            raise RuntimeError(_OVERLAP_MSG)
        _pair_strided(a, b)


_TL_TYPES = {
    torch.float16: "*fp16",
    torch.bfloat16: "*bf16",
    torch.float32: "*fp32",
    torch.float64: "*fp64",
}

# (jit function, device, dtype, n specialization) -> (compiled kernel, driver,
# triton knobs) once a launch has been captured, or _NORMAL when the direct
# launch is unavailable for that specialization.
_LAUNCHERS = {}
_NORMAL = object()


def _all_aligned(args):
    """Triton's only value-dependent pointer specialization is 16B alignment."""
    for i in range(len(args) - 1):
        if args[i].data_ptr() & 15:
            return False
    return True


def _capture_launcher(kernel, device, dtype, arity, cache, before):
    """Identify the compiled kernel the launch just made produced.

    Accepted only when its recorded signature is exactly the one this path will
    relaunch (same arity, same pointer dtype, ``n`` int32, one constexpr), so a
    captured handle can never be reused for a different specialization.
    """
    tstr = _TL_TYPES.get(dtype)
    if tstr is None:
        return None
    want = {"n": "i32", "BLOCK": "constexpr"}
    for i in range(arity):
        want["a%d" % i] = tstr
        want["b%d" % i] = tstr

    new_keys = [k for k in list(cache.keys()) if k not in before]
    if len(new_keys) == 1:
        candidates = [cache[new_keys[0]]]
    else:
        candidates = list(cache.values())
    found = None
    for compiled in candidates:
        try:
            if hasattr(compiled, "result"):
                compiled = compiled.result()
            src = getattr(compiled, "src", None)
            if getattr(src, "signature", None) != want:
                continue
            if not (
                hasattr(compiled, "foreach_div_list_")
                and hasattr(compiled, "function")
                and hasattr(compiled, "packed_metadata")
                and hasattr(compiled, "launch_metadata")
            ):
                continue
        except Exception:
            return None
        if found is not None:
            return None
        found = compiled
    if found is None:
        return None
    try:
        from triton.runtime.driver import driver

        knobs = getattr(type(kernel), "foreach_div_list_").__globals__["knobs"]
    except Exception:
        return None
    return found, driver.active, knobs


def _launch_uniform(kernel, dtype, arity, args, n):
    """Launch the homogeneous list, bypassing the per-call JIT dispatch.

    The generic path re-derives the specialization key from every argument on
    every call, which dominates a call whose kernel only runs for a few
    microseconds. Once the compiled kernel for this exact specialization is
    known, the same launcher is invoked directly. The fast path is taken only
    while every operand stays 16-byte aligned and ``n`` keeps the divisibility
    class it had when the handle was captured -- the two value-dependent
    specializations Triton applies here -- and any unexpected failure falls back
    to the ordinary launch.
    """
    grid = (n + _BLOCK - 1) // _BLOCK
    try:
        device = torch.cuda.current_device()
        key = (kernel, device, dtype, (n & 15) == 0, n == 1)
    except Exception:
        device = None
        key = None

    if key is not None and key in _LAUNCHERS:
        captured = _LAUNCHERS[key]
        if captured is not _NORMAL and _all_aligned(args):
            handle, drv, knobs = captured
            try:
                stream = drv.get_current_stream(device)
                meta = handle.launch_metadata((grid, 1, 1), stream, *args)
                handle.run(
                    grid,
                    1,
                    1,
                    stream,
                    handle.function,
                    handle.packed_metadata,
                    meta,
                    knobs.runtime.launch_enter_hook,
                    knobs.runtime.launch_exit_hook,
                    *args,
                    _BLOCK,
                )
                return
            except Exception:
                _LAUNCHERS[key] = _NORMAL
        kernel[(grid,)](*args, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
        return

    cache = None
    before = None
    if key is not None:
        try:
            cache = kernel.device_caches[device][0]
            before = set(cache.keys())
        except Exception:
            cache = None
    kernel[(grid,)](*args, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    if key is not None and key not in _LAUNCHERS:
        captured = None
        if cache is not None and _all_aligned(args):
            try:
                captured = _capture_launcher(
                    kernel, device, dtype, arity, cache, before
                )
            except Exception:
                captured = None
        _LAUNCHERS[key] = _NORMAL if captured is None else captured


def foreach_div_list_(self, other):
    if not isinstance(self, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    n_self = len(self)
    if n_self == 0:
        return None

    if not isinstance(other, (list, tuple)):
        for i in range(n_self):
            a = self[i]
            if not isinstance(a, torch.Tensor):
                raise TypeError("argument 'self' must be tuple of Tensors")
            _pair(a, other)
        return None

    if len(other) != n_self:
        raise RuntimeError(
            "Tensor list must have same number of elements as tensor list, "
            f"got {n_self} and {len(other)}"
        )

    # Lean homogeneous path: identical floating dtype in every pair makes the
    # promotion check a no-op, and one shared element count plus contiguity
    # makes one same-offset launch over the list valid. Anything else goes
    # through the fully validating per-pair path below.
    kernel = _UNIFORM.get(n_self)
    if kernel is not None:
        try:
            a0 = self[0]
            b0 = other[0]
            dtype = a0.dtype
            n = a0.numel()
            if (
                n
                and dtype.is_floating_point
                and a0.is_contiguous()
                and b0.dtype is dtype
                and b0.numel() == n
                and b0.is_contiguous()
            ):
                for i in range(1, n_self):
                    a = self[i]
                    b = other[i]
                    if not (
                        a.dtype is dtype
                        and b.dtype is dtype
                        and a.numel() == n
                        and b.numel() == n
                        and a.is_contiguous()
                        and b.is_contiguous()
                    ):
                        break
                else:
                    if n <= _SHORT_MAX:
                        kernel = _SHORT.get(n_self, kernel)
                    args = []
                    add = args.append
                    for i in range(n_self):
                        add(self[i])
                        add(other[i])
                    args.append(n)
                    _launch_uniform(kernel, dtype, n_self, args, n)
                    return None
        except (AttributeError, IndexError, TypeError):
            pass

    for i in range(n_self):
        a = self[i]
        if not isinstance(a, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
        _pair(a, other[i])
    return None
