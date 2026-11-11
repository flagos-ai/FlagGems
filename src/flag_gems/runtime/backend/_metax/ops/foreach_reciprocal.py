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

"""Triton implementation of aten::_foreach_reciprocal.

Architecture: contiguous same-dtype tensors with equal numel are fused into a
single launch (grid axis 1 selects the tensor slot) to avoid paying Triton's
per-launch CPU cost once per tensor; a flat kernel handles single contiguous
tensors, a generic index-decomposition kernel handles strided layouts, and a
pair kernel handles complex inputs via a zero-copy dtype view.  Floating
inputs keep their dtype (fp32 opmath for fp16/bf16), integer/bool inputs are
promoted to the default floating dtype exactly like ATen, and empty tensors
are returned without a launch.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_MAX_NDIM = 8
_MAX_FUSED = 16
_MODE_BY_DTYPE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 0,
    torch.float64: 0,
}
# Bounded launch-configuration search, keyed on the element count and the
# arithmetic mode (so a dtype does not inherit another dtype's winner).  The
# per-thread tile shape is what separates the winners, and the 2-byte modes
# picked the top of the {1024..4096} range, so 8192 is searched as well.
# Autotune keys are per device, so a C500 run searches its own winners.
_FLAT_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=4),
    triton.Config({"BLOCK": 4096}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
    triton.Config({"BLOCK": 4096}, num_warps=8),
    triton.Config({"BLOCK": 8192}, num_warps=4),
    triton.Config({"BLOCK": 8192}, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_warps=8),
    triton.Config({"BLOCK": 16384}, num_warps=16),
]


@triton.jit
def _recip_value(x, MODE: tl.constexpr):
    # MODE 0: native dtype division (fp32/fp64).
    # MODE 1: fp32 opmath, IEEE division; the backend folds fp16 back to a
    #         native half division.
    # MODE 2: fp32 opmath with a non-IEEE (fast) division.  Used for bf16,
    #         whose explicit fp32 upcast does not fold, leaving the strict
    #         div.rn sequence on the critical path.
    if MODE == 0:
        return 1.0 / x
    elif MODE == 1:
        return 1.0 / x.to(tl.float32)
    else:
        return tl.fdiv(1.0, x.to(tl.float32), ieee_rounding=False)


@triton.autotune(configs=_FLAT_CONFIGS, key=["n_elements", "MODE"])
@triton.jit
def _reciprocal_flat_kernel(
    x_ptr, o_ptr, n_elements, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # Pure stream: no line is reused, so bypass L1 (".cg") and mark both
    # directions evict-first instead of letting them claim cache residency.
    x = tl.load(
        x_ptr + offs,
        mask=mask,
        other=1.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    r = _recip_value(x, MODE)
    tl.store(
        o_ptr + offs,
        r.to(o_ptr.dtype.element_ty),
        mask=mask,
        eviction_policy="evict_first",
    )


@triton.autotune(configs=_FLAT_CONFIGS, key=["n_elements", "MODE", "K"])
@triton.jit
def _reciprocal_fused_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    n_elements,
    MODE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    xs = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
    os_ = (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15)
    for i in tl.static_range(K):
        if slot == i:
            x = tl.load(
                xs[i] + offs,
                mask=mask,
                other=1.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )
            r = _recip_value(x, MODE)
            tl.store(
                os_[i] + offs,
                r.to(os_[i].dtype.element_ty),
                mask=mask,
                eviction_policy="evict_first",
            )


@triton.jit
def _reciprocal_strided_kernel(
    x_ptr,
    o_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    rem = offs
    xoff = tl.zeros((BLOCK,), dtype=tl.int32)
    ooff = tl.zeros((BLOCK,), dtype=tl.int32)

    i7 = rem % s7
    rem = rem // s7
    xoff += i7 * xs7
    ooff += i7 * os7
    i6 = rem % s6
    rem = rem // s6
    xoff += i6 * xs6
    ooff += i6 * os6
    i5 = rem % s5
    rem = rem // s5
    xoff += i5 * xs5
    ooff += i5 * os5
    i4 = rem % s4
    rem = rem // s4
    xoff += i4 * xs4
    ooff += i4 * os4
    i3 = rem % s3
    rem = rem // s3
    xoff += i3 * xs3
    ooff += i3 * os3
    i2 = rem % s2
    rem = rem // s2
    xoff += i2 * xs2
    ooff += i2 * os2
    i1 = rem % s1
    rem = rem // s1
    xoff += i1 * xs1
    ooff += i1 * os1
    i0 = rem % s0
    xoff += i0 * xs0
    ooff += i0 * os0

    x = tl.load(x_ptr + xoff, mask=mask, other=1.0)
    r = _recip_value(x, MODE)
    tl.store(o_ptr + ooff, r.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _reciprocal_complex_kernel(x_ptr, o_ptr, n_pairs, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_pairs
    off = 2 * idx
    a = tl.load(x_ptr + off, mask=mask, other=1.0)
    b = tl.load(x_ptr + off + 1, mask=mask, other=0.0)
    den = a * a + b * b
    ra = a / den
    rb = -b / den
    tl.store(o_ptr + off, ra, mask=mask)
    tl.store(o_ptr + off + 1, rb, mask=mask)


def _padded_meta(t, o):
    sizes = []
    xst = []
    ost = []
    for d in range(min(t.dim(), _MAX_NDIM)):
        sizes.append(t.shape[d])
        xst.append(t.stride(d))
        ost.append(o.stride(d))
    pad = _MAX_NDIM - len(sizes)
    sizes = [1] * pad + sizes
    xst = [0] * pad + xst
    ost = [0] * pad + ost
    return sizes + xst + ost


def _launch_flat(x, o, n, mode):
    _reciprocal_flat_kernel[lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)](
        x, o, n, mode
    )


def _launch_fused(xs, os_, n, mode, k):
    pad = _MAX_FUSED - k
    _reciprocal_fused_kernel[
        lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"], k)
    ](
        *(list(xs) + [xs[0]] * pad),
        *(list(os_) + [os_[0]] * pad),
        n,
        mode,
        k,
    )


def _launch_strided(x, o, n, mode):
    meta = _padded_meta(x, o)
    grid = (triton.cdiv(n, _BLOCK),)
    _reciprocal_strided_kernel[grid](x, o, n, *meta, mode, _BLOCK, num_warps=4)


def _launch_complex(xr, orr, n_pairs):
    grid = (triton.cdiv(n_pairs, _BLOCK),)
    _reciprocal_complex_kernel[grid](xr, orr, n_pairs, _BLOCK, num_warps=4)


def foreach_reciprocal(self):
    tensors = list(self)
    total = len(tensors)
    if total == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")

    results = [None] * total
    i = 0
    while i < total:
        t = tensors[i]
        dt = t.dtype
        if dt.is_complex:
            real_dtype = torch.float32 if dt == torch.complex64 else torch.float64
            out = torch.empty_like(t)
            results[i] = out
            n = t.numel()
            i += 1
            if n == 0:
                continue
            if t.is_contiguous() and out.is_contiguous():
                _launch_complex(t.view(real_dtype), out.view(real_dtype), n)
            else:
                raise RuntimeError(
                    "foreach_reciprocal: unsupported non-contiguous complex layout"
                )
        elif dt.is_floating_point:
            out = torch.empty_like(t)
            results[i] = out
            n = t.numel()
            if n == 0:
                i += 1
                continue
            upcast = _MODE_BY_DTYPE.get(dt, 1)
            if t.is_contiguous() and out.is_contiguous():
                # Fuse consecutive same-dtype, equal-numel, contiguous tensors
                # into one launch: grid axis 1 selects the tensor slot.
                gx = [t]
                go = [out]
                j = i + 1
                while j < total and len(gx) < _MAX_FUSED:
                    t2 = tensors[j]
                    if t2.dtype != dt or t2.numel() != n:
                        break
                    if not t2.is_contiguous():
                        break
                    o2 = torch.empty_like(t2)
                    results[j] = o2
                    gx.append(t2)
                    go.append(o2)
                    j += 1
                if j - i > 1:
                    _launch_fused(gx, go, n, upcast, j - i)
                else:
                    _launch_flat(t, out, n, upcast)
                i = j
            else:
                _launch_strided(t, out, n, upcast)
                i += 1
        else:
            out = torch.empty(t.shape, dtype=torch.get_default_dtype(), device=t.device)
            results[i] = out
            n = t.numel()
            i += 1
            if n == 0:
                continue
            if t.is_contiguous() and out.is_contiguous():
                _launch_flat(t, out, n, 1)
            else:
                _launch_strided(t, out, n, 1)
    return results
