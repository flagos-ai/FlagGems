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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fractional_max_pool3d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_out,
    out_per_nc,
    in_dhw,
    BLOCK: tl.constexpr,
):
    """One lane per output position, non-atomic scatter (Kunlunxin/XPU).

    Fast path, valid whenever ``alpha >= k`` in every dim (``in - k >= k *
    (out - 1)``, or ``out == 1``).  The pool window of output ``o`` is
    ``[start(o), start(o) + k)`` with ``start(o) = trunc((o + s) * alpha) -
    trunc(s * alpha)``; consecutive starts satisfy ``start(o + 1) - start(o)
    >= floor(alpha) >= k`` (``start(out - 1)`` is clamped to ``in - k`` but
    ``(out - 1) * alpha == in - k`` up to sub-ulp fp32 rounding, so the final
    gap is ``>= k`` as well).  Hence the k-wide windows are pairwise disjoint,
    every input position belongs to at most one window, each output's argmax
    index is distinct, and the plain (non-atomic) stores can never race --
    exact, deterministic, O(n_out).  This is the same pattern as the proven
    ``fractional_max_pool2d`` scatter fast path (``tl.atomic_add`` scatter is
    not lowerable on this backend at all: the generic atomic kernel fails at
    ``ConvertTritonXPUToLLVM`` with ``atomicSim = true``).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_out
    # ``n_out`` is not always a multiple of ``BLOCK``, so tail lanes would read
    # past ``grad_output``/``indices``; out-of-bounds accesses are a
    # device-wedge hazard on this backend, hence predicate the loads too.
    idx = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.int32)
    val = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    nc = offsets // out_per_nc
    tl.store(
        grad_input_ptr + nc * in_dhw + idx,
        val.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _fractional_max_pool3d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    n_elems,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    inv_alpha_d,
    inv_alpha_h,
    inv_alpha_w,
    k_d: tl.constexpr,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather-based fractional max pool 3d backward (Kunlunxin/XPU).

    General fallback (used when ``alpha < k`` in some dim, i.e. overlapping
    windows where the scatter fast path is not race-free).  One lane per input
    position (flat 1-D over n, c, d, h, w).  Every output's argmax is looked up
    in the per-plane flat index ``d * in_hw + h * in_w + w`` produced by the
    Gems forward and the upstream gradient is accumulated in registers wherever
    it equals this lane's position, then written with one masked store.  This
    is the exact, deterministic, race-free pattern of the proven
    ``fractional_max_pool2d`` gather kernel (``tl.atomic_add`` scatter is not
    lowerable on this backend).

    The candidate output box is bounded *without* knowing the random samples.
    With ``start(o) = trunc((o + s) * alpha) - trunc(s * alpha)`` we have
    ``o*alpha - 1 <= start(o) <= o*alpha + 1``, and ``start(out - 1)`` is the
    clamped ``in - k``; hence every output whose k-window may contain input
    position ``x`` satisfies ``(x - k) / alpha <= o <= (x + 1) / alpha``.
    ``inv_alpha`` is passed host-side as ``1 / alpha`` (0 when ``out == 1`` or
    ``in == k``, where the only candidate is ``o = 0``), the ``+-2`` margin
    absorbs fp32 rounding, and each candidate is verified exactly against the
    stored indices, so the result is identical to scanning every output (even
    across overlapping windows, multiple outputs can map to the same input
    position and their gradients all accumulate).  ``MAX_*`` is the
    host-computed tight bound ``(k + 1) / alpha + 6`` (see the host function);
    the scan range is rolled (not ``tl.static_range``): the XPU unroll control
    pass fails when the per-program body is fully unrolled.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    in_hw = in_h * in_w
    in_dhw = in_d * in_hw
    nc = safe_offsets // in_dhw
    rem = safe_offsets % in_dhw
    d = rem // in_hw
    rem2 = rem % in_hw
    h = rem2 // in_w
    w = rem2 % in_w
    my_flat = d * in_hw + h * in_w + w

    od_lo = tl.maximum(
        0, (tl.maximum(d - k_d, 0).to(tl.float32) * inv_alpha_d).to(tl.int32) - 2
    )
    od_hi = tl.minimum(
        out_d - 1, ((d + 1).to(tl.float32) * inv_alpha_d).to(tl.int32) + 2
    )
    oh_lo = tl.maximum(
        0, (tl.maximum(h - k_h, 0).to(tl.float32) * inv_alpha_h).to(tl.int32) - 2
    )
    oh_hi = tl.minimum(
        out_h - 1, ((h + 1).to(tl.float32) * inv_alpha_h).to(tl.int32) + 2
    )
    ow_lo = tl.maximum(
        0, (tl.maximum(w - k_w, 0).to(tl.float32) * inv_alpha_w).to(tl.int32) - 2
    )
    ow_hi = tl.minimum(
        out_w - 1, ((w + 1).to(tl.float32) * inv_alpha_w).to(tl.int32) + 2
    )

    out_hw = out_h * out_w
    out_dhw = out_d * out_hw
    gop = grad_output_ptr + nc * out_dhw
    iop = indices_ptr + nc * out_dhw

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for od in range(0, MAX_D):
        o_d = od_lo + od
        d_ok = (o_d <= od_hi) & (o_d < out_d)
        c_d = tl.minimum(o_d, out_d - 1)
        for oh in range(0, MAX_H):
            o_h = oh_lo + oh
            h_ok = (o_h <= oh_hi) & (o_h < out_h)
            c_h = tl.minimum(o_h, out_h - 1)
            for ow in range(0, MAX_W):
                o_w = ow_lo + ow
                w_ok = (o_w <= ow_hi) & (o_w < out_w)
                c_w = tl.minimum(o_w, out_w - 1)
                o_off = c_d * out_hw + c_h * out_w + c_w
                idx = tl.load(iop + o_off).to(tl.int32)
                val = tl.load(gop + o_off).to(tl.float32)
                active = mask & d_ok & h_ok & w_ok
                acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def _parse_size(value):
    if isinstance(value, int):
        return value, value, value
    return value[0], value[1], value[2]


def fractional_max_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    output_size,
    indices,
):
    """Backward pass for fractional_max_pool3d (Kunlunxin/XPU).

    ``tl.atomic_add`` (atomicSim) cannot lower on the XPU Triton backend
    (``ConvertTritonXPUToLLVM``), so the generic KernelGen implementation is
    unusable here; this specialization uses the two proven non-atomic
    patterns of the backend ``fractional_max_pool2d``: a plain scatter store
    when the pool windows are pairwise disjoint (``alpha >= k`` in every dim),
    otherwise a gather kernel with one lane per input position that
    accumulates the upstream gradients of all candidate output windows.
    """
    logger.debug("GEMS_KUNLUNXIN FRACTIONAL_MAX_POOL3D_BACKWARD")
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    in_n, in_c, in_d, in_h, in_w = input.shape
    kernel_d, kernel_h, kernel_w = _parse_size(kernel_size)
    out_d, out_h, out_w = _parse_size(output_size)

    if input.numel() == 0 or grad_output.numel() == 0:
        return torch.zeros_like(input)

    out_per_nc = out_d * out_h * out_w
    in_dhw = in_d * in_h * in_w
    n_out = grad_output.numel()

    scatter_d = (out_d == 1) or (in_d - kernel_d >= kernel_d * (out_d - 1))
    scatter_h = (out_h == 1) or (in_h - kernel_h >= kernel_h * (out_h - 1))
    scatter_w = (out_w == 1) or (in_w - kernel_w >= kernel_w * (out_w - 1))

    with torch_device_fn.device(input.device):
        if scatter_d and scatter_h and scatter_w:
            grad_input = torch.zeros_like(input)
            _fractional_max_pool3d_backward_scatter_kernel[(triton.cdiv(n_out, 256),)](
                grad_output,
                indices,
                grad_input,
                n_out,
                out_per_nc,
                in_dhw,
                BLOCK=256,
                num_warps=4,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        else:
            inv_alpha_d = (out_d - 1) / (in_d - kernel_d) if in_d > kernel_d else 0.0
            inv_alpha_h = (out_h - 1) / (in_h - kernel_h) if in_h > kernel_h else 0.0
            inv_alpha_w = (out_w - 1) / (in_w - kernel_w) if in_w > kernel_w else 0.0
            max_d = max(
                1,
                min(out_d, int(math.ceil((kernel_d + 1) * inv_alpha_d)) + 6),
            )
            max_h = max(
                1,
                min(out_h, int(math.ceil((kernel_h + 1) * inv_alpha_h)) + 6),
            )
            max_w = max(
                1,
                min(out_w, int(math.ceil((kernel_w + 1) * inv_alpha_w)) + 6),
            )
            n_elems = input.numel()
            grad_input = torch.empty_like(input)
            _fractional_max_pool3d_backward_gather_kernel[(triton.cdiv(n_elems, 128),)](
                grad_output,
                indices,
                grad_input,
                n_elems,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                inv_alpha_d,
                inv_alpha_h,
                inv_alpha_w,
                kernel_d,
                kernel_h,
                kernel_w,
                MAX_D=max_d,
                MAX_H=max_h,
                MAX_W=max_w,
                BLOCK=128,
                num_warps=2,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
    return grad_input
