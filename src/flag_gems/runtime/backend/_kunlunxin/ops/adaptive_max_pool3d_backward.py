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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _adaptive_max_pool3d_backward_recompute_indices_kernel(
    input_ptr,
    index_out_ptr,
    n_elems,  # in_n * in_c * out_d * out_h * out_w
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    WIN_D: tl.constexpr,
    WIN_H: tl.constexpr,
    WIN_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Recompute the adaptive-max-pool3d argmax indices (forward semantics).

    One lane per output position.  Each adaptive window
    [start, end) = [floor(o * in / out), ceil((o + 1) * in / out)) is scanned
    in row-major (d, h, w) order with the same tie-break as the ATen reference
    (first tap attaining the strict max; NaNs win over any value).  The
    recomputed indices are what the CPU reference tolerates: on this XPU stack
    the vendor ``adaptive_max_pool3d`` forward returns uninitialized index
    memory, so a backward whose output depended on the caller-supplied
    ``indices`` would either fault or disagree with the reference.

    Windows are provably non-empty and all loads below stay in the (n, c)
    plane: candidate positions are clamped to the window start when the
    predicate is dropped (the XPU backend treats compound i1 masked loads as a
    slow path and ``other=`` is unreliable there).
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    out_per_nc = out_d * out_h * out_w
    nc = safe_offsets // out_per_nc
    rem = safe_offsets % out_per_nc
    od = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w

    # Adaptive window: start = floor(o * in / out), end = ceil((o + 1) * in / out).
    d_start = (od * in_d) // out_d
    d_end = tl.minimum(((od + 1) * in_d + out_d - 1) // out_d, in_d)
    h_start = (oh * in_h) // out_h
    h_end = tl.minimum(((oh + 1) * in_h + out_h - 1) // out_h, in_h)
    w_start = (ow * in_w) // out_w
    w_end = tl.minimum(((ow + 1) * in_w + out_w - 1) // out_w, in_w)

    plane_base = input_ptr + nc * (in_d * in_h * in_w)
    in_hw = in_h * in_w

    acc_val = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    acc_idx = tl.full((BLOCK,), -1, dtype=tl.int32)

    # Rolled loops (compile-time constant bounds) instead of tl.static_range:
    # the XPU unroll control pass (TritonXPUUnrollControl) fails with uni_sram
    # OOR when the window scan is fully unrolled for large ratios
    # (e.g. in=8/out=1 gives 9^3 = 729 bodies).
    for kd in range(0, WIN_D):
        d = d_start + kd
        d_ok = d < d_end
        d_safe = tl.where(d_ok, d, d_start)
        for kh in range(0, WIN_H):
            h = h_start + kh
            h_ok = h < h_end
            h_safe = tl.where(h_ok, h, h_start)
            for kw in range(0, WIN_W):
                w = w_start + kw
                w_ok = w < w_end
                w_safe = tl.where(w_ok, w, w_start)
                value = tl.load(
                    plane_base + d_safe * in_hw + h_safe * in_w + w_safe
                ).to(tl.float32)
                active = d_ok & h_ok & w_ok
                is_new = active & (
                    (value > acc_val) | (value != value) | (acc_idx < 0)
                )
                acc_val = tl.where(is_new, value, acc_val)
                acc_idx = tl.where(is_new, d * in_hw + h * in_w + w, acc_idx)

    tl.store(index_out_ptr + offsets, acc_idx, mask=mask)


@libentry()
@triton.jit
def _adaptive_max_pool3d_backward_gather_kernel(
    grad_output_ptr,
    indices_ptr,  # recomputed int32 indices, layout (n, c, out_d, out_h, out_w)
    grad_input_ptr,
    n_elems,  # in_n * in_c * in_d * in_h * in_w
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    MAX_D: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather-based adaptive max pool 3d backward (Kunlunxin/XPU).

    One lane per input position.  The output positions whose adaptive window
    may contain this input element form the small box
    ``[d_min, d_max) x [h_min, h_max) x [w_min, w_max)`` with
    ``d_min = floor(d * out / in)``, ``d_max = ceil((d + 1) * out / in)``.
    For each candidate we load its (recomputed) argmax index and the upstream
    gradient, and accumulate the gradient whose index equals this position.

    This is the same exact, deterministic, race-free pattern as the
    Kunlunxin ``max_pool3d_backward``: ``tl.atomic_add`` scatter loses updates
    on this backend (~1e-5 per op, seed-dependent), so every output element's
    contribution is accumulated in registers and written with a single
    masked store.  The candidate box is bounded by the constexpr
    ``MAX_* = (out + in - 1) // in + 1``, and candidate addresses are clamped
    so the (unmasked) loads can never go out of the (n, c) plane.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    safe_offsets = tl.where(mask, offsets, 0)

    in_hw = in_h * in_w
    in_spatial = in_d * in_hw
    nc = safe_offsets // in_spatial
    rem = safe_offsets % in_spatial
    d = rem // in_hw
    rem2 = rem % in_hw
    h = rem2 // in_w
    w = rem2 % in_w

    my_flat = d * in_hw + h * in_w + w

    d_min = (d * out_d) // in_d
    d_max = tl.minimum(((d + 1) * out_d + in_d - 1) // in_d, out_d)
    h_min = (h * out_h) // in_h
    h_max = tl.minimum(((h + 1) * out_h + in_h - 1) // in_h, out_h)
    w_min = (w * out_w) // in_w
    w_max = tl.minimum(((w + 1) * out_w + in_w - 1) // in_w, out_w)

    out_per_nc = out_d * out_h * out_w
    gop = grad_output_ptr + nc * out_per_nc
    iop = indices_ptr + nc * out_per_nc

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Small static bounds (MAX_* <= 2 whenever out <= in, the only legal
    # adaptive-pool configuration): fully unrolled bodies let the compiler
    # issue all candidate loads up front (ILP), matching the proven
    # Kunlunxin ``max_pool3d_backward_flat_kernel`` pattern.  The recompute
    # kernel above stays rolled because its window bound can reach 9^3 bodies.
    for od in tl.static_range(0, MAX_D):
        o_d = d_min + od
        d_ok = o_d < d_max
        c_d = tl.minimum(o_d, out_d - 1)
        for oh in tl.static_range(0, MAX_H):
            o_h = h_min + oh
            h_ok = o_h < h_max
            c_h = tl.minimum(o_h, out_h - 1)
            for ow in tl.static_range(0, MAX_W):
                o_w = w_min + ow
                w_ok = o_w < w_max
                active = mask & d_ok & h_ok & w_ok
                c_w = tl.minimum(o_w, out_w - 1)
                o_off = c_d * (out_h * out_w) + c_h * out_w + c_w
                idx = tl.load(iop + o_off).to(tl.int32)
                val = tl.load(gop + o_off).to(tl.float32)
                acc += tl.where(active & (idx == my_flat), val, 0.0)

    tl.store(
        grad_input_ptr + offsets,
        acc.to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


def adaptive_max_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
):
    """Gradient of adaptive_max_pool3d (Kunlunxin/XPU implementation).

    Recomputes the forward argmax indices inside the (n, c) planes (see the
    recompute kernel) and then accumulates the gradient with the deterministic
    gather pattern.  The caller-supplied ``indices`` argument is accepted for
    ATen schema compatibility but not trusted: on this XPU stack the vendor
    ``adaptive_max_pool3d`` forward writes uninitialized index memory, so a
    backward honoring those indices would either fault on out-of-bounds
    addresses or disagree with the reference.  When the caller's indices come
    from a correct forward, the recomputed indices coincide with them (same
    window math, same tie-break) and the result is identical.
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL3D_BACKWARD")

    grad_output = grad_output.contiguous()
    self = self.contiguous()
    in_n, in_c, in_d, in_h, in_w = self.shape
    out_d, out_h, out_w = grad_output.shape[-3:]

    grad_input = torch.empty_like(self)

    n_in = grad_input.numel()
    if n_in == 0 or grad_output.numel() == 0:
        return grad_input

    n_out = in_n * in_c * out_d * out_h * out_w
    # Exact upper bounds for the per-dim iteration counts (see kernel
    # comments).  For the adaptive windows with r = in / out >= 1 the
    # recompute-scan length is exactly r when r is an integer and at most
    # floor(r) + 2 otherwise, so the naive ``(in + out - 1) // out + 1``
    # bound spends (3/2)^3 = 3.4x of the loop bodies on the common
    # integer-ratio shapes (in = 2 * out); the gather candidate count is 1
    # whenever out divides in, and at most 2 otherwise.
    win_d = in_d // out_d + (0 if in_d % out_d == 0 else 2)
    win_h = in_h // out_h + (0 if in_h % out_h == 0 else 2)
    win_w = in_w // out_w + (0 if in_w % out_w == 0 else 2)
    max_d = 1 if in_d % out_d == 0 else (out_d // in_d + 2)
    max_h = 1 if in_h % out_h == 0 else (out_h // in_h + 2)
    max_w = 1 if in_w % out_w == 0 else (out_w // in_w + 2)

    indices_tmp = torch.empty((n_out,), dtype=torch.int32, device=self.device)

    # Two launch shapes tuned on Kunlunxin XL (2026-09-05), re-measured shape
    # by shape against the enabled-vendor baseline:
    # - recompute scan: 64-lane tiles / num_warps=1 keep the rolled window
    #   scan within the TritonXPUUnrollControl vrf-budget / uni_sram limits
    #   for every ratio (BLOCK=128 with a WIN=8 scan overruns uni_sram);
    #   when the window is at most 2 per dim (the common out = in / 2
    #   shapes) the scan fits 128-lane tiles / num_warps=2 as well.
    # - gather: the candidate box is at most 2 elements per dim and fully
    #   statically unrolled, so 128-lane tiles / num_warps=2 fit and roughly
    #   halve its latency.  BLOCK=256 / num_warps=4 overrun uni_sram even on
    #   these small bodies; do not raise without re-measuring every shape.
    recompute_block = 128 if max(win_d, win_h, win_w) <= 2 else 64
    recompute_warps = 2 if recompute_block == 128 else 1
    with torch_device_fn.device(self.device):
        _adaptive_max_pool3d_backward_recompute_indices_kernel[(triton.cdiv(n_out, recompute_block),)](
            self,
            indices_tmp,
            n_out,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            WIN_D=win_d,
            WIN_H=win_h,
            WIN_W=win_w,
            BLOCK=recompute_block,
            num_warps=recompute_warps,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )
        _adaptive_max_pool3d_backward_gather_kernel[(triton.cdiv(n_in, 128),)](
            grad_output,
            indices_tmp,
            grad_input,
            n_in,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            MAX_D=max_d,
            MAX_H=max_h,
            MAX_W=max_w,
            BLOCK=128,
            num_warps=2,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return grad_input