# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MTHREADS avg_pool2d implementations.

The generic FlagGems backward kernel maps one program to an input tile and
enumerates every KxK tap.  That is particularly expensive on MTT because the
receiver test contains vector integer modulo/division.  The kernels here use
the inverse pooling relation instead: an input element directly computes the
small output interval that can produce it.

All fast-path kernels bake the tensor geometry (in_h/in_w/out_h/out_w) into
``tl.constexpr`` parameters.  The launch C ABI parses one PyArg slot per
signature entry, so every runtime int argument costs a PyLong parse on each
steady-state call while constexpr entries parse as a plain 'O' incref;
folding the geometry removes the per-call parse AND lets the steady-state
launcher drop the arguments entirely (they ride the pre-packed constexpr
tail, and the triton disk cache persists the per-geometry binaries).  The
tuning keys still name in_h/in_w/out_h/out_w, which now arrive as keyword
arguments, so autotuning granularity is unchanged.

The fast path only serves contiguous fp32 operands (dispatch gates on the
cold path and the steady-state hot path re-verifies contiguity per call), so
the kernels address with the canonical contiguous strides
``in_stride_h == in_w, in_stride_w == 1, out_stride_h == out_w,
out_stride_w == 1`` and do not take stride arguments; the n/c-level strides
were never referenced (``pid_nc * out_h * out_w`` base addressing already
assumes contiguity).
"""

import logging

# Triton is optional in some backend environments; keep this import block
# stable across isort's installed-module classification.
# isort: off
import torch
import torch_musa._MUSAC as _musac
import triton
import triton.language as tl
from triton import knobs as _triton_knobs
from flag_gems.ops.avg_pool2d import _parse_pool_params, pool2d_output_size
from flag_gems.ops.avg_pool2d import avg_pool2d as _generic_avg_pool2d
from flag_gems.ops.avg_pool2d import avg_pool2d_backward as _generic_avg_pool2d_backward
from flag_gems.utils import libentry, libtuner

# isort: on

logger = logging.getLogger(__name__)

# --- Steady-state launch helpers (V6 pure-Triton route, no C++ extension) ---
# ``_get_raw_musa_stream`` queries the current stream of a device index
# directly through the torch_musa C extension -- the same primitive
# torch_musa's dynamo device interface imports -- measured at ~0.15us vs
# ~3.9us for the Stream-object chain (driver.get_current_stream ->
# torch.musa.current_stream -> Stream.__new__ -> lazy_init).
_get_raw_musa_stream = _musac._musa_getCurrentRawStream
# ``_get_raw_musa_device`` (~0.17us) mirrors driver.get_current_device()
# for the per-call current-device guard (the MUSA launch API requires the
# stream to belong to the current context, so a current-device mismatch
# must enter the tensor device's context before launching).
_get_raw_musa_device = _musac._musa_getDevice
# Context manager for the rare current-device-mismatch branch: the launch
# must run from the operands' device context, with that device's current
# stream (matching what a native aten op's DeviceGuard would do).
_device_guard = torch.musa.device
# Hoisted singleton for the per-call launch-hook check: attribute reads on
# the cached knobset reflect later assignments because knobs.runtime is a
# module-level singleton in triton.knobs.
_RUNTIME_KNOBS = _triton_knobs.runtime
# Cached dtype singleton for the hot-path gate (LOAD_GLOBAL instead of
# LOAD_GLOBAL torch + LOAD_ATTR float32).
_FLOAT32 = torch.float32
# Forward fast path additionally serves the half-precision dtypes that the
# official benchmark exercises (fp16/bf16 accumulate in fp32, like ATen).
_FLOAT16 = torch.float16
_BFLOAT16 = torch.bfloat16
# Unbound constructor (one attribute load at import instead of a per-call
# bound-method creation on the hot path).
_TENSOR_NEW_EMPTY = torch.Tensor.new_empty


@triton.jit
def _ceil_pool_inv(o, in_len):
    # count_include_pad=True + ceil tail: the divisor counts the explicit
    # padding but NOT the ceil-induced shortening, i.e. per axis
    #   pool(o) = clamp(min(2o+2, in_len+1) - (2o-1), 0, ...)  in {1,2,3}
    # (pool == 3 everywhere except the shortened ceil tail rows/cols).
    # The select-chain reciprocal avoids FP division; substituting
    # g * (1/3, 1/6, 1/9) for g / (3, 6, 9) costs <= ~2 ulp, verified
    # against the torch CPU autograd reference at atol=rtol=1e-5 (fp32).
    start = o * 2 - 1
    p = tl.minimum(start + 3, in_len + 1) - start
    p = tl.maximum(p, 0)
    return tl.where(p == 1, 1.0, tl.where(p == 2, 0.5, 1.0 / 3.0))


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=[
        "in_h",
        "in_w",
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
    ],
    # Persistent ConfigCache (sqlite) so autotuning results survive across
    # processes: only the very first process per shape ever benchmarks;
    # every later process reuses the recorded config, keeping first-call
    # latency at a single binary load.  Without this, the first call of
    # every new process re-benched all configs (~1-3 s), which poisoned the
    # official benchmark's 5-call calibration and collapsed n_rep to 1.
    # warmup/rep are milliseconds (do_bench): 25/100 ms windows keep the
    # p50 robust on small kernels -- 5/5 ms windows only fit a handful of
    # samples per config and were observed picking a 60% slower block shape
    # for the 7x7 class on a noise spike (the first write wins forever).
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_direct_range_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    CEIL_TAIL_H: tl.constexpr,
    CEIL_TAIL_W: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h2 = h[:, None]
    w2 = w[None, :]
    in_mask = (h2 < in_h) & (w2 < in_w)

    # oh_min = ceil((h + p - (k - 1)) / s), oh_max=floor((h+p)/s).
    # The numerator of oh_min can be negative, so use a sign-safe ceil-div.
    h_min_num = h2 + padding_h - (kernel_h - 1)
    w_min_num = w2 + padding_w - (kernel_w - 1)
    h_min = tl.where(
        h_min_num >= 0,
        (h_min_num + stride_h - 1) // stride_h,
        -((-h_min_num) // stride_h),
    )
    w_min = tl.where(
        w_min_num >= 0,
        (w_min_num + stride_w - 1) // stride_w,
        -((-w_min_num) // stride_w),
    )
    h_max = (h2 + padding_h) // stride_h
    w_max = (w2 + padding_w) // stride_w

    # At most ceil(K/S) producers exist in each dimension.  All loops are
    # compile-time unrolled and masked by the exact direct range.
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    go_base = grad_output_ptr + pid_nc * out_h * out_w
    for dh in tl.static_range(0, (kernel_h + stride_h - 1) // stride_h):
        oh = h_min + dh
        oh_mask = (oh >= h_min) & (oh <= h_max) & (oh >= 0) & (oh < out_h)
        if DIVISOR_OVERRIDE != 0:
            divisor_h = tl.full((BLOCK_H, BLOCK_W), float(DIVISOR_OVERRIDE), tl.float32)
        elif COUNT_INCLUDE_PAD:
            if CEIL_TAIL_H:
                h_start = oh * stride_h - padding_h
                pool_h = tl.minimum(h_start + kernel_h, in_h + padding_h) - h_start
                pool_h = tl.maximum(pool_h, 0)
                # PyTorch counts explicit padding, but not the shortened tail
                # introduced solely by ceil_mode.
                divisor_h = tl.where(
                    h_start + kernel_h > in_h + padding_h, pool_h, kernel_h
                ).to(tl.float32)
            else:
                divisor_h = tl.full((BLOCK_H, BLOCK_W), float(kernel_h), tl.float32)
        else:
            h_start = oh * stride_h - padding_h
            valid_h = tl.minimum(h_start + kernel_h, in_h) - tl.maximum(h_start, 0)
            valid_h = tl.maximum(valid_h, 0)
            divisor_h = valid_h.to(tl.float32)

        for dw in tl.static_range(0, (kernel_w + stride_w - 1) // stride_w):
            ow = w_min + dw
            ow_mask = (ow >= w_min) & (ow <= w_max) & (ow >= 0) & (ow < out_w)
            producer_mask = in_mask & oh_mask & ow_mask
            if DIVISOR_OVERRIDE != 0:
                divisor = divisor_h
            elif COUNT_INCLUDE_PAD:
                if CEIL_TAIL_W:
                    w_start = ow * stride_w - padding_w
                    pool_w = tl.minimum(w_start + kernel_w, in_w + padding_w) - w_start
                    pool_w = tl.maximum(pool_w, 0)
                    divisor = divisor_h * tl.where(
                        w_start + kernel_w > in_w + padding_w, pool_w, kernel_w
                    ).to(tl.float32)
                else:
                    divisor = divisor_h * kernel_w
            else:
                w_start = ow * stride_w - padding_w
                valid_w = tl.minimum(w_start + kernel_w, in_w) - tl.maximum(w_start, 0)
                valid_w = tl.maximum(valid_w, 0)
                divisor = divisor_h * valid_w.to(tl.float32)
            divisor = tl.where(divisor == 0, 1.0, divisor)
            go_ptr = go_base + oh * out_w + ow
            go = tl.load(go_ptr, mask=producer_mask, other=0.0).to(tl.float32)
            grad_acc += tl.where(producer_mask, go / divisor, 0.0)

    gi_base = grad_input_ptr + pid_nc * in_h * in_w
    gi_ptr = gi_base + h2 * in_w + w2
    tl.store(gi_ptr, grad_acc, mask=in_mask)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
    # Persistent ConfigCache: see direct_range kernel note above.
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_k3s2p1_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    CEIL_TAIL_H: tl.constexpr,
    CEIL_TAIL_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h2 = h[:, None]
    w2 = w[None, :]
    in_mask = (h2 < in_h) & (w2 < in_w)

    # For K=3,S=2,P=1, the inverse range is exactly [h//2,(h+1)//2]
    # (and analogously for W).  This removes both modulo and general divide.
    oh0 = h2 >> 1
    oh1 = (h2 + 1) >> 1
    ow0 = w2 >> 1
    ow1 = (w2 + 1) >> 1
    go_base = grad_output_ptr + pid_nc * out_h * out_w
    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    m00 = in_mask & (oh0 < out_h) & (ow0 < out_w)
    m01 = in_mask & (oh0 < out_h) & (ow1 < out_w) & (ow1 != ow0)
    m10 = in_mask & (oh1 < out_h) & (ow0 < out_w) & (oh1 != oh0)
    m11 = in_mask & (oh1 < out_h) & (ow1 < out_w) & (oh1 != oh0) & (ow1 != ow0)

    g00 = tl.load(go_base + oh0 * out_w + ow0, mask=m00, other=0.0)
    g01 = tl.load(go_base + oh0 * out_w + ow1, mask=m01, other=0.0)
    g10 = tl.load(go_base + oh1 * out_w + ow0, mask=m10, other=0.0)
    g11 = tl.load(go_base + oh1 * out_w + ow1, mask=m11, other=0.0)
    if not CEIL_TAIL_H and not CEIL_TAIL_W:
        inv9 = 1.0 / 9.0
        grad_acc = (
            g00.to(tl.float32) * inv9
            + g01.to(tl.float32) * inv9
            + g10.to(tl.float32) * inv9
            + g11.to(tl.float32) * inv9
        )
    else:
        # Per-axis reciprocal of the pool size (once per axis instead of a
        # full _k3_divisor + FP division per tap); multiply-only scaling.
        if CEIL_TAIL_H:
            inv_h0 = _ceil_pool_inv(oh0, in_h)
            inv_h1 = _ceil_pool_inv(oh1, in_h)
        else:
            inv_h0 = 1.0 / 3.0
            inv_h1 = 1.0 / 3.0
        if CEIL_TAIL_W:
            inv_w0 = _ceil_pool_inv(ow0, in_w)
            inv_w1 = _ceil_pool_inv(ow1, in_w)
        else:
            inv_w0 = 1.0 / 3.0
            inv_w1 = 1.0 / 3.0
        grad_acc = inv_h0 * (
            g00.to(tl.float32) * inv_w0 + g01.to(tl.float32) * inv_w1
        ) + inv_h1 * (g10.to(tl.float32) * inv_w0 + g11.to(tl.float32) * inv_w1)

    gi_base = grad_input_ptr + pid_nc * in_h * in_w
    tl.store(gi_base + h2 * in_w + w2, grad_acc, mask=in_mask)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_k3s2p1_nopad_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # K3/S2/P1, count_include_pad=False, no ceil: same residue mapping as
    # _avg_pool2d_backward_k3s2p1_kernel, but the divisor is the valid-window
    # size valid_h*valid_w (each in {1,2,3}) instead of the constant 9.
    pid_nc = tl.program_id(0)
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h2 = h[:, None]
    w2 = w[None, :]
    in_mask = (h2 < in_h) & (w2 < in_w)

    oh0 = h2 >> 1
    oh1 = (h2 + 1) >> 1
    ow0 = w2 >> 1
    ow1 = (w2 + 1) >> 1
    go_base = grad_output_ptr + pid_nc * out_h * out_w
    m00 = in_mask & (oh0 < out_h) & (ow0 < out_w)
    m01 = in_mask & (oh0 < out_h) & (ow1 < out_w) & (ow1 != ow0)
    m10 = in_mask & (oh1 < out_h) & (ow0 < out_w) & (oh1 != oh0)
    m11 = in_mask & (oh1 < out_h) & (ow1 < out_w) & (oh1 != oh0) & (ow1 != ow0)

    g00 = tl.load(go_base + oh0 * out_w + ow0, mask=m00, other=0.0)
    g01 = tl.load(go_base + oh0 * out_w + ow1, mask=m01, other=0.0)
    g10 = tl.load(go_base + oh1 * out_w + ow0, mask=m10, other=0.0)
    g11 = tl.load(go_base + oh1 * out_w + ow1, mask=m11, other=0.0)

    # Valid-window size per axis: min(2o+2, in_len) - max(2o-1, 0), in
    # {1,2,3} for live producers.  Masked-out producers may yield <= 0,
    # which folds to the 1/3 slot: harmless because their grad is 0 and the
    # value stays finite.  Select-chain reciprocal instead of FP division
    # (same <= ~2 ulp substitution cost as _ceil_pool_inv, verified against
    # the torch CPU autograd reference at atol=rtol=1e-5, fp32).
    hs0 = oh0 * 2 - 1
    hs1 = oh1 * 2 - 1
    ws0 = ow0 * 2 - 1
    ws1 = ow1 * 2 - 1
    vh0 = tl.minimum(hs0 + 3, in_h) - tl.maximum(hs0, 0)
    vh1 = tl.minimum(hs1 + 3, in_h) - tl.maximum(hs1, 0)
    vw0 = tl.minimum(ws0 + 3, in_w) - tl.maximum(ws0, 0)
    vw1 = tl.minimum(ws1 + 3, in_w) - tl.maximum(ws1, 0)
    inv_h0 = tl.where(vh0 == 1, 1.0, tl.where(vh0 == 2, 0.5, 1.0 / 3.0))
    inv_h1 = tl.where(vh1 == 1, 1.0, tl.where(vh1 == 2, 0.5, 1.0 / 3.0))
    inv_w0 = tl.where(vw0 == 1, 1.0, tl.where(vw0 == 2, 0.5, 1.0 / 3.0))
    inv_w1 = tl.where(vw1 == 1, 1.0, tl.where(vw1 == 2, 0.5, 1.0 / 3.0))

    # divisor = vh*vw; grad = g / divisor == g * inv_vh * inv_vw.
    # Factored form: 4 full-tile multiplies + 3 adds per element.
    grad_acc = inv_h0 * (
        g00.to(tl.float32) * inv_w0 + g01.to(tl.float32) * inv_w1
    ) + inv_h1 * (g10.to(tl.float32) * inv_w0 + g11.to(tl.float32) * inv_w1)

    gi_base = grad_input_ptr + pid_nc * in_h * in_w
    tl.store(gi_base + h2 * in_w + w2, grad_acc, mask=in_mask)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_OH": 4, "BLOCK_OW": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_OH": 4, "BLOCK_OW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_OH": 16, "BLOCK_OW": 16}, num_stages=2, num_warps=4),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_k3s2p1_quad_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    oh_block = tl.program_id(1)
    ow_block = tl.program_id(2)

    oh = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)
    oh2 = oh[:, None]
    ow2 = ow[None, :]
    oh1 = oh2 + 1
    ow1 = ow2 + 1

    go_base = grad_output_ptr + pid_nc * out_h * out_w
    m00 = (oh2 < out_h) & (ow2 < out_w)
    m01 = (oh2 < out_h) & (ow1 < out_w)
    m10 = (oh1 < out_h) & (ow2 < out_w)
    m11 = (oh1 < out_h) & (ow1 < out_w)

    g00 = tl.load(go_base + oh2 * out_w + ow2, mask=m00, other=0.0)
    g01 = tl.load(go_base + oh2 * out_w + ow1, mask=m01, other=0.0)
    g10 = tl.load(go_base + oh1 * out_w + ow2, mask=m10, other=0.0)
    g11 = tl.load(go_base + oh1 * out_w + ow1, mask=m11, other=0.0)

    inv9 = 1.0 / 9.0
    s00 = g00.to(tl.float32) * inv9
    s01 = g01.to(tl.float32) * inv9
    s10 = g10.to(tl.float32) * inv9
    s11 = g11.to(tl.float32) * inv9

    h0 = oh2 << 1
    h1 = h0 + 1
    w0 = ow2 << 1
    w1 = w0 + 1
    gi_base = grad_input_ptr + pid_nc * in_h * in_w

    tl.store(
        gi_base + h0 * in_w + w0,
        s00,
        mask=(h0 < in_h) & (w0 < in_w),
    )
    tl.store(
        gi_base + h0 * in_w + w1,
        s00 + s01,
        mask=(h0 < in_h) & (w1 < in_w),
    )
    tl.store(
        gi_base + h1 * in_w + w0,
        s00 + s10,
        mask=(h1 < in_h) & (w0 < in_w),
    )
    tl.store(
        gi_base + h1 * in_w + w1,
        s00 + s01 + s10 + s11,
        mask=(h1 < in_h) & (w1 < in_w),
    )


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_OH": 4, "BLOCK_OW": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_OH": 4, "BLOCK_OW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_OH": 16, "BLOCK_OW": 16}, num_stages=2, num_warps=4),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_k3s2p1_quad_merge_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    # Merged-store variant of _avg_pool2d_backward_k3s2p1_quad_kernel: the
    # four per-tap stores into the (2h,2w) receiver quad are combined into
    # one contiguous (2*BLOCK_OH, 2*BLOCK_OW) tile -- row pairs are
    # interleaved along W, then the two row stripes are interleaved along H
    # (via trans) -- issuing a single coalesced store per program instead
    # of four strided ones (4x fewer store instructions, no smem staging).
    # ~1.13x on large spatial (112x112) but 0.76-0.83x on small spatial
    # (the interleave/trans chain forces smem round-trips there), hence
    # the shape gate.
    pid_nc = tl.program_id(0)
    oh_block = tl.program_id(1)
    ow_block = tl.program_id(2)

    oh = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)
    oh2 = oh[:, None]
    ow2 = ow[None, :]
    oh1 = oh2 + 1
    ow1 = ow2 + 1

    go_base = grad_output_ptr + pid_nc * out_h * out_w
    m00 = (oh2 < out_h) & (ow2 < out_w)
    m01 = (oh2 < out_h) & (ow1 < out_w)
    m10 = (oh1 < out_h) & (ow2 < out_w)
    m11 = (oh1 < out_h) & (ow1 < out_w)

    g00 = tl.load(go_base + oh2 * out_w + ow2, mask=m00, other=0.0)
    g01 = tl.load(go_base + oh2 * out_w + ow1, mask=m01, other=0.0)
    g10 = tl.load(go_base + oh1 * out_w + ow2, mask=m10, other=0.0)
    g11 = tl.load(go_base + oh1 * out_w + ow1, mask=m11, other=0.0)

    inv9 = 1.0 / 9.0
    s00 = g00.to(tl.float32) * inv9
    s01 = g01.to(tl.float32) * inv9
    s10 = g10.to(tl.float32) * inv9
    s11 = g11.to(tl.float32) * inv9

    # Receiver quad gradients:
    #   (2h, 2w)     -> s00
    #   (2h, 2w+1)   -> s00 + s01
    #   (2h+1, 2w)   -> s00 + s10
    #   (2h+1, 2w+1) -> s00 + s01 + s10 + s11
    a = s00
    b = s00 + s01
    c = s00 + s10
    d = s00 + s01 + s10 + s11
    r0 = tl.interleave(a, b)
    r1 = tl.interleave(c, d)
    full = tl.trans(tl.interleave(tl.trans(r0), tl.trans(r1)))

    oh_base = oh_block * BLOCK_OH
    ow_base = ow_block * BLOCK_OW
    rv = oh_base * 2 + tl.arange(0, 2 * BLOCK_OH)
    cv = ow_base * 2 + tl.arange(0, 2 * BLOCK_OW)
    rv2 = rv[:, None]
    cv2 = cv[None, :]
    gi_base = grad_input_ptr + pid_nc * in_h * in_w
    tl.store(
        gi_base + rv2 * in_w + cv2,
        full,
        mask=(rv2 < in_h) & (cv2 < in_w),
    )


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=1),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_k2s1p0_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)

    h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    h2 = h[:, None]
    w2 = w[None, :]
    in_mask = (h2 < in_h) & (w2 < in_w)

    # K=2,S=1,P=0: input (h,w) is produced by output taps
    # (h-1,w-1),(h-1,w),(h,w-1),(h,w), clamped to the output extent.
    oh0 = h2 - 1
    oh1 = h2
    ow0 = w2 - 1
    ow1 = w2
    m00 = in_mask & (oh0 >= 0) & (oh0 < out_h) & (ow0 >= 0) & (ow0 < out_w)
    m01 = in_mask & (oh0 >= 0) & (oh0 < out_h) & (ow1 < out_w)
    m10 = in_mask & (oh1 < out_h) & (ow0 >= 0) & (ow0 < out_w)
    m11 = in_mask & (oh1 < out_h) & (ow1 < out_w)

    go_base = grad_output_ptr + pid_nc * out_h * out_w
    g00 = tl.load(go_base + oh0 * out_w + ow0, mask=m00, other=0.0)
    g01 = tl.load(go_base + oh0 * out_w + ow1, mask=m01, other=0.0)
    g10 = tl.load(go_base + oh1 * out_w + ow0, mask=m10, other=0.0)
    g11 = tl.load(go_base + oh1 * out_w + ow1, mask=m11, other=0.0)

    inv = 1.0 / float(DIVISOR_OVERRIDE)
    grad_acc = (
        g00.to(tl.float32) * inv
        + g01.to(tl.float32) * inv
        + g10.to(tl.float32) * inv
        + g11.to(tl.float32) * inv
    )

    gi_base = grad_input_ptr + pid_nc * in_h * in_w
    tl.store(gi_base + h2 * in_w + w2, grad_acc, mask=in_mask)


@libentry()
@libtuner(
    configs=[
        # 1D W-strips only: every config keeps BLOCK_W >= 32 * num_warps so
        # each thread holds a contiguous strip along W (the loop-carried
        # rows are (BLOCK_W,) vectors, not 2D tiles).
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 128}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 256}, num_stages=2, num_warps=8),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_backward_k2s1p0_carry_kernel(
    grad_output_ptr,
    grad_input_ptr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Loop-carry variant of _avg_pool2d_backward_k2s1p0_kernel: input row h
    # needs the output rows h-1 and h at column taps w-1 and w, so instead
    # of four shifted tile loads per receiver block, each iteration loads
    # only the two (BLOCK_W,) strips of row h and reuses the row h-1 strips
    # carried in registers from the previous iteration.  Per program this
    # issues 2*(BLOCK_H+1) strip loads instead of 4*BLOCK_H tile loads
    # (~2x fewer load instructions / elements); ~1.75x on large spatial
    # (512x512) but 0.4-0.9x on small spatial, hence the shape gate.
    pid_nc = tl.program_id(0)
    h_block = tl.program_id(1)
    w_block = tl.program_id(2)

    w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    wm1 = w - 1
    h0 = h_block * BLOCK_H

    go_base = grad_output_ptr + pid_nc * out_h * out_w
    gi_base = grad_input_ptr + pid_nc * in_h * in_w

    # Column taps of input column w are the output columns w-1 and w,
    # clamped to the output extent.
    mcol_a = (wm1 >= 0) & (wm1 < out_w)
    mcol_b = w < out_w

    # Row h0 - 1 feeds the first receiver row through the (h-1, *) taps;
    # load it once, then carry row h-1 in registers across the loop.
    r = h0 - 1
    mrow0 = (r >= 0) & (r < out_h)
    prev_a = tl.load(
        go_base + r * out_w + wm1,
        mask=mrow0 & mcol_a,
        other=0.0,
    ).to(tl.float32)
    prev_b = tl.load(
        go_base + r * out_w + w,
        mask=mrow0 & mcol_b,
        other=0.0,
    ).to(tl.float32)

    inv = 1.0 / float(DIVISOR_OVERRIDE)
    for i in tl.range(0, BLOCK_H):
        h = h0 + i
        mrow = h < out_h
        cur_a = tl.load(
            go_base + h * out_w + wm1,
            mask=mrow & mcol_a,
            other=0.0,
        ).to(tl.float32)
        cur_b = tl.load(
            go_base + h * out_w + w,
            mask=mrow & mcol_b,
            other=0.0,
        ).to(tl.float32)
        grad_acc = (prev_a + prev_b + cur_a + cur_b) * inv
        tl.store(
            gi_base + h * in_w + w,
            grad_acc,
            mask=(h < in_h) & (w < in_w),
        )
        prev_a = cur_a
        prev_b = cur_b


class _DirectLauncher:
    """Steady-state launcher captured after the first (autotuned) launch.

    ``run`` is the MusaLauncher instance behind CompiledKernel.run and
    ``launch`` is its raw C entry, whose ABI is (see
    triton/backends/mthreads/driver.py, MusaLauncher.launch):

        launch(gridX, gridY, gridZ, stream, function, kernel_metadata,
               launch_metadata, launch_enter_hook, launch_exit_hook,
               *kernel_args)

    Bypassing the MusaLauncher.__call__ wrapper (closure rebuild plus
    global-scratch bookkeeping) is only valid when the kernel needs no
    per-launch global scratch and no runtime argument expansion, so
    ``launch`` stays None -- and the wrapper is used -- otherwise.
    """

    __slots__ = (
        "run",
        "launch",
        "function",
        "packed_metadata",
        "grid",
        "tail",
        "runner",
    )

    def __init__(self, compiled, grid, tail):
        run = compiled.run
        self.run = run
        self.launch = (
            run.launch
            if not getattr(run, "_needs_runtime_expansion", False)
            and not getattr(run, "global_scratch_size", 0)
            else None
        )
        self.function = compiled.function
        self.packed_metadata = compiled.packed_metadata
        self.grid = grid
        self.tail = tail
        self.runner = compiled[grid]


_direct_launch_cache = {}


def _hooks_active():
    hook = _RUNTIME_KNOBS.launch_enter_hook
    if hook is None:
        return False
    calls = getattr(hook, "calls", None)
    return True if calls is None else len(calls) > 0


_dispatch_memo = {}

# Hoisted bound method for the steady-state lookup (LOAD_GLOBAL + CALL
# instead of LOAD_GLOBAL dict + LOAD_ATTR get + CALL).
_memo_get = _dispatch_memo.get


def _make_dispatch_entry(
    launcher, static_tail, grid, device_id, go_aligned, gi_aligned, slow_call
):
    """Build the steady-state memo entry (a flat tuple) for one resolved case.

    The memo entry is consumed inline by the dispatch wrapper's hot path
    (no closure call frame): the wrapper re-creates the output tensor,
    re-verifies the contract the cached binary was specialized for, and
    launches the cached binary through the raw C launcher entry with a
    pre-packed static parameter tail: per call only the two pointer
    arguments and the stream vary.  Pointer arguments are passed as
    ``data_ptr`` integers, which the launcher C ABI accepts directly (its
    ``getPointer`` has a PyLong fast path that skips the per-tensor
    ``data_ptr`` method call and the ``muPointerGetAttribute`` driver
    round-trip; both pointers are read once by the wrapper anyway for the
    alignment re-check).  The stream is queried per call via
    ``_get_raw_musa_stream`` on the memoized device index, so
    non-default-stream and multi-device semantics are preserved.  Shapes
    and the grad_output device are pinned by the memo key, so the entry
    only re-checks what a fresh call can violate: pointer 16-byte
    alignment (pre-folded here into ``go_mask``/``gi_mask``: 15 when the
    compiled binary specialized on a 16-aligned pointer and the fresh
    pointer must stay aligned, 0 when any address is tolerated), the
    input tensor's device and contiguity, and the current device (a
    mismatch enters the tensor device's context for the launch, mirroring
    a native op's DeviceGuard).  Any deviation (misaligned pointers,
    cross-device input, non-contiguous operands, installed triton launch
    hooks) falls back to the full dispatch path; launch exceptions
    propagate natively.
    """
    return (
        launcher.launch,
        launcher.function,
        launcher.packed_metadata,
        grid[0],
        grid[1],
        grid[2],
        static_tail,
        device_id,
        15 if go_aligned else 0,
        15 if gi_aligned else 0,
        launcher.run,
        launcher.runner,
        slow_call,
    )


def _direct_launch(
    entry,
    cache_key,
    grid_fn,
    args,
    extra,
    memo_key=None,
    slow_call=None,
    memo=None,
    make_entry=None,
):
    """Launch ``entry`` through a per-geometry cached CompiledKernel.

    The first call goes through the full @libentry + autotune path, which
    selects the config and compiles the kernel.  The returned
    CompiledKernel, the resolved grid, and the trailing constexpr values
    are captured so that subsequent calls with an identical specialization
    profile launch the binary directly through the raw C launcher entry,
    skipping the per-call key reconstruction and config rescans of the
    generic dispatch path.

    ``cache_key`` carries every value the launcher specializes on: the
    constexpr kwargs (including the baked geometry), and the 16-byte
    alignment of both tensor pointers.  The stream is queried per call from
    the current device, so non-default-stream semantics are preserved.
    When triton launch hooks are installed the hook-aware CompiledKernel
    runner is used instead.

    ``memo``/``make_entry`` optionally redirect the steady-state dispatch
    registration (the forward wrapper uses its own memo dict and a flat
    entry that also pins the output shape); the default registers the
    backward-style entry into ``_dispatch_memo``.
    """
    launcher = _direct_launch_cache.get(cache_key)
    device_id = args[0].get_device()
    if launcher is None:
        # First call: full @libentry + autotune path (also performs the
        # launch).  Enter the tensor device's context first when the
        # current device differs, so the tuning/compile/launch all see
        # the right device (matching a native op's DeviceGuard).
        if _get_raw_musa_device() == device_id:
            compiled, constexprs = entry[grid_fn](*args, **extra)
        else:
            with _device_guard(device_id):
                compiled, constexprs = entry[grid_fn](*args, **extra)
        meta = dict(zip(entry.arg_names, args))
        meta.update(extra)
        meta.update(constexprs)
        grid = tuple(grid_fn(meta))
        tail = []
        for name in list(entry.signature.parameters)[len(args) :]:
            tail.append(extra[name] if name in extra else constexprs[name])
        launcher = _DirectLauncher(compiled, grid, tuple(tail))
        _direct_launch_cache[cache_key] = launcher
        # entry[grid_fn] above already performed this call's launch.
        launched = True
    else:
        launched = False
    # Register the steady-state dispatch entry for every distinct memo key
    # that resolves to this launcher, not only the one that paid the compile:
    # e.g. the int and tuple spellings of the same pooling config map to
    # the same cache_key but different memo keys, and each deserves a
    # memo entry instead of forever re-running the impl dispatch.
    if memo is None:
        memo = _dispatch_memo
    if make_entry is None:
        make_entry = _make_dispatch_entry
    if memo_key is not None and memo_key not in memo:
        memo[memo_key] = make_entry(
            launcher,
            tuple(args[2:]) + launcher.tail,
            launcher.grid,
            device_id,
            not args[0].data_ptr() % 16,
            not args[1].data_ptr() % 16,
            slow_call,
        )
    if launched:
        return
    if _hooks_active():
        launcher.runner(*args, *launcher.tail)
        return
    g = launcher.grid
    # Stream is queried per call from the *operands'* device: the MUSA
    # launch API requires the stream to belong to the current context, so
    # a current-device mismatch enters the tensor device's context for
    # the launch (this branch serves the hot path's fallbacks, whose
    # launches on the tensor device's current stream are then valid for
    # pointers, binary and stream alike).
    if _get_raw_musa_device() == device_id:
        stream = _get_raw_musa_stream(device_id)
        if launcher.launch is not None:
            launcher.launch(
                g[0],
                g[1],
                g[2],
                stream,
                launcher.function,
                launcher.packed_metadata,
                None,
                None,
                None,
                args[0].data_ptr(),
                args[1].data_ptr(),
                *args[2:],
                *launcher.tail,
            )
        else:
            launcher.run(
                g[0],
                g[1],
                g[2],
                stream,
                launcher.function,
                launcher.packed_metadata,
                None,
                None,
                None,
                *args,
                *launcher.tail,
            )
    else:
        with _device_guard(device_id):
            stream = _get_raw_musa_stream(device_id)
            if launcher.launch is not None:
                launcher.launch(
                    g[0],
                    g[1],
                    g[2],
                    stream,
                    launcher.function,
                    launcher.packed_metadata,
                    None,
                    None,
                    None,
                    args[0].data_ptr(),
                    args[1].data_ptr(),
                    *args[2:],
                    *launcher.tail,
                )
            else:
                launcher.run(
                    g[0],
                    g[1],
                    g[2],
                    stream,
                    launcher.function,
                    launcher.packed_metadata,
                    None,
                    None,
                    None,
                    *args,
                    *launcher.tail,
                )


def _launch_backward(
    kernel,
    grad_output,
    grad_input,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    ceil_mode,
    count_include_pad,
    divisor_override,
    memo_key=None,
    slow_call=None,
):
    in_n = grad_input.shape[0]
    nc = in_n * grad_input.shape[1]
    if (
        kernel is _avg_pool2d_backward_k3s2p1_quad_kernel
        or kernel is _avg_pool2d_backward_k3s2p1_quad_merge_kernel
    ):
        grid_fn = lambda meta: (
            nc,
            triton.cdiv(out_h, meta["BLOCK_OH"]),
            triton.cdiv(out_w, meta["BLOCK_OW"]),
        )
        extra = {}
    else:
        grid_fn = lambda meta: (
            nc,
            triton.cdiv(in_h, meta["BLOCK_H"]),
            triton.cdiv(in_w, meta["BLOCK_W"]),
        )
        if (
            kernel is _avg_pool2d_backward_k2s1p0_kernel
            or kernel is _avg_pool2d_backward_k2s1p0_carry_kernel
        ):
            extra = {"DIVISOR_OVERRIDE": divisor_override}
        elif kernel is _avg_pool2d_backward_k3s2p1_nopad_kernel:
            extra = {}
        else:
            ceil_tail_h = bool(
                ceil_mode
                and out_h > 0
                and (out_h - 1) * stride_h - padding_h + kernel_h > in_h + padding_h
            )
            ceil_tail_w = bool(
                ceil_mode
                and out_w > 0
                and (out_w - 1) * stride_w - padding_w + kernel_w > in_w + padding_w
            )
            if kernel is _avg_pool2d_backward_direct_range_kernel:
                extra = {
                    "COUNT_INCLUDE_PAD": count_include_pad,
                    "CEIL_TAIL_H": ceil_tail_h,
                    "CEIL_TAIL_W": ceil_tail_w,
                    "DIVISOR_OVERRIDE": divisor_override or 0,
                }
            else:
                extra = {"CEIL_TAIL_H": ceil_tail_h, "CEIL_TAIL_W": ceil_tail_w}
    # The geometry ints are constexpr kwargs (see the module docstring): the
    # compiled binary folds their values, so ``args`` carries only the two
    # tensor pointers and every integer rides the pre-packed constexpr tail
    # instead of being passed -- and PyLong-parsed -- per steady-state call.
    # The tuning keys still name in_h/in_w/out_h/out_w, and kwargs are
    # visible to the tuner's key lookup, so autotuning granularity is
    # unchanged.  A separate binary is compiled per geometry (persisted by
    # the triton disk cache; the first process on cold caches pays a one-time
    # compile per shape, later processes only load).
    extra = {
        "in_h": in_h,
        "in_w": in_w,
        "out_h": out_h,
        "out_w": out_w,
        **extra,
    }
    args = [grad_output, grad_input]
    if kernel is _avg_pool2d_backward_direct_range_kernel:
        extra.update(
            {
                "kernel_h": kernel_h,
                "kernel_w": kernel_w,
                "stride_h": stride_h,
                "stride_w": stride_w,
                "padding_h": padding_h,
                "padding_w": padding_w,
            }
        )
    cache_key = (
        kernel,
        # nc, not in_n: the memo entry bakes the grid's first dimension
        # (in_n * in_c) at compile time, so two calls that differ only in
        # the channel count must not share a memo entry (they share the
        # binary -- the kernels ignore in_c -- but not the grid).
        nc,
        grad_output.get_device(),
        tuple(extra.items()),
        grad_output.data_ptr() % 16 == 0,
        grad_input.data_ptr() % 16 == 0,
    )
    _direct_launch(kernel, cache_key, grid_fn, args, extra, memo_key, slow_call)


# Shape gates for the V5 specialized variants (measured on GPU6, MUSA,
# fp32, steady-state CompiledKernel timing vs the torch CPU autograd
# reference; full sweeps in the V5 experiment logs).  Both gates only
# feed kernel selection; the dispatch memo key already contains the full
# input/grad_output shapes, so the gate dimensions (in_h/in_w/nc) are
# covered by the memoization key.

# _avg_pool2d_backward_k2s1p0_carry_kernel: 1.48-1.75x on large spatial
# maps (in_w >= 256, e.g. (1,32,512,512) 1.57-1.75x with 44% fewer global
# loads and +46% DRAM bandwidth), but 0.75-0.95x on tall/narrow (512x64)
# and mid feature-map sizes, where the 1D strip layout loses occupancy
# (56x56 through the C++ heuristic regressed to 0.83x in the official
# kernel-mode benchmark; the Python-path sweep showed 1.01x at best).
# The 50..79 window was therefore dropped: keep the gate conservative
# and only route wide maps to the loop-carry kernel.
_K2S1P0_CARRY_LARGE_W = 256

# _avg_pool2d_backward_k3s2p1_quad_merge_kernel: 1.13-1.14x on
# (2, 512, 112, 112) but 0.73-0.83x on 56x56/72x72 spatial sizes, where
# the interleave/trans chain forces smem layout-conversion round-trips.
# (Probes show wins at 64/80/96 too, but non-monotonically with
# config-dependent outliers; 112 is the only stably-confirmed class.)
# Route large spatial (in_h, in_w >= 112) to the merged single store.
_QUAD_MERGE_MIN_HW = 112


def _k2s1p0_use_carry(in_h, in_w, nc):
    return in_w >= _K2S1P0_CARRY_LARGE_W


def _quad_use_merge(in_h, in_w):
    return in_h >= _QUAD_MERGE_MIN_HW and in_w >= _QUAD_MERGE_MIN_HW


def avg_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    """MTHREADS dispatch with a semantics-preserving generic fallback.

    Steady state is served from a dispatch memo keyed on the minimal
    discriminative set: the grad_output device index, both tensor shapes
    and the raw pooling configuration.  int and tuple spellings of a
    config hash to distinct (but equally valid) keys; the dispatcher
    boxes every IntArrayRef config as a fresh Python list, which is
    unhashable, so list spellings are normalized to tuples with three
    cheap ``type is list`` probes BEFORE the single key build -- paying
    ~0.4us of tuple conversion instead of a per-call TypeError
    round-trip on the dict probe (list and tuple spellings then converge
    on one memo entry).  The registered entry is a flat tuple of
    per-case constants that the hot path unpacks and runs inline (no
    closure call frame): it re-allocates the output and launches the
    cached CompiledKernel through the raw C launcher entry, re-verifying
    pointer alignment, the input device and contiguity per call; the
    launch stream is queried per call, preserving non-default-stream
    semantics.  ``divisor_override == 0`` can never be memoized (the cold
    path raises before registration), so its validation lives on the
    memo-miss branch.
    """
    if input.dtype is _FLOAT32 and grad_output.dtype is _FLOAT32:
        if type(kernel_size) is list:
            kernel_size = tuple(kernel_size)
        if type(stride) is list:
            stride = tuple(stride)
        if type(padding) is list:
            padding = tuple(padding)
        key = (
            grad_output.get_device(),
            input.shape,
            grad_output.shape,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        ent = _memo_get(key)
        if ent is not None:
            # Inline steady-state hot path: the memo entry is a flat tuple
            # of per-case constants, so the launch body runs directly in
            # this frame instead of through a closure call (one Python
            # call frame saved per hit).  Semantics mirror the former
            # closure exactly, including the hook-aware runner, the
            # current-device guard and the slow-path fallback.
            (
                c_launch,
                fn_handle,
                packed,
                g0,
                g1,
                g2,
                static_tail,
                dev_id,
                go_mask,
                gi_mask,
                run,
                runner,
                slow,
            ) = ent
            grad_input = torch.empty_like(input)
            go_ptr = grad_output.data_ptr()
            gi_ptr = grad_input.data_ptr()
            if (
                not go_ptr & go_mask
                and not gi_ptr & gi_mask
                and grad_input.get_device() == dev_id
                and input.is_contiguous()
                and grad_output.is_contiguous()
            ):
                # Inlined _hooks_active: the steady state has the default
                # empty HookChain installed, so the common route is two
                # attribute reads.  A plain callable hook (no .calls) or a
                # non-empty chain takes the hook-aware runner instead.
                hook = _RUNTIME_KNOBS.launch_enter_hook
                if hook is not None:
                    try:
                        calls = hook.calls
                    except AttributeError:
                        calls = True
                    if calls:
                        runner(grad_output, grad_input, *static_tail)
                        return grad_input
                if _get_raw_musa_device() == dev_id:
                    if c_launch is None:
                        run(
                            g0,
                            g1,
                            g2,
                            _get_raw_musa_stream(dev_id),
                            fn_handle,
                            packed,
                            None,
                            None,
                            None,
                            grad_output,
                            grad_input,
                            *static_tail,
                        )
                        return grad_input
                    c_launch(
                        g0,
                        g1,
                        g2,
                        _get_raw_musa_stream(dev_id),
                        fn_handle,
                        packed,
                        None,
                        None,
                        None,
                        go_ptr,
                        gi_ptr,
                        *static_tail,
                    )
                else:
                    # Current device differs from the operands': the MUSA
                    # launch API requires the stream to belong to the
                    # current context, so enter the tensor device's
                    # context and launch on that device's current stream.
                    with _device_guard(dev_id):
                        if c_launch is None:
                            run(
                                g0,
                                g1,
                                g2,
                                _get_raw_musa_stream(dev_id),
                                fn_handle,
                                packed,
                                None,
                                None,
                                None,
                                grad_output,
                                grad_input,
                                *static_tail,
                            )
                        else:
                            c_launch(
                                g0,
                                g1,
                                g2,
                                _get_raw_musa_stream(dev_id),
                                fn_handle,
                                packed,
                                None,
                                None,
                                None,
                                go_ptr,
                                gi_ptr,
                                *static_tail,
                            )
                return grad_input
            return slow(grad_output, input)

        def slow_call(go, inp):
            return _avg_pool2d_backward_impl(
                go,
                inp,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            )

        return _avg_pool2d_backward_impl(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memo_key=key,
            slow_call=slow_call,
        )
    return _avg_pool2d_backward_impl(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def _avg_pool2d_backward_impl(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    memo_key=None,
    slow_call=None,
):
    """MTHREADS dispatch with a semantics-preserving generic fallback."""
    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    # Keep the existing implementation for layouts/dtypes outside the tuned
    # FP32 contiguous contract.  In particular this preserves arbitrary
    # strides and all PyTorch corner-case behavior.
    if input.dtype != torch.float32 or grad_output.dtype != torch.float32:
        return _generic_avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )
    # With no spatial decimation the inverse range has exactly K^2 producers,
    # so the extra range/divisor arithmetic costs more than the old kernel.
    # A runtime divisor override is likewise already a cheap constant path;
    # retain the established implementation for these non-winning cases.
    if (
        kernel_h == 2
        and kernel_w == 2
        and stride_h == 1
        and stride_w == 1
        and padding_h == 0
        and padding_w == 0
        and divisor_override is not None
        and input.is_contiguous()
        and grad_output.is_contiguous()
    ):
        # Fully-overlapping 2x2 taps with a constant divisor: each input
        # element has at most four producers at compile-time offsets, and
        # the divisor degenerates to a single compile-time reciprocal.
        in_n, in_c, in_h, in_w = input.shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        grad_input = torch.empty_like(input, dtype=torch.float32)
        if grad_output.numel() == 0:
            # Empty batch (or degenerate spatial size): ATen returns zeros
            # for the former and cannot express the latter; zeros are the
            # safe superset either way.
            grad_input.zero_()
            return grad_input
        # Large / mid spatial favors the loop-carry strip kernel
        # (see _k2s1p0_use_carry); small spatial keeps the tiled kernel.
        _launch_backward(
            (
                _avg_pool2d_backward_k2s1p0_carry_kernel
                if _k2s1p0_use_carry(in_h, in_w, in_n * in_c)
                else _avg_pool2d_backward_k2s1p0_kernel
            ),
            grad_output,
            grad_input,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memo_key,
            slow_call,
        )
        return grad_input
    if (stride_h == 1 and stride_w == 1) or divisor_override is not None:
        return _generic_avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    if not input.is_contiguous() or not grad_output.is_contiguous():
        return _generic_avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]
    # The kernel writes every input element, including receivers with no
    # producer, so no zero-initialization launch is needed.
    grad_input = torch.empty_like(input, dtype=torch.float32)
    if grad_output.numel() == 0:
        # The only reachable empty-grad_output configuration is an empty
        # batch dimension, for which ATen returns zeros; zero-fill here to
        # never return uninitialized memory.
        grad_input.zero_()
        return grad_input

    if (
        kernel_h == 3
        and kernel_w == 3
        and stride_h == 2
        and stride_w == 2
        and padding_h == 1
        and padding_w == 1
        and count_include_pad
        and divisor_override is None
        and not ceil_mode
        and in_n * in_c >= 1024
        and in_h * in_w > 64
    ):
        # Large spatial (in_h, in_w >= 112) favors the merged single
        # store (see _quad_use_merge); smaller spatial keeps the
        # four-store quad kernel.
        _launch_backward(
            (
                _avg_pool2d_backward_k3s2p1_quad_merge_kernel
                if _quad_use_merge(in_h, in_w)
                else _avg_pool2d_backward_k3s2p1_quad_kernel
            ),
            grad_output,
            grad_input,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memo_key,
            slow_call,
        )
    elif (
        kernel_h == 3
        and kernel_w == 3
        and stride_h == 2
        and stride_w == 2
        and padding_h == 1
        and padding_w == 1
        and not count_include_pad
        and divisor_override is None
        and not ceil_mode
    ):
        # count_include_pad=False keeps the residue mapping of the k3s2p1
        # fast path but needs the valid-window divisor; the dedicated
        # no-division kernel beats direct_range on every shape (up to ~4x
        # on small spatial sizes such as (128, 512, 7, 7)).
        _launch_backward(
            _avg_pool2d_backward_k3s2p1_nopad_kernel,
            grad_output,
            grad_input,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memo_key,
            slow_call,
        )
    elif (
        kernel_h == 3
        and kernel_w == 3
        and stride_h == 2
        and stride_w == 2
        and padding_h == 1
        and padding_w == 1
        and count_include_pad
        and divisor_override is None
    ):
        _launch_backward(
            _avg_pool2d_backward_k3s2p1_kernel,
            grad_output,
            grad_input,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memo_key,
            slow_call,
        )
    else:
        _launch_backward(
            _avg_pool2d_backward_direct_range_kernel,
            grad_output,
            grad_input,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            ceil_mode,
            count_include_pad,
            divisor_override,
            memo_key,
            slow_call,
        )
    return grad_input


# --- Forward (V10) -------------------------------------------------------
#
# The generic FlagGems forward kernel maps one CTA per (n*c, spatial tile)
# and pays runtime integer division plus an FP division per output element.
# On MTT that collapses for small spatial / high channel counts (the 7x7
# ResNet tail allocates 65536 CTAs of sixteen outputs each; official kernel
# speedups of 0.30-0.49x).  The kernels below flatten the output to a
# per-channel linear index handled by a (BLOCK_C, BLOCK_S) tile so one CTA
# covers several channels, and bake the geometry into constexpr parameters
# exactly like the backward kernels: the s->(h, w) decomposition divides by
# a compile-time out_w (strength-reduced to multiply/shift), the window
# masks fold against constexpr in_h/in_w, and the divisor degenerates to
# compile-time reciprocals -- a select-chain of 3 slots per axis for the
# count_include_pad=False / ceil-tail families, mirroring the <= ~2 ulp
# reciprocal substitution already validated for the backward kernels.


_FORWARD_TUNING_CONFIGS = [
    # Small tiles: mid feature maps (56x56/28x28/14x14 inputs, nc 1-16k,
    # out 28x28..7x7) want many CTAs of 128-512 elements, mirroring the
    # small spatial tiles the generic kernel's autotuner picks there.
    triton.Config({"BLOCK_C": 2, "BLOCK_S": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_C": 4, "BLOCK_S": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_C": 4, "BLOCK_S": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_C": 8, "BLOCK_S": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_C": 8, "BLOCK_S": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 16, "BLOCK_S": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 2, "BLOCK_S": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_C": 4, "BLOCK_S": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 1, "BLOCK_S": 256}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_C": 2, "BLOCK_S": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 8, "BLOCK_S": 256}, num_warps=4, num_stages=2),
    # Large tiles: high-NC tails (128x512x7x7 -> nc 65536, out 4x4) want
    # fat multi-channel CTAs instead of 65k tiny programs.
    triton.Config({"BLOCK_C": 32, "BLOCK_S": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 64, "BLOCK_S": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 128, "BLOCK_S": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 256, "BLOCK_S": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_C": 4, "BLOCK_S": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 1, "BLOCK_S": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 2, "BLOCK_S": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_C": 1, "BLOCK_S": 1024}, num_warps=8, num_stages=2),
]


@libentry()
@libtuner(
    configs=_FORWARD_TUNING_CONFIGS,
    key=["in_h", "in_w", "out_h", "out_w", "nc"],
    # Persistent ConfigCache (sqlite) and 25/100 ms do_bench windows: see
    # the direct_range kernel note above.
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_forward_k3s2p1_kernel(
    input_ptr,
    output_ptr,
    nc: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    VARY_DIV: tl.constexpr,
    PAIR: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # One CTA covers BLOCK_C channels x one BLOCK_S tile of the flattened
    # out_h*out_w output.  All div/mod below use compile-time constants.
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    # Layout pin: when out_h*out_w is even the store offsets are 2-element
    # vectorizable and the coalescing pass then picks a [1,2] blocked
    # layout, which halves the stride-2 window loads' warp density (each
    # warp spans 512B instead of 256B per tap, 1.6x slower end to end --
    # measured on the 28x28/14x14 classes).  Capping the claimed
    # contiguity of the linear output index keeps every specialization on
    # the dense [1,1] pattern; the store stays a fully coalesced scalar
    # warp store.
    s_off = tl.max_contiguous(
        tl.multiple_of(pid_s * BLOCK_S + tl.arange(0, BLOCK_S), BLOCK_S), 1
    )
    S: tl.constexpr = out_h * out_w
    s_mask = s_off < S
    h = s_off // out_w
    w = s_off - h * out_w
    m_cs = (c_off < nc)[:, None] & s_mask[None, :]

    in_base = c_off[:, None] * (in_h * in_w)
    acc = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)
    w2 = w * 2
    for kh in tl.static_range(3):
        ih = h * 2 - 1 + kh
        vh = (ih >= 0) & (ih < in_h) & s_mask
        rb = ih * in_w
        if PAIR:
            # kw=0 tap: x[2w-1], a stride-2 load (left pad masked).
            iw0 = w2 - 1
            m0 = m_cs & (vh & (iw0 >= 0) & (iw0 < in_w))[None, :]
            acc += tl.load(
                input_ptr + in_base + rb + iw0[None, :], mask=m0, other=0.0
            ).to(tl.float32)
            # kw=1 and kw=2 taps fused: (x[2w], x[2w+1]) as one (BC, BS, 2)
            # pair load whose last axis is contiguous -- with even in_w the
            # 8-byte pairs are aligned and load vectorized, and the in-axis
            # sum is a pure in-register reduction (no cross-lane movement).
            ok0 = vh & (w2 < in_w)
            ok1 = vh & (w2 + 1 < in_w)
            pair_ptr = (
                input_ptr
                + in_base[:, None]
                + rb[None, :, None]
                + w2[None, :, None]
                + tl.arange(0, 2)[None, None, :]
            )
            pair_mask = m_cs[:, :, None] & tl.where(
                tl.arange(0, 2)[None, None, :] == 0,
                ok0[None, :, None],
                ok1[None, :, None],
            )
            pair = tl.load(pair_ptr, mask=pair_mask, other=0.0).to(tl.float32)
            acc += tl.sum(pair, axis=2)
        else:
            # Odd in_w: the pairs are never 8B-aligned, keep the plain
            # 9-tap formulation.
            for kw in tl.static_range(3):
                iw = w2 - 1 + kw
                vw = (iw >= 0) & (iw < in_w) & s_mask
                m = m_cs & (vh & vw)[None, :]
                acc += tl.load(
                    input_ptr + in_base + rb + iw[None, :], mask=m, other=0.0
                ).to(tl.float32)

    if not VARY_DIV:
        # Plain count_include_pad=True without a ceil tail: every window
        # counts exactly K*K taps (padding included).
        acc = acc * (1.0 / 9.0)
    else:
        # Per-axis window length in {1, 2, 3}: with the pad counted the axis
        # length is clamp(min(2o+2, in_len+1) - (2o-1), 0, .), without the
        # pad it is min(2o+2, in_len) - max(2o-1, 0).  Select-chain
        # reciprocals per axis (factored divisor = p_h * p_w) replace the
        # FP division, exactly like the backward nopad/ceil kernels.
        if COUNT_INCLUDE_PAD:
            ph = tl.minimum(h * 2 + 2, in_h + 1) - (h * 2 - 1)
            pw = tl.minimum(w * 2 + 2, in_w + 1) - (w * 2 - 1)
        else:
            ph = tl.minimum(h * 2 + 2, in_h) - tl.maximum(h * 2 - 1, 0)
            pw = tl.minimum(w * 2 + 2, in_w) - tl.maximum(w * 2 - 1, 0)
        inv_h = tl.where(ph == 1, 1.0, tl.where(ph == 2, 0.5, 1.0 / 3.0))
        inv_w = tl.where(pw == 1, 1.0, tl.where(pw == 2, 0.5, 1.0 / 3.0))
        acc = acc * inv_h[None, :] * inv_w[None, :]

    tl.store(
        output_ptr + c_off[:, None] * S + s_off[None, :],
        acc.to(output_ptr.type.element_ty),
        mask=m_cs,
    )


@libentry()
@libtuner(
    configs=_FORWARD_TUNING_CONFIGS,
    key=["in_h", "in_w", "out_h", "out_w", "nc"],
    strategy="align32",
    warmup=25,
    rep=100,
)
@triton.jit
def _avg_pool2d_forward_k2s1p0_kernel(
    input_ptr,
    output_ptr,
    nc: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    DIVISOR: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # K2/S1/P0 (fully in-bounds, stride 1): no boundary masks at all and a
    # single compile-time reciprocal divisor (divisor_override or K*K).
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    S: tl.constexpr = out_h * out_w
    s_mask = s_off < S
    h = s_off // out_w
    w = s_off - h * out_w
    m_cs = (c_off < nc)[:, None] & s_mask[None, :]

    base = input_ptr + c_off[:, None] * (in_h * in_w) + h[None, :] * in_w + w[None, :]
    acc = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)
    for kh in tl.static_range(2):
        for kw in tl.static_range(2):
            acc += tl.load(base + kh * in_w + kw, mask=m_cs, other=0.0).to(tl.float32)
    acc = acc * (1.0 / DIVISOR)

    tl.store(
        output_ptr + c_off[:, None] * S + s_off[None, :],
        acc.to(output_ptr.type.element_ty),
        mask=m_cs,
    )


_fwd_dispatch_memo = {}
_fwd_memo_get = _fwd_dispatch_memo.get
# Monomorphic fast slot for the most recently served forward case: a
# (shape, dtype, kernel_size, stride, padding, ceil_mode, count_include_pad,
# divisor_override, device, entry) tuple.  Every semantic parameter is
# re-validated per call (no object identity or stale-output shortcuts), so a
# miss simply falls through to the general tuple-key memo below.  Benign
# under concurrency: the GIL makes the tuple swap atomic and either the old
# or the new slot is a fully validated entry.
_fwd_fast = None


def _make_forward_dispatch_entry(
    launcher,
    static_tail,
    grid,
    device_id,
    in_aligned,
    out_shape,
    slow_call,
):
    """Steady-state memo entry for one resolved forward case.

    Mirrors ``_make_dispatch_entry`` (the ``out_aligned`` call-site slot is
    repurposed to carry the pinned output shape, bound by
    ``_launch_forward`` where it is known): the hot path re-creates the
    output via ``input.new_empty`` with that shape, re-verifies pointer
    alignment and contiguity, and launches the cached binary through the
    raw C launcher entry with the pre-packed constexpr tail.  The fresh
    output of a caching-allocator allocation is always 16-byte aligned, so
    the output pointer is not re-checked.
    """
    return (
        launcher.launch,
        launcher.function,
        launcher.packed_metadata,
        grid[0],
        grid[1],
        grid[2],
        static_tail,
        device_id,
        15 if in_aligned else 0,
        out_shape,
        launcher.run,
        launcher.runner,
        slow_call,
    )


def _launch_forward(
    kernel,
    input,
    output,
    nc,
    in_h,
    in_w,
    out_h,
    out_w,
    extra_cfg,
    memo_key=None,
    slow_call=None,
):
    in_n, in_c = input.shape[0], input.shape[1]
    S = out_h * out_w
    grid_fn = lambda meta: (
        triton.cdiv(nc, meta["BLOCK_C"]),
        triton.cdiv(S, meta["BLOCK_S"]),
        1,
    )
    extra = {
        "nc": nc,
        "in_h": in_h,
        "in_w": in_w,
        "out_h": out_h,
        "out_w": out_w,
        **extra_cfg,
    }
    args = [input, output]
    cache_key = (
        kernel,
        nc,
        input.get_device(),
        input.dtype,
        tuple(extra.items()),
        input.data_ptr() % 16 == 0,
    )
    out_shape = (in_n, in_c, out_h, out_w)

    def _make_entry(
        launcher, static_tail, grid, device_id, in_aligned, _out_aligned, slow
    ):
        return _make_forward_dispatch_entry(
            launcher,
            static_tail,
            grid,
            device_id,
            in_aligned,
            out_shape,
            slow,
        )

    _direct_launch(
        kernel,
        cache_key,
        grid_fn,
        args,
        extra,
        memo_key,
        slow_call,
        memo=_fwd_dispatch_memo,
        make_entry=_make_entry,
    )


def avg_pool2d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    """MTHREADS forward dispatch with a semantics-preserving generic fallback.

    Steady state is served through a monomorphic fast slot holding the most
    recently resolved case: every semantic parameter (shape, dtype, the raw
    pooling configuration and the device) is re-validated per call with
    direct comparisons, which is cheaper than normalizing the dispatcher's
    ``IntArrayRef`` lists into hashable tuples and probing the general memo
    dict on every hit.  Any mismatch falls back to the tuple-key
    ``_fwd_dispatch_memo`` (keyed on input device, shape, dtype and the
    normalized pooling configuration, the same contract the backward
    wrapper uses; a memo miss re-dispatches through
    ``_avg_pool2d_forward_impl``'s generic gate, and generic-fallback cases
    never register so they simply stay on that dispatch path).

    The registered entry re-creates the output via ``input.new_empty``
    with the pinned output shape (a fresh allocation every call) and
    launches the cached CompiledKernel through the raw C launcher entry,
    re-verifying pointer alignment and contiguity per call; the device is
    pinned by the validated key, so the launch guard only has to reconcile
    the current device with the operands'.
    """
    global _fwd_fast
    ent = None
    fast = _fwd_fast
    if fast is not None:
        f_shape, f_dtype, f_ks, f_st, f_pd, f_ceil, f_cip, f_div, f_dev, f_ent = fast
        if (
            input.dtype is f_dtype
            and input.shape == f_shape
            and kernel_size == f_ks
            and stride == f_st
            and padding == f_pd
            and ceil_mode == f_ceil
            and count_include_pad == f_cip
            and divisor_override == f_div
            and input.get_device() == f_dev
        ):
            ent = f_ent
    if ent is None:
        raw_kernel_size, raw_stride, raw_padding = kernel_size, stride, padding
        if type(kernel_size) is list:
            kernel_size = tuple(kernel_size)
        if type(stride) is list:
            stride = tuple(stride)
        if type(padding) is list:
            padding = tuple(padding)
        key = (
            input.get_device(),
            input.shape,
            input.dtype,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        ent = _fwd_memo_get(key)
        if ent is None:
            out = _avg_pool2d_forward_impl(
                input,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
                memo_key=key,
                slow_call=lambda inp: _avg_pool2d_forward_impl(
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                ),
            )
            # The impl registers the steady-state entry for fast-path
            # geometries; promote it into the monomorphic slot for the next
            # call (generic-fallback cases never register and keep taking
            # this dispatch path, exactly like the dict-only contract).
            ent = _fwd_memo_get(key)
            if ent is not None:
                _fwd_fast = (
                    input.shape,
                    input.dtype,
                    raw_kernel_size,
                    raw_stride,
                    raw_padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                    key[0],
                    ent,
                )
            return out
        _fwd_fast = (
            input.shape,
            input.dtype,
            raw_kernel_size,
            raw_stride,
            raw_padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            key[0],
            ent,
        )
    # Unified steady-state hit path (mirrors the backward wrapper).
    (
        c_launch,
        fn_handle,
        packed,
        g0,
        g1,
        g2,
        static_tail,
        dev_id,
        in_mask,
        out_shape,
        run,
        runner,
        slow,
    ) = ent
    output = _TENSOR_NEW_EMPTY(input, out_shape)
    in_ptr = input.data_ptr()
    if not in_ptr & in_mask and input.is_contiguous():
        hook = _RUNTIME_KNOBS.launch_enter_hook
        if hook is not None:
            try:
                calls = hook.calls
            except AttributeError:
                calls = True
            if calls:
                runner(input, output, *static_tail)
                return output
        if _get_raw_musa_device() == dev_id:
            if c_launch is None:
                run(
                    g0,
                    g1,
                    g2,
                    _get_raw_musa_stream(dev_id),
                    fn_handle,
                    packed,
                    None,
                    None,
                    None,
                    input,
                    output,
                    *static_tail,
                )
                return output
            c_launch(
                g0,
                g1,
                g2,
                _get_raw_musa_stream(dev_id),
                fn_handle,
                packed,
                None,
                None,
                None,
                in_ptr,
                output.data_ptr(),
                *static_tail,
            )
        else:
            # Current device differs from the operands': the MUSA
            # launch API requires the stream to belong to the current
            # context, so enter the tensor device's context and launch
            # on that device's current stream.
            with _device_guard(dev_id):
                if c_launch is None:
                    run(
                        g0,
                        g1,
                        g2,
                        _get_raw_musa_stream(dev_id),
                        fn_handle,
                        packed,
                        None,
                        None,
                        None,
                        input,
                        output,
                        *static_tail,
                    )
                else:
                    c_launch(
                        g0,
                        g1,
                        g2,
                        _get_raw_musa_stream(dev_id),
                        fn_handle,
                        packed,
                        None,
                        None,
                        None,
                        in_ptr,
                        output.data_ptr(),
                        *static_tail,
                    )
        return output
    return slow(input)


def _avg_pool2d_forward_impl(
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    memo_key=None,
    slow_call=None,
):
    """MTHREADS forward dispatch with a semantics-preserving generic fallback."""
    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    dt = input.dtype
    if (
        (dt is not _FLOAT32 and dt is not _FLOAT16 and dt is not _BFLOAT16)
        or input.dim() != 4
        or input.numel() > 0x7FFFFFFF
    ):
        return _generic_avg_pool2d(
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    if not input.is_contiguous():
        # Normalize to a contiguous copy so the specialized kernels (with
        # the ATen-correct ceil-tail window-length divisors) also serve
        # non-contiguous operands; the generic fallback's fixed K*K
        # divisor predates the ceil-tail shortening.
        input = input.contiguous()

    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w = _parse_pool_params(
        kernel_size, stride, padding
    )
    in_n, in_c, in_h, in_w = input.shape
    out_h = pool2d_output_size(in_h, kernel_h, stride_h, padding_h, 1, ceil_mode)
    out_w = pool2d_output_size(in_w, kernel_w, stride_w, padding_w, 1, ceil_mode)

    output = input.new_empty((in_n, in_c, out_h, out_w))
    if out_h * out_w == 0 or in_n * in_c == 0:
        return output

    nc = in_n * in_c
    if (
        kernel_h == 3
        and kernel_w == 3
        and stride_h == 2
        and stride_w == 2
        and padding_h == 1
        and padding_w == 1
        and divisor_override is None
    ):
        # count_include_pad=True without a ceil tail has the constant K*K
        # divisor; the other families need the per-axis window length.
        ceil_tail_h = ceil_mode and out_h > 0 and (out_h - 1) * 2 - 1 + 3 > in_h + 1
        ceil_tail_w = ceil_mode and out_w > 0 and (out_w - 1) * 2 - 1 + 3 > in_w + 1
        vary_div = (not count_include_pad) or ceil_tail_h or ceil_tail_w
        _launch_forward(
            _avg_pool2d_forward_k3s2p1_kernel,
            input,
            output,
            nc,
            in_h,
            in_w,
            out_h,
            out_w,
            {
                "COUNT_INCLUDE_PAD": count_include_pad,
                "VARY_DIV": vary_div,
                # The fused (x[2w], x[2w+1]) pair load is only 8B-aligned
                # (vectorizable) when in_w is even.
                "PAIR": in_w % 2 == 0,
            },
            memo_key,
            slow_call,
        )
        return output
    if (
        kernel_h == 2
        and kernel_w == 2
        and stride_h == 1
        and stride_w == 1
        and padding_h == 0
        and padding_w == 0
        and out_h == in_h - 1
        and out_w == in_w - 1
    ):
        # All taps in bounds: divisor_override or the full K*K count.
        _launch_forward(
            _avg_pool2d_forward_k2s1p0_kernel,
            input,
            output,
            nc,
            in_h,
            in_w,
            out_h,
            out_w,
            {"DIVISOR": divisor_override if divisor_override is not None else 4},
            memo_key,
            slow_call,
        )
        return output
    return _generic_avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


__all__ = ["avg_pool2d", "avg_pool2d_backward"]
