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
"""Kunlunxin XPU3 override for ``_upsample_lanczos2d_aa`` (and out/vec).

Why this backend-local copy exists (root cause found on XPU3):

* The generic implementation runs the tap loop as a *runtime* ``for`` loop with
  a 256-lane tile. On this toolchain that aborts ``TritonXPUUnrollControl``
  (``arith.addi`` type mismatch / uni_sram), so every case fails to compile.

This implementation keeps the same ATen-visible semantics (Lanczos-3
antialiased separable resampling, weights normalized per output index) but:
  * uses a flat 1-D tile with a ``tl.static_range`` tap loop (compiles);
  * always precomputes the 1-D weights (cached host-side per geometry) in a
    ``[tap, out]`` layout so per-tap weight loads are contiguous;
  * clamps every sample address (no OOB store/gather) and zeroes out-of-window
    weights at precompute time, so the hot loop is mask-free when the flat
    extent is a multiple of the tile.
"""

import logging
import math
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

logger = logging.getLogger(__name__)
device = device.name

_CACHE_MAX_ENTRIES = 256
_WEIGHT_CACHE = {}

# Dynamic-loop tap counts above this are handled by a narrow-tile variant
# (dynamic loops only compile with <= 128 lanes on this toolchain).
_STATIC_TAP_CAP = 21


@triton.jit
def _lanczos3(x):
    abs_x = tl.abs(x)
    pix = x * 3.141592653589793
    pix_over_three = pix / 3.0
    sinc_x = tl.where(abs_x == 0.0, 1.0, tl.sin(pix) / pix)
    sinc_x_over_three = tl.where(
        abs_x == 0.0, 1.0, tl.sin(pix_over_three) / pix_over_three
    )
    return tl.where(abs_x < 3.0, sinc_x * sinc_x_over_three, 0.0)


@triton.jit
def _lanczos_dense_kernel(
    wmat,  # [out, in] row-major
    output_size,
    input_size,
    SCALE,
    SUPPORT,
    INVSCALE,
    BLOCK_IN: tl.constexpr,
):
    """One program per output row: build the dense normalized [out, in] weight row.

    The separable resample is a rank-1 2-D filter, so the whole forward is
    ``Wh @ in @ Ww^T`` -- two GEMMs on the vendor tensor cores.  This table is
    the same normalized Lanczos-3 weight as the sparse tap table, just laid out
    densely so it can feed ``tl.dot``.
    """
    o = tl.program_id(0)
    i = tl.arange(0, BLOCK_IN)
    mask = i < input_size
    center = SCALE * (o.to(tl.float32) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int32), 0)
    index_max = tl.minimum((center + SUPPORT + 0.5).to(tl.int32), input_size)
    arg = (i.to(tl.float32) - center + 0.5) * INVSCALE
    wv = _lanczos3(arg)
    wv = tl.where(mask & (i >= index_min) & (i < index_max), wv, 0.0)
    total = tl.sum(wv, axis=0)
    total = tl.where(total != 0.0, total, 1.0)
    tl.store(
        wmat + o * input_size + i,
        (wv / total).to(wmat.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gemm_h_kernel(
    src,
    wt,
    dst,
    M,
    IW: tl.constexpr,
    OW: tl.constexpr,
    NPAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Horizontal pass ``out[m, n] = sum_k src[m, k] * wt[n, k]`` (NT GEMM).

    ``src`` is the flat ``(M, IW)`` input, ``wt`` the dense ``(OW, IW)`` weight
    matrix, ``dst`` an over-allocated ``(grid_m*BM, NPAD)`` buffer (row padding,
    column padding) so the store is fully affine and unmasked.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, IW, BK):
        kk = k0 + tl.arange(0, BK)
        if EVEN_M:
            if EVEN_K:
                a = tl.load(src + rm[:, None] * IW + kk[None, :])
            else:
                a = tl.load(
                    src + rm[:, None] * IW + kk[None, :],
                    mask=(kk[None, :] < IW),
                    other=0.0,
                )
        else:
            if EVEN_K:
                a = tl.load(
                    src + rm[:, None] * IW + kk[None, :],
                    mask=(rm[:, None] < M),
                    other=0.0,
                )
            else:
                a = tl.load(
                    src + rm[:, None] * IW + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < IW),
                    other=0.0,
                )
        b = tl.load(
            wt + rn[:, None] * IW + kk[None, :],
            mask=(rn[:, None] < OW) & (kk[None, :] < IW),
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
    tl.store(dst + rm[:, None] * NPAD + rn[None, :], acc)


@triton.jit
def _gemm_v_kernel(
    v1,
    wh,
    out,
    IH: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    NPAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Vertical pass, batched over ``nc``: ``out[nc,oh,ow] = sum_ih wh[oh,ih] * v1[nc,ih,ow]``.

    NN GEMM per (nc, oh-tile, ow-tile).  ``wh`` is the dense ``(OH, IH)`` weight
    matrix (load-rooted, affine).  Row/column tails clamp instead of masking so
    every lane keeps an in-bounds idempotent write on the final output.
    """
    pz = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    oh_raw = pid_m * BM + tl.arange(0, BM)
    rn_raw = pid_n * BN + tl.arange(0, BN)
    if EVEN_ROWS:
        oh = oh_raw
    else:
        oh = tl.minimum(oh_raw, OH - 1)
    if EVEN_COLS:
        rn = rn_raw
    else:
        rn = tl.minimum(rn_raw, OW - 1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    base = pz * IH
    for k0 in range(0, IH, BK):
        kk = k0 + tl.arange(0, BK)
        if EVEN_K:
            a = tl.load(wh + oh[:, None] * IH + kk[None, :])
            b = tl.load(v1 + (base + kk)[:, None] * NPAD + rn[None, :])
        else:
            a = tl.load(
                wh + oh[:, None] * IH + kk[None, :],
                mask=(kk[None, :] < IH),
                other=0.0,
            )
            b = tl.load(
                v1 + (base + kk)[:, None] * NPAD + rn[None, :],
                mask=(kk[:, None] < IH),
                other=0.0,
            )
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(out + (pz * OH + oh)[:, None] * OW + rn[None, :], acc)


@triton.jit
def _clamp_u8(acc, IS_UINT8: tl.constexpr):
    if IS_UINT8:
        acc = tl.floor(tl.minimum(tl.maximum(acc, 0.0), 255.0) + 0.5)
    return acc


@triton.jit
def _h_kernel_static(
    src,
    dst,
    weights,
    index_mins,
    numel,
    IW: tl.constexpr,
    OW: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        ow = offs % OW
        row = offs // OW
        index_min = tl.load(index_mins + ow)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for tap in tl.static_range(MAX_TAPS):
            w = tl.load(weights + tap * OW + ow)
            ii = tl.minimum(index_min + tap, IW - 1)
            s = tl.load(src + row * IW + ii).to(tl.float32)
            acc += s * w
        acc = _clamp_u8(acc, IS_UINT8)
        tl.store(dst + offs, acc)
    else:
        mask = offs < numel
        offs_safe = tl.where(mask, offs, 0)
        ow = offs_safe % OW
        row = offs_safe // OW
        index_min = tl.load(index_mins + ow, mask=mask)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for tap in tl.static_range(MAX_TAPS):
            w = tl.load(weights + tap * OW + ow, mask=mask)
            ii = tl.minimum(index_min + tap, IW - 1)
            s = tl.load(src + row * IW + ii, mask=mask).to(tl.float32)
            acc += s * w
        acc = _clamp_u8(acc, IS_UINT8)
        tl.store(dst + offs_safe, acc, mask=mask)


@triton.jit
def _v_kernel_static(
    src,
    dst,
    weights,
    index_mins,
    numel,
    IH: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        ow = offs % OW
        oh = (offs // OW) % OH
        nc = offs // (OH * OW)
        index_min = tl.load(index_mins + oh)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for tap in tl.static_range(MAX_TAPS):
            w = tl.load(weights + tap * OH + oh)
            ii = tl.minimum(index_min + tap, IH - 1)
            s = tl.load(src + (nc * IH + ii) * OW + ow).to(tl.float32)
            acc += s * w
        acc = _clamp_u8(acc, IS_UINT8)
        tl.store(dst + offs, acc)
    else:
        mask = offs < numel
        offs_safe = tl.where(mask, offs, 0)
        ow = offs_safe % OW
        oh = (offs_safe // OW) % OH
        nc = offs_safe // (OH * OW)
        index_min = tl.load(index_mins + oh, mask=mask)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for tap in tl.static_range(MAX_TAPS):
            w = tl.load(weights + tap * OH + oh, mask=mask)
            ii = tl.minimum(index_min + tap, IH - 1)
            s = tl.load(src + (nc * IH + ii) * OW + ow, mask=mask).to(tl.float32)
            acc += s * w
        acc = _clamp_u8(acc, IS_UINT8)
        tl.store(dst + offs_safe, acc, mask=mask)


@triton.jit
def _lanczos_weights_kernel(
    weights,  # [MAX_TAPS, output_size] (transposed, contiguous per tap)
    index_mins,  # [output_size] int32
    index_sizes,  # [output_size] int32
    output_size,
    input_size,
    SCALE,
    SUPPORT,
    INVSCALE,
    MAX_TAPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < output_size
    center = SCALE * (offs.to(tl.float32) + 0.5)
    index_min = tl.maximum((center - SUPPORT + 0.5).to(tl.int32), 0)
    index_max = tl.minimum((center + SUPPORT + 0.5).to(tl.int32), input_size)
    index_size = tl.minimum(tl.maximum(index_max - index_min, 0), MAX_TAPS)
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in tl.static_range(MAX_TAPS):
        arg = (index_min.to(tl.float32) - center + (0.5 + tap)) * INVSCALE
        w = _lanczos3(arg)
        w = tl.where(tap < index_size, w, 0.0)
        total += w
    total = tl.where(total != 0.0, total, 1.0)
    for tap in tl.static_range(MAX_TAPS):
        arg = (index_min.to(tl.float32) - center + (0.5 + tap)) * INVSCALE
        w = _lanczos3(arg)
        w = tl.where(tap < index_size, w, 0.0)
        tl.store(weights + tap * output_size + offs, w / total, mask=mask)
    tl.store(index_mins + offs, index_min, mask=mask)
    tl.store(index_sizes + offs, index_size, mask=mask)


@triton.jit
def _h_kernel_dyn(
    src,
    dst,
    weights,
    index_mins,
    numel,
    IW,
    OW,
    MAX_TAPS,
    BLOCK: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    offs_safe = tl.where(mask, offs, 0)
    ow = offs_safe % OW
    row = offs_safe // OW
    index_min = tl.load(index_mins + ow, mask=mask)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in range(0, MAX_TAPS):
        w = tl.load(weights + tap * OW + ow, mask=mask)
        ii = tl.minimum(index_min + tap, IW - 1)
        s = tl.load(src + row * IW + ii, mask=mask).to(tl.float32)
        acc += s * w
    acc = _clamp_u8(acc, IS_UINT8)
    tl.store(dst + offs_safe, acc, mask=mask)


@triton.jit
def _v_kernel_dyn(
    src,
    dst,
    weights,
    index_mins,
    numel,
    IH,
    OH,
    OW,
    MAX_TAPS,
    BLOCK: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    offs_safe = tl.where(mask, offs, 0)
    ow = offs_safe % OW
    oh = (offs_safe // OW) % OH
    nc = offs_safe // (OH * OW)
    index_min = tl.load(index_mins + oh, mask=mask)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in range(0, MAX_TAPS):
        w = tl.load(weights + tap * OH + oh, mask=mask)
        ii = tl.minimum(index_min + tap, IH - 1)
        s = tl.load(src + (nc * IH + ii) * OW + ow, mask=mask).to(tl.float32)
        acc += s * w
    acc = _clamp_u8(acc, IS_UINT8)
    tl.store(dst + offs_safe, acc, mask=mask)


def _reciprocal_scale(
    input_size: int,
    output_size: int,
    align_corners: bool,
    scale: Optional[float],
) -> float:
    if align_corners:
        return (input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    if scale is not None and scale > 0.0:
        return 1.0 / scale
    return input_size / output_size


def _support_and_taps(scale: float) -> Tuple[float, int]:
    support = 3.0 * scale if scale >= 1.0 else 3.0
    return support, math.ceil(support) * 2 + 1


def _validate(input: torch.Tensor, output_size: Sequence[int]) -> Tuple[int, int]:
    if input.device.type != device:
        raise RuntimeError(f"Expected input on {device}, but got {input.device.type}")
    if input.ndim != 4:
        raise RuntimeError(
            f"It is expected input_size equals to 4, but got size {input.ndim}"
        )
    if len(output_size) != 2:
        raise RuntimeError(
            f"It is expected output_size equals to 2, but got size {len(output_size)}"
        )
    if not input.is_floating_point() and input.dtype != torch.uint8:
        raise RuntimeError(
            f'"compute_index_ranges_weights" not implemented for {input.dtype}'
        )
    output_h, output_w = int(output_size[0]), int(output_size[1])
    if min(input.shape[-2], input.shape[-1], output_h, output_w) <= 0:
        raise RuntimeError("Input and output sizes should be greater than 0")
    if input.shape[1] == 0:
        raise RuntimeError("Non-empty 4D data tensor expected")
    return output_h, output_w


_DENSE_CACHE = {}


def _make_dense_weights(
    input_size: int,
    output_size: int,
    scale: float,
    support: float,
    dtype: torch.dtype,
    device_: torch.device,
):
    key = (
        input_size,
        output_size,
        float(scale),
        float(support),
        str(dtype),
        str(device_),
    )
    cached = _DENSE_CACHE.get(key)
    if cached is not None:
        return cached
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    block_in = max(16, triton.next_power_of_2(input_size))
    wmat = torch.empty((output_size, input_size), device=device_, dtype=dtype)
    with torch_device_fn.device(device_):
        _lanczos_dense_kernel[(output_size,)](
            wmat,
            output_size,
            input_size,
            scale,
            support,
            invscale,
            BLOCK_IN=block_in,
        )
    if len(_DENSE_CACHE) >= _CACHE_MAX_ENTRIES:
        _DENSE_CACHE.pop(next(iter(_DENSE_CACHE)))
    _DENSE_CACHE[key] = wmat
    return wmat


def _pick_tile(divisor: int, cap: int = 128, floor: int = 16) -> Tuple[int, bool]:
    """Largest power-of-two block in ``[floor, cap]`` dividing ``divisor``.

    Returns ``(block, even)``.  When no such block exists the largest one <= cap
    is returned with ``even=False`` (the kernel masks/clamps the tail).
    """
    for b in (128, 64, 32, 16):
        if b > cap:
            continue
        if b >= floor and divisor % b == 0:
            return b, True
    b = cap
    while b > floor and b > 16:
        b //= 2
    return max(b, 16), False


def _gemm_forward(
    input: torch.Tensor,
    output_h: int,
    output_w: int,
    scale_h: float,
    scale_w: float,
    support_h: float,
    support_w: float,
):
    """Tensor-core separable resample for floating dtypes.

    Returns ``None`` when the geometry cannot be expressed as two clean GEMMs
    (the caller then uses the elementwise gather path).
    """
    n, c, input_h, input_w = input.shape
    dtype = input.dtype
    dev = input.device
    M = n * c * input_h
    NC = n * c
    do_h = input_w != output_w
    do_v = input_h != output_h
    if not do_h and not do_v:
        return input.clone()

    # H is the *final* pass whenever the vertical axis is identity: its output
    # must be the exact contiguous result, so require clean row/col tiling.
    h_final = do_h and not do_v
    if h_final:
        bm, even_m = _pick_tile(M, cap=128)
        bn, even_n = _pick_tile(output_w, cap=64)
        if not (even_m and even_n):
            return None
    else:
        bm, _ = _pick_tile(M, cap=128)
        bn, _ = _pick_tile(output_w, cap=64)
    bk = 64 if input_w % 64 == 0 else (32 if input_w % 32 == 0 else 16)
    even_k = input_w % bk == 0

    with torch_device_fn.device(dev):
        if do_h:
            wmat_w = _make_dense_weights(
                input_w, output_w, scale_w, support_w, dtype, dev
            )
            if h_final:
                dst = torch.empty((M, output_w), device=dev, dtype=dtype)
                grid = (triton.cdiv(M, bm), triton.cdiv(output_w, bn))
                _gemm_h_kernel[grid](
                    input,
                    wmat_w,
                    dst,
                    M,
                    IW=input_w,
                    OW=output_w,
                    NPAD=output_w,
                    BM=bm,
                    BN=bn,
                    BK=bk,
                    EVEN_M=True,
                    EVEN_K=even_k,
                )
                return dst.view(n, c, output_h, output_w)
            grid_n = triton.cdiv(output_w, bn)
            npad = grid_n * bn
            grid_m = triton.cdiv(M, bm)
            v1 = torch.empty((grid_m * bm, npad), device=dev, dtype=dtype)
            even_m = M % bm == 0
            _gemm_h_kernel[(grid_m, grid_n)](
                input,
                wmat_w,
                v1,
                M,
                IW=input_w,
                OW=output_w,
                NPAD=npad,
                BM=bm,
                BN=bn,
                BK=bk,
                EVEN_M=even_m,
                EVEN_K=even_k,
            )
            npad_w = npad
            v1_rows = grid_m * bm
        else:
            v1 = input
            npad_w = input_w
            v1_rows = M

        if do_v:
            wmat_h = _make_dense_weights(
                input_h, output_h, scale_h, support_h, dtype, dev
            )
            bmv, even_rows = _pick_tile(output_h, cap=64)
            bnv, even_cols = _pick_tile(output_w, cap=64)
            bkv = 64 if input_h % 64 == 0 else (32 if input_h % 32 == 0 else 16)
            even_kv = input_h % bkv == 0
            out = torch.empty((NC * output_h, output_w), device=dev, dtype=dtype)
            grid = (
                NC,
                triton.cdiv(output_h, bmv),
                triton.cdiv(output_w, bnv),
            )
            _gemm_v_kernel[grid](
                v1,
                wmat_h,
                out,
                IH=input_h,
                OH=output_h,
                OW=output_w,
                NPAD=npad_w,
                BM=bmv,
                BN=bnv,
                BK=bkv,
                EVEN_ROWS=even_rows,
                EVEN_COLS=even_cols,
                EVEN_K=even_kv,
            )
            return out.view(n, c, output_h, output_w)

        # No vertical pass: reshape the padded H buffer (guarded by h_final).
        return v1.view(n, c, output_h, output_w)


def _make_weights(
    input_size: int,
    output_size: int,
    scale: float,
    support: float,
    taps: int,
    device_: torch.device,
):
    key = (
        input_size,
        output_size,
        float(scale),
        float(support),
        int(taps),
        str(device_),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weights = torch.empty((taps, output_size), device=device_, dtype=torch.float32)
    index_mins = torch.empty(output_size, device=device_, dtype=torch.int32)
    index_sizes = torch.empty(output_size, device=device_, dtype=torch.int32)
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    block = 128
    _lanczos_weights_kernel[(triton.cdiv(output_size, block),)](
        weights,
        index_mins,
        index_sizes,
        output_size,
        input_size,
        scale,
        support,
        invscale,
        MAX_TAPS=taps,
        BLOCK=block,
    )
    if len(_WEIGHT_CACHE) >= _CACHE_MAX_ENTRIES:
        _WEIGHT_CACHE.pop(next(iter(_WEIGHT_CACHE)))
    _WEIGHT_CACHE[key] = (weights, index_mins, index_sizes)
    return weights, index_mins, index_sizes


def _pick_block(numel: int) -> int:
    if numel <= 1 << 16:
        return 512
    if numel <= 1 << 20:
        return 1024
    return 2048


def _launch_1d(kind, src, dst, weights, index_mins, numel, taps, consts, is_uint8):
    block = _pick_block(numel)
    even = (numel % block) == 0
    wide = numel >= (1 << 31)
    grid = (triton.cdiv(numel, block),)
    if taps <= _STATIC_TAP_CAP:
        if kind == "h":
            _h_kernel_static[grid](
                src,
                dst,
                weights,
                index_mins,
                numel,
                IW=consts["IW"],
                OW=consts["OW"],
                MAX_TAPS=taps,
                BLOCK=block,
                EVEN=even,
                WIDE=wide,
                IS_UINT8=is_uint8,
            )
        else:
            _v_kernel_static[grid](
                src,
                dst,
                weights,
                index_mins,
                numel,
                IH=consts["IH"],
                OH=consts["OH"],
                OW=consts["OW"],
                MAX_TAPS=taps,
                BLOCK=block,
                EVEN=even,
                WIDE=wide,
                IS_UINT8=is_uint8,
            )
    else:
        if kind == "h":
            _h_kernel_dyn[grid](
                src,
                dst,
                weights,
                index_mins,
                numel,
                consts["IW"],
                consts["OW"],
                taps,
                BLOCK=128,
                IS_UINT8=is_uint8,
            )
        else:
            _v_kernel_dyn[grid](
                src,
                dst,
                weights,
                index_mins,
                numel,
                consts["IH"],
                consts["OH"],
                consts["OW"],
                taps,
                BLOCK=128,
                IS_UINT8=is_uint8,
            )


def _upsample_lanczos2d_aa_contiguous(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scales_h: Optional[float],
    scales_w: Optional[float],
) -> torch.Tensor:
    n, c, input_h, input_w = input.shape
    output_h, output_w = output_size
    if input_h == output_h and input_w == output_w:
        return input.clone()

    scale_w = _reciprocal_scale(input_w, output_w, align_corners, scales_w)
    scale_h = _reciprocal_scale(input_h, output_h, align_corners, scales_h)
    support_w, taps_w = _support_and_taps(scale_w)
    support_h, taps_h = _support_and_taps(scale_h)
    is_uint8 = input.dtype == torch.uint8

    # fp16/fp32 take the tensor-core separable GEMM path (two ``tl.dot``
    # passes).  bf16 stays on the elementwise gather path: bf16 ``tl.dot`` with
    # ``tl.trans`` + ``input_precision="ieee"`` faulted the device (status=700 /
    # KL_XID_KERNEL_EXCEPTION) in batch18, and the UPCAST-to-fp32 variant that was
    # safe on the power-of-2 benchmark geometries still faults for bf16 on odd
    # geometries whose K axis is not a multiple of BK (the ``EVEN_K=False``
    # masked-load branch feeding ``tl.dot`` -- see batch19 notes).  Re-enable only
    # after a mask-free (zero-padded K) bf16 formulation is proven safe.
    # The GEMM path falls back to elementwise for geometries that cannot be
    # expressed as clean GEMMs (e.g. an identity vertical axis whose exact output
    # layout would need a copy).
    if input.dtype in (torch.float16, torch.float32):
        gemm_out = _gemm_forward(
            input, output_h, output_w, scale_h, scale_w, support_h, support_w
        )
        if gemm_out is not None:
            return gemm_out

    with torch_device_fn.device(input.device):
        if input_w != output_w:
            weights_w, imins_w, _ = _make_weights(
                input_w, output_w, scale_w, support_w, taps_w, input.device
            )
            horizontal = torch.empty(
                (n, c, input_h, output_w), device=input.device, dtype=input.dtype
            )
            _launch_1d(
                "h",
                input,
                horizontal,
                weights_w,
                imins_w,
                horizontal.numel(),
                taps_w,
                {"IW": input_w, "OW": output_w},
                is_uint8,
            )
        else:
            horizontal = input

        if input_h != output_h:
            weights_h, imins_h, _ = _make_weights(
                input_h, output_h, scale_h, support_h, taps_h, input.device
            )
            output = torch.empty(
                (n, c, output_h, output_w), device=input.device, dtype=input.dtype
            )
            _launch_1d(
                "v",
                horizontal,
                output,
                weights_h,
                imins_h,
                output.numel(),
                taps_h,
                {"IH": input_h, "OH": output_h, "OW": output_w},
                is_uint8,
            )
        else:
            output = horizontal
    return output


def _upsample_lanczos2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LANCZOS2D_AA")
    output_size = _validate(input, output_size)
    channels_last = input.is_contiguous(memory_format=torch.channels_last)
    if input.shape[0] == 0:
        output = torch.empty(
            (*input.shape[:2], *output_size), device=input.device, dtype=input.dtype
        )
    else:
        output = _upsample_lanczos2d_aa_contiguous(
            input.contiguous(),
            output_size,
            align_corners,
            scales_h,
            scales_w,
        )
    if channels_last:
        output = output.contiguous(memory_format=torch.channels_last)
    return output


def _upsample_lanczos2d_aa_out(
    input: torch.Tensor,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    if out.device != input.device:
        raise RuntimeError(
            f"Expected out tensor on {input.device}, but got {out.device}"
        )
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype}"
        )
    result = _upsample_lanczos2d_aa(
        input, output_size, align_corners, scales_h, scales_w
    )
    out.resize_(result.shape)
    out.copy_(result)
    return out


def _upsample_lanczos2d_aa_vec(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    if (output_size is None) == (scale_factors is None):
        raise RuntimeError("Must specify exactly one of output_size and scale_factors")
    if output_size is not None:
        return _upsample_lanczos2d_aa(input, tuple(output_size), align_corners)
    if len(scale_factors) != 2:
        raise RuntimeError("scale_factors must have two elements")
    output_size = (
        int(input.shape[-2] * scale_factors[0]),
        int(input.shape[-1] * scale_factors[1]),
    )
    return _upsample_lanczos2d_aa(
        input,
        output_size,
        align_corners,
        float(scale_factors[0]),
        float(scale_factors[1]),
    )
