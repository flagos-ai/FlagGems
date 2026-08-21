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

"""Ascend-specific ``_unsafe_index_put`` (pure Triton).

Design notes:

- The generic implementation's non-CUDA fallback calls ``_index_put_impl_``,
  which re-enters FlagGems' own patched implementation on Ascend and produces
  wrong results / crashes. This module implements the op directly instead.
- All offset arithmetic runs in int32 (native on Ascend); calls whose tensors
  are too large for int32 offsets fall back to a native CPU run (a direct
  native re-dispatch is not possible under ``use_gems`` because it re-enters
  FlagGems' own patched ``_index_put_impl_``).
- Two Triton kernels are used:
  1. A prep kernel computes, for every broadcast index-space position, the
     flat destination offset (``self_base``) and the flat values offset
     (``val_base``). This hoists all per-position index gathering /
     coordinate decomposition out of the scatter loop.
  2. A scatter kernel over a 2D grid (index positions x suffix positions):
     for ``accumulate=False`` it stores vectorized 2D tiles; for
     ``accumulate=True`` it emits one-dimensional ``tl.atomic_add`` tiles
     (the Ascend lowering of a 2D atomic tile is substantially slower) and
     skips program 0 of axis 0, which may be re-executed during Ascend
     lowering and would double-apply atomic updates.
- ``accumulate=True`` for dtypes without a supported Ascend atomic (bf16,
  fp16 atomics are lossy, narrow ints/bool have none) accumulates in a
  widened scratch buffer (fp32 / int32) seeded with ``out.to(dtype)`` and
  written back with ``out.copy_(scratch)``: both are native (unpatched) ops
  on Ascend and give PyTorch's opmath semantics with a single rounding on
  writeback.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MAX_NDIM = 6
# int32 offset arithmetic guards (native fallback beyond these)
_MAX_NUMEL = 2**30
_MAX_IDX_NUMEL = 2**26


def _native_fallback(inp, indices, values, accumulate):
    """Native-implementation fallback for cases the Triton path cannot cover.

    On Ascend the patched dispatcher makes a direct native re-dispatch
    unreliable (it re-enters FlagGems' own ``_index_put_impl_`` python
    registration, which crashes on Ascend), so the fallback runs the op on
    CPU instead. This path is only taken for fp64 (unsupported by the Ascend
    Triton backend) and for tensors too large for int32 offsets; correctness
    is exact, performance is secondary.
    """
    logger.warning(
        "GEMS_ASCEND _UNSAFE_INDEX_PUT: native CPU fallback " "(dtype=%s, numel=%s)",
        inp.dtype,
        inp.numel(),
    )
    out = torch._unsafe_index_put(
        inp.cpu(), [i.cpu() for i in indices], values.cpu(), accumulate
    )
    return out.to(inp.device)


# Widened scratch dtype for accumulate when the output dtype has no fast /
# exact atomic add on Ascend. Mirrors PyTorch's opmath_t accumulation.
_SCRATCH_DTYPES = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.int8: torch.int32,
    torch.int16: torch.int32,
    torch.uint8: torch.int32,
    torch.bool: torch.int32,
}

# Meta layout (int32): offsets into the meta tensor.
#   [0:6]    idx_shape (padded 1)
#   [6:12]   idx_div (trailing divisors of idx_shape, padded 1)
#   [12:48]  ts (M x _MAX_NDIM broadcast strides of the index tensors)
#   [48:54]  val_adv (broadcast strides of values into idx_shape)
#   [54:60]  self_stride (out strides of the indexed dims)
#   [60:66]  self_size (out sizes of the indexed dims)
#   [66:72]  suf_shape (padded 1)
#   [72:78]  suf_div (trailing divisors of suffix_shape, padded 1)
#   [78:84]  val_suf_stride (broadcast strides of values into suffix_shape)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def unsafe_index_put_prep_kernel(
    base_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    idx3_ptr,
    idx4_ptr,
    idx5_ptr,
    meta_ptr,
    idx_numel,
    M: tl.constexpr,
    IDX_NDIM: tl.constexpr,
    BPP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Computes per-position [self_base, val_base] flat offsets into the
    # (2 * idx_numel,) int32 base buffer. All offsets are flat element offsets
    # into a contiguous output tensor, so the suffix dims only shift by the
    # flat suffix offset at scatter time.
    idx_shape0 = tl.load(meta_ptr + 0)
    idx_shape1 = tl.load(meta_ptr + 1)
    idx_shape2 = tl.load(meta_ptr + 2)
    idx_shape3 = tl.load(meta_ptr + 3)
    idx_shape4 = tl.load(meta_ptr + 4)
    idx_shape5 = tl.load(meta_ptr + 5)
    idx_div0 = tl.load(meta_ptr + 6)
    idx_div1 = tl.load(meta_ptr + 7)
    idx_div2 = tl.load(meta_ptr + 8)
    idx_div3 = tl.load(meta_ptr + 9)
    idx_div4 = tl.load(meta_ptr + 10)
    idx_div5 = tl.load(meta_ptr + 11)
    ts_0_0 = tl.load(meta_ptr + 12)
    ts_0_1 = tl.load(meta_ptr + 13)
    ts_0_2 = tl.load(meta_ptr + 14)
    ts_0_3 = tl.load(meta_ptr + 15)
    ts_0_4 = tl.load(meta_ptr + 16)
    ts_0_5 = tl.load(meta_ptr + 17)
    ts_1_0 = tl.load(meta_ptr + 18)
    ts_1_1 = tl.load(meta_ptr + 19)
    ts_1_2 = tl.load(meta_ptr + 20)
    ts_1_3 = tl.load(meta_ptr + 21)
    ts_1_4 = tl.load(meta_ptr + 22)
    ts_1_5 = tl.load(meta_ptr + 23)
    ts_2_0 = tl.load(meta_ptr + 24)
    ts_2_1 = tl.load(meta_ptr + 25)
    ts_2_2 = tl.load(meta_ptr + 26)
    ts_2_3 = tl.load(meta_ptr + 27)
    ts_2_4 = tl.load(meta_ptr + 28)
    ts_2_5 = tl.load(meta_ptr + 29)
    ts_3_0 = tl.load(meta_ptr + 30)
    ts_3_1 = tl.load(meta_ptr + 31)
    ts_3_2 = tl.load(meta_ptr + 32)
    ts_3_3 = tl.load(meta_ptr + 33)
    ts_3_4 = tl.load(meta_ptr + 34)
    ts_3_5 = tl.load(meta_ptr + 35)
    ts_4_0 = tl.load(meta_ptr + 36)
    ts_4_1 = tl.load(meta_ptr + 37)
    ts_4_2 = tl.load(meta_ptr + 38)
    ts_4_3 = tl.load(meta_ptr + 39)
    ts_4_4 = tl.load(meta_ptr + 40)
    ts_4_5 = tl.load(meta_ptr + 41)
    ts_5_0 = tl.load(meta_ptr + 42)
    ts_5_1 = tl.load(meta_ptr + 43)
    ts_5_2 = tl.load(meta_ptr + 44)
    ts_5_3 = tl.load(meta_ptr + 45)
    ts_5_4 = tl.load(meta_ptr + 46)
    ts_5_5 = tl.load(meta_ptr + 47)
    val_adv0 = tl.load(meta_ptr + 48)
    val_adv1 = tl.load(meta_ptr + 49)
    val_adv2 = tl.load(meta_ptr + 50)
    val_adv3 = tl.load(meta_ptr + 51)
    val_adv4 = tl.load(meta_ptr + 52)
    val_adv5 = tl.load(meta_ptr + 53)
    self_stride0 = tl.load(meta_ptr + 54)
    self_stride1 = tl.load(meta_ptr + 55)
    self_stride2 = tl.load(meta_ptr + 56)
    self_stride3 = tl.load(meta_ptr + 57)
    self_stride4 = tl.load(meta_ptr + 58)
    self_stride5 = tl.load(meta_ptr + 59)
    self_size0 = tl.load(meta_ptr + 60)
    self_size1 = tl.load(meta_ptr + 61)
    self_size2 = tl.load(meta_ptr + 62)
    self_size3 = tl.load(meta_ptr + 63)
    self_size4 = tl.load(meta_ptr + 64)
    self_size5 = tl.load(meta_ptr + 65)

    pid = tle.program_id(axis=0)
    for b in tl.static_range(0, BPP):
        off = (pid * BPP + b) * BLOCK + tl.arange(0, BLOCK)
        mask = off < idx_numel

        # Coordinate decomposition of the flat index-space position.
        val_base = tl.zeros((BLOCK,), dtype=tl.int32)
        if IDX_NDIM >= 1:
            c0 = (off // idx_div0) % idx_shape0
            val_base += c0 * val_adv0
        if IDX_NDIM >= 2:
            c1 = (off // idx_div1) % idx_shape1
            val_base += c1 * val_adv1
        if IDX_NDIM >= 3:
            c2 = (off // idx_div2) % idx_shape2
            val_base += c2 * val_adv2
        if IDX_NDIM >= 4:
            c3 = (off // idx_div3) % idx_shape3
            val_base += c3 * val_adv3
        if IDX_NDIM >= 5:
            c4 = (off // idx_div4) % idx_shape4
            val_base += c4 * val_adv4
        if IDX_NDIM >= 6:
            c5 = (off // idx_div5) % idx_shape5
            val_base += c5 * val_adv5

        self_base = tl.zeros((BLOCK,), dtype=tl.int32)
        for mi in tl.static_range(0, M):
            if mi == 0:
                toff = tl.zeros((BLOCK,), dtype=tl.int32)
                if IDX_NDIM >= 1:
                    toff += c0 * ts_0_0
                if IDX_NDIM >= 2:
                    toff += c1 * ts_0_1
                if IDX_NDIM >= 3:
                    toff += c2 * ts_0_2
                if IDX_NDIM >= 4:
                    toff += c3 * ts_0_3
                if IDX_NDIM >= 5:
                    toff += c4 * ts_0_4
                if IDX_NDIM >= 6:
                    toff += c5 * ts_0_5
                ind = tl.load(idx0_ptr + toff, mask=mask, other=0).to(tl.int32)
                ind = tl.where(ind < 0, ind + self_size0, ind)
                self_base += ind * self_stride0
            if mi == 1:
                toff = tl.zeros((BLOCK,), dtype=tl.int32)
                if IDX_NDIM >= 1:
                    toff += c0 * ts_1_0
                if IDX_NDIM >= 2:
                    toff += c1 * ts_1_1
                if IDX_NDIM >= 3:
                    toff += c2 * ts_1_2
                if IDX_NDIM >= 4:
                    toff += c3 * ts_1_3
                if IDX_NDIM >= 5:
                    toff += c4 * ts_1_4
                if IDX_NDIM >= 6:
                    toff += c5 * ts_1_5
                ind = tl.load(idx1_ptr + toff, mask=mask, other=0).to(tl.int32)
                ind = tl.where(ind < 0, ind + self_size1, ind)
                self_base += ind * self_stride1
            if mi == 2:
                toff = tl.zeros((BLOCK,), dtype=tl.int32)
                if IDX_NDIM >= 1:
                    toff += c0 * ts_2_0
                if IDX_NDIM >= 2:
                    toff += c1 * ts_2_1
                if IDX_NDIM >= 3:
                    toff += c2 * ts_2_2
                if IDX_NDIM >= 4:
                    toff += c3 * ts_2_3
                if IDX_NDIM >= 5:
                    toff += c4 * ts_2_4
                if IDX_NDIM >= 6:
                    toff += c5 * ts_2_5
                ind = tl.load(idx2_ptr + toff, mask=mask, other=0).to(tl.int32)
                ind = tl.where(ind < 0, ind + self_size2, ind)
                self_base += ind * self_stride2
            if mi == 3:
                toff = tl.zeros((BLOCK,), dtype=tl.int32)
                if IDX_NDIM >= 1:
                    toff += c0 * ts_3_0
                if IDX_NDIM >= 2:
                    toff += c1 * ts_3_1
                if IDX_NDIM >= 3:
                    toff += c2 * ts_3_2
                if IDX_NDIM >= 4:
                    toff += c3 * ts_3_3
                if IDX_NDIM >= 5:
                    toff += c4 * ts_3_4
                if IDX_NDIM >= 6:
                    toff += c5 * ts_3_5
                ind = tl.load(idx3_ptr + toff, mask=mask, other=0).to(tl.int32)
                ind = tl.where(ind < 0, ind + self_size3, ind)
                self_base += ind * self_stride3
            if mi == 4:
                toff = tl.zeros((BLOCK,), dtype=tl.int32)
                if IDX_NDIM >= 1:
                    toff += c0 * ts_4_0
                if IDX_NDIM >= 2:
                    toff += c1 * ts_4_1
                if IDX_NDIM >= 3:
                    toff += c2 * ts_4_2
                if IDX_NDIM >= 4:
                    toff += c3 * ts_4_3
                if IDX_NDIM >= 5:
                    toff += c4 * ts_4_4
                if IDX_NDIM >= 6:
                    toff += c5 * ts_4_5
                ind = tl.load(idx4_ptr + toff, mask=mask, other=0).to(tl.int32)
                ind = tl.where(ind < 0, ind + self_size4, ind)
                self_base += ind * self_stride4
            if mi == 5:
                toff = tl.zeros((BLOCK,), dtype=tl.int32)
                if IDX_NDIM >= 1:
                    toff += c0 * ts_5_0
                if IDX_NDIM >= 2:
                    toff += c1 * ts_5_1
                if IDX_NDIM >= 3:
                    toff += c2 * ts_5_2
                if IDX_NDIM >= 4:
                    toff += c3 * ts_5_3
                if IDX_NDIM >= 5:
                    toff += c4 * ts_5_4
                if IDX_NDIM >= 6:
                    toff += c5 * ts_5_5
                ind = tl.load(idx5_ptr + toff, mask=mask, other=0).to(tl.int32)
                ind = tl.where(ind < 0, ind + self_size5, ind)
                self_base += ind * self_stride5

        tl.store(base_ptr + off, self_base, mask=mask)
        tl.store(base_ptr + off + idx_numel, val_base, mask=mask)


@libentry()
@triton.jit
def unsafe_index_put_scatter_kernel(
    out_ptr,
    values_ptr,
    target_ptr,
    base_ptr,
    meta_ptr,
    idx_numel,
    suffix_numel,
    ACCUMULATE: tl.constexpr,
    USE_SCRATCH: tl.constexpr,
    STORE_BPP: tl.constexpr,
    IDX_PP: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
    BLOCK_SUF: tl.constexpr,
    SUF_NDIM: tl.constexpr,
):
    # Scatter over a 2D grid (index positions x suffix positions). The output
    # is contiguous, so the suffix contributes its flat offset directly; the
    # values tensor keeps its (broadcast) strides, decomposed via a small
    # div/mod chain on the suffix axis only.
    suf_shape0 = tl.load(meta_ptr + 66)
    suf_shape1 = tl.load(meta_ptr + 67)
    suf_shape2 = tl.load(meta_ptr + 68)
    suf_shape3 = tl.load(meta_ptr + 69)
    suf_shape4 = tl.load(meta_ptr + 70)
    suf_shape5 = tl.load(meta_ptr + 71)
    suf_div0 = tl.load(meta_ptr + 72)
    suf_div1 = tl.load(meta_ptr + 73)
    suf_div2 = tl.load(meta_ptr + 74)
    suf_div3 = tl.load(meta_ptr + 75)
    suf_div4 = tl.load(meta_ptr + 76)
    suf_div5 = tl.load(meta_ptr + 77)
    val_suf_stride0 = tl.load(meta_ptr + 78)
    val_suf_stride1 = tl.load(meta_ptr + 79)
    val_suf_stride2 = tl.load(meta_ptr + 80)
    val_suf_stride3 = tl.load(meta_ptr + 81)
    val_suf_stride4 = tl.load(meta_ptr + 82)
    val_suf_stride5 = tl.load(meta_ptr + 83)

    pid0 = tle.program_id(axis=0)
    pid1 = tle.program_id(axis=1)

    if ACCUMULATE:
        # 1D atomics per index position. Program 0 of axis 0 is a no-op: it
        # may be re-executed during Ascend lowering, which would double-apply
        # the atomic updates.
        suf_off = pid1 * BLOCK_SUF + tl.arange(0, BLOCK_SUF)
        mask_suf = suf_off < suffix_numel
        vs = tl.zeros((BLOCK_SUF,), dtype=tl.int32)
        if SUF_NDIM >= 1:
            cs0 = (suf_off // suf_div0) % suf_shape0
            vs += cs0 * val_suf_stride0
        if SUF_NDIM >= 2:
            cs1 = (suf_off // suf_div1) % suf_shape1
            vs += cs1 * val_suf_stride1
        if SUF_NDIM >= 3:
            cs2 = (suf_off // suf_div2) % suf_shape2
            vs += cs2 * val_suf_stride2
        if SUF_NDIM >= 4:
            cs3 = (suf_off // suf_div3) % suf_shape3
            vs += cs3 * val_suf_stride3
        if SUF_NDIM >= 5:
            cs4 = (suf_off // suf_div4) % suf_shape4
            vs += cs4 * val_suf_stride4
        if SUF_NDIM >= 6:
            cs5 = (suf_off // suf_div5) % suf_shape5
            vs += cs5 * val_suf_stride5
        for r in tl.static_range(0, IDX_PP):
            pos = (pid0 - 1) * IDX_PP + r
            ok = (pid0 > 0) & (pos >= 0) & (pos < idx_numel)
            # Program 0 of axis 0 may be re-executed during Ascend lowering;
            # masks on loads/atomics are not reliable in that re-execution, so
            # keep every address valid (clamped) and force the added value to
            # exactly zero with tl.where instead of relying on any mask.
            pos_safe = tl.maximum(tl.minimum(pos, idx_numel - 1), 0)
            self_base = tl.load(base_ptr + pos_safe)
            val_base = tl.load(base_ptr + pos_safe + idx_numel)
            row_mask = ok & mask_suf
            self_off = self_base + suf_off
            val_off = val_base + vs
            v = tl.load(values_ptr + val_off)
            if USE_SCRATCH:
                v = v.to(target_ptr.dtype.element_ty)
            v = tl.where(row_mask, v, tl.zeros((BLOCK_SUF,), dtype=v.dtype))
            tl.atomic_add(target_ptr + self_off, v, sem="relaxed")
    else:
        # Plain stores: 2D tiles are efficient and re-executing program 0 is
        # harmless (the same value is stored again).
        suf_off2d = pid1 * BLOCK_SUF + tl.arange(0, BLOCK_SUF)[None, :]
        mask_suf = suf_off2d < suffix_numel
        vs2d = tl.zeros((1, BLOCK_SUF), dtype=tl.int32)
        if SUF_NDIM >= 1:
            cs0 = (suf_off2d // suf_div0) % suf_shape0
            vs2d += cs0 * val_suf_stride0
        if SUF_NDIM >= 2:
            cs1 = (suf_off2d // suf_div1) % suf_shape1
            vs2d += cs1 * val_suf_stride1
        if SUF_NDIM >= 3:
            cs2 = (suf_off2d // suf_div2) % suf_shape2
            vs2d += cs2 * val_suf_stride2
        if SUF_NDIM >= 4:
            cs3 = (suf_off2d // suf_div3) % suf_shape3
            vs2d += cs3 * val_suf_stride3
        if SUF_NDIM >= 5:
            cs4 = (suf_off2d // suf_div4) % suf_shape4
            vs2d += cs4 * val_suf_stride4
        if SUF_NDIM >= 6:
            cs5 = (suf_off2d // suf_div5) % suf_shape5
            vs2d += cs5 * val_suf_stride5
        for b in tl.static_range(0, STORE_BPP):
            idx_off2d = (pid0 * STORE_BPP + b) * BLOCK_IDX + tl.arange(0, BLOCK_IDX)[
                :, None
            ]
            mask_idx = idx_off2d < idx_numel
            self_base = tl.load(base_ptr + idx_off2d, mask=mask_idx, other=0)
            val_base = tl.load(base_ptr + idx_off2d + idx_numel, mask=mask_idx, other=0)
            mask = mask_idx & mask_suf
            self_off = self_base + suf_off2d
            val_off = val_base + vs2d
            v = tl.load(values_ptr + val_off, mask=mask, other=0)
            tl.store(out_ptr + self_off, v, mask=mask)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _volume(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _broadcast_shapes(shapes):
    """Fast local replacement for torch.broadcast_shapes (no dispatch)."""
    ndim = max(len(s) for s in shapes)
    out = [1] * ndim
    for shape in shapes:
        pad = ndim - len(shape)
        for i, s in enumerate(shape):
            j = pad + i
            if s != 1:
                if out[j] == 1:
                    out[j] = s
                elif out[j] != s:
                    raise RuntimeError(
                        f"The size of tensor a ({out[j]}) must match the size of "
                        f"tensor b ({s}) at non-singleton dimension {j}"
                    )
    return tuple(out)


def _broadcast_strides(shape, stride, target_shape):
    """Broadcast (shape, stride) into target_shape; 0 stride on expanded dims."""
    ndim = len(target_shape)
    pad = ndim - len(shape)
    out = [0] * ndim
    for i, s in enumerate(shape):
        j = pad + i
        if s == 1 and target_shape[j] != 1:
            out[j] = 0
        elif s != target_shape[j]:
            raise RuntimeError(
                f"The expanded size of the tensor ({target_shape[j]}) must match "
                f"the existing size ({s}) at non-singleton dimension {i}."
            )
        else:
            out[j] = stride[i]
    return out


def _trailing_divisors(shape):
    ndim = len(shape)
    div = [1] * ndim
    acc = 1
    for i in range(ndim - 1, -1, -1):
        div[i] = acc
        acc *= shape[i]
    return div


def _pad(seq, n, fill):
    out = list(seq)
    out.extend([fill] * (n - len(out)))
    return out


def _next_pow2(x):
    v = 1
    while v < x:
        v *= 2
    return v


@lru_cache(maxsize=1024)
def _store_meta(
    device,
    out_shape,
    out_stride,
    idx_key,
    val_shape,
    val_stride,
    idx_shape,
    suffix_shape,
    m,
):
    """Small int32 kernel-parameter buffer, cached per exact shapes/strides."""
    tensor_strides = [
        _broadcast_strides(idx_key[i][0], idx_key[i][1], idx_shape) for i in range(m)
    ]
    ts = []
    for i in range(m):
        ts.extend(_pad(tensor_strides[i], _MAX_NDIM, 0))
    ts = _pad(ts, _MAX_NDIM * _MAX_NDIM, 0)
    val_strides = _broadcast_strides(
        val_shape, val_stride, tuple(idx_shape) + tuple(suffix_shape)
    )
    idx_ndim = len(idx_shape)
    vals = (
        list(_pad(list(idx_shape), _MAX_NDIM, 1))
        + list(_pad(_trailing_divisors(idx_shape), _MAX_NDIM, 1))
        + ts
        + list(_pad(val_strides[:idx_ndim], _MAX_NDIM, 0))
        + list(_pad(list(out_stride[:m]), _MAX_NDIM, 0))
        + list(_pad(list(out_shape[:m]), _MAX_NDIM, 1))
        + list(_pad(list(suffix_shape), _MAX_NDIM, 1))
        + list(_pad(_trailing_divisors(suffix_shape), _MAX_NDIM, 1))
        + list(_pad(val_strides[idx_ndim:], _MAX_NDIM, 0))
    )
    return torch.tensor(vals, dtype=torch.int32, device=device)


def _prep_block_config(idx_numel):
    # block >= 2: a statically-1 block size crashes the Ascend lowering
    block = max(2, min(_next_pow2(idx_numel), 1024))
    blocks = triton.cdiv(idx_numel, block)
    bpp = _next_pow2(triton.cdiv(blocks, 65535)) if blocks > 65535 else 1
    return block, bpp


def _store_block_config(idx_numel, suffix_numel):
    """(BLOCK_IDX, BLOCK_SUF, STORE_BPP) for the store scatter path."""
    # BLOCK_IDX <= 512: wider tiles overflow the unified buffer on Ascend.
    # BLOCK_IDX >= 2: a statically-1 first tile dim crashes the Ascend
    # lowering (SubViewOp offset inference).
    block_idx = max(2, min(_next_pow2(idx_numel), 512))
    block_suf = min(_next_pow2(suffix_numel), 256)
    # keep the 2D tile small enough for the unified buffer
    while block_idx * block_suf > 2048:
        if block_idx >= block_suf:
            block_idx //= 2
        else:
            block_suf //= 2
    idx_blocks = triton.cdiv(idx_numel, block_idx)
    # amortize the per-program meta loads over several idx blocks
    store_bpp = min(8, _next_pow2(triton.cdiv(idx_blocks, 1024)))
    # respect the 65535 program limit per grid axis via the block loop
    if triton.cdiv(idx_numel, block_idx * store_bpp) > 65535:
        store_bpp = _next_pow2(triton.cdiv(triton.cdiv(idx_numel, block_idx), 65535))
    if triton.cdiv(suffix_numel, block_suf) > 65535:
        block_suf = _next_pow2(triton.cdiv(suffix_numel, 65535))
        if block_idx * block_suf > 8192:
            return None
    return block_idx, block_suf, store_bpp


def _atomic_block_config(idx_numel, suffix_numel):
    """(BLOCK_SUF, IDX_PP) for the accumulate scatter path (1D atomics)."""
    block_suf = min(_next_pow2(suffix_numel), 512)
    if triton.cdiv(suffix_numel, block_suf) > 65535:
        block_suf = _next_pow2(triton.cdiv(suffix_numel, 65535))
    if block_suf > 2048:
        return None
    idx_pp = 1
    while triton.cdiv(idx_numel, idx_pp) + 1 > 65535:
        idx_pp *= 2
    if idx_pp > 16:
        return None
    return block_suf, idx_pp


def _launch_prep(out, indices, meta, idx_shape, idx_numel):
    m = len(indices)
    idx_ndim = len(idx_shape)
    kernel_idx = indices + [indices[0]] * (_MAX_NDIM - m)
    base = torch.empty(idx_numel * 2, dtype=torch.int32, device=out.device)
    block, bpp = _prep_block_config(idx_numel)
    grid = (triton.cdiv(idx_numel, block * bpp),)
    with torch_device_fn.device(out.device):
        unsafe_index_put_prep_kernel[grid](
            base,
            *kernel_idx,
            meta,
            idx_numel,
            M=m,
            IDX_NDIM=idx_ndim,
            BPP=bpp,
            BLOCK=block,
        )
    return base


def _launch_scatter(
    out, values, target, base, meta, idx_numel, suffix_numel, suf_ndim, accumulate
):
    use_scratch = target is not out
    if accumulate:
        config = _atomic_block_config(idx_numel, suffix_numel)
        if config is None:
            return False
        block_suf, idx_pp = config
        grid = (
            triton.cdiv(idx_numel, idx_pp) + 1,
            triton.cdiv(suffix_numel, block_suf),
        )
        with torch_device_fn.device(out.device):
            unsafe_index_put_scatter_kernel[grid](
                out,
                values,
                target,
                base,
                meta,
                idx_numel,
                suffix_numel,
                ACCUMULATE=True,
                USE_SCRATCH=use_scratch,
                STORE_BPP=1,
                IDX_PP=idx_pp,
                BLOCK_IDX=1,
                BLOCK_SUF=block_suf,
                SUF_NDIM=suf_ndim,
            )
    else:
        config = _store_block_config(idx_numel, suffix_numel)
        if config is None:
            return False
        block_idx, block_suf, store_bpp = config
        grid = (
            triton.cdiv(idx_numel, block_idx * store_bpp),
            triton.cdiv(suffix_numel, block_suf),
        )
        with torch_device_fn.device(out.device):
            unsafe_index_put_scatter_kernel[grid](
                out,
                values,
                target,
                base,
                meta,
                idx_numel,
                suffix_numel,
                ACCUMULATE=False,
                USE_SCRATCH=use_scratch,
                STORE_BPP=store_bpp,
                IDX_PP=1,
                BLOCK_IDX=block_idx,
                BLOCK_SUF=block_suf,
                SUF_NDIM=suf_ndim,
            )
    return True


# ---------------------------------------------------------------------------
# Operator entry point
# ---------------------------------------------------------------------------


def _unsafe_index_put(inp, indices, values, accumulate=False):
    """_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    Functional advanced indexing scatter. Returns a new tensor. Bool/byte
    masks are expanded via nonzero; each of the first ``len(indices)`` dims
    of ``self`` is indexed by the corresponding index tensor and the
    remaining dims form the value suffix. ``unsafe`` means no upper bounds
    checking is performed (negative indices wrap), matching PyTorch's
    contract. accumulate=True uses atomic_add (with a widened scratch buffer
    for dtypes without a supported Ascend atomic), so duplicate-index
    summation order is arbitrary GPU scheduling; integer accumulation is
    exact.
    """
    logger.debug("GEMS_ASCEND _UNSAFE_INDEX_PUT")

    if torch.is_tensor(indices):
        indices = [indices]
    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")

    # Device/dtype fixups mirroring aten's _index_put_impl_.
    if values.device != inp.device:
        if values.numel() == 1 and values.dim() == 0:
            values = values.to(inp.device)
        else:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but found at "
                f"least two devices, {inp.device} and {values.device}!"
            )
    if values.dtype != inp.dtype:
        raise RuntimeError(
            f"Index put requires the source and destination dtypes match, got "
            f"{inp.dtype} for the destination and {values.dtype} for the source."
        )

    # Index preprocessing: expand bool/byte masks, fix devices.
    processed = []
    for idx in indices:
        if idx is None:
            raise TypeError(
                "_unsafe_index_put does not accept None indices "
                "(expected Tensor, but got NoneType)"
            )
        if idx.device != inp.device:
            idx = idx.to(inp.device)
        if idx.dtype in (torch.bool, torch.uint8):
            processed.extend(idx.nonzero(as_tuple=True))
        elif idx.dtype in (torch.int32, torch.int64):
            processed.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    m = len(processed)
    if m > inp.dim():
        raise IndexError(f"too many indices for tensor of dimension {inp.dim()}")
    if m > _MAX_NDIM:
        raise IndexError(f"too many index tensors (max {_MAX_NDIM})")

    suffix_shape = inp.shape[m:]
    suf_ndim = len(suffix_shape)
    if suf_ndim > _MAX_NDIM:
        raise IndexError(f"suffix ndim out of range: {suf_ndim}")

    idx_shape = _broadcast_shapes([t.shape for t in processed])
    idx_ndim = len(idx_shape)
    if idx_ndim > _MAX_NDIM:
        raise IndexError(f"index space rank too large: {idx_ndim}")

    idx_numel = _volume(idx_shape)
    suffix_numel = _volume(suffix_shape)

    # Cases outside the int32-offset Triton path (fp64 is not supported by
    # the Ascend Triton backend at all): fall back to a CPU-native run.
    if (
        inp.dtype == torch.float64
        or inp.numel() > _MAX_NUMEL
        or idx_numel > _MAX_IDX_NUMEL
        or suffix_numel > _MAX_NUMEL
    ):
        return _native_fallback(inp, indices, values, accumulate)

    # Functional contract: the input is never modified. The scatter kernels
    # require a contiguous output; clone preserves the input layout.
    out = inp.clone()
    if not out.is_contiguous():
        out = out.contiguous()
    if not values.is_contiguous():
        values = values.contiguous()

    if idx_numel == 0 or suffix_numel == 0:
        return out

    use_scratch = accumulate and out.dtype in _SCRATCH_DTYPES
    if use_scratch:
        # Native (unpatched on Ascend) full-tensor cast seeds the widened
        # scratch; the epilogue cast-back rounds each element exactly once.
        scratch = out.to(_SCRATCH_DTYPES[out.dtype])
        target = scratch
    else:
        target = out

    meta = _store_meta(
        str(out.device),
        tuple(out.shape),
        tuple(out.stride()),
        tuple((tuple(t.shape), tuple(t.stride())) for t in processed),
        tuple(values.shape),
        tuple(values.stride()),
        tuple(idx_shape),
        tuple(suffix_shape),
        m,
    )

    base = _launch_prep(out, processed, meta, idx_shape, idx_numel)

    if not _launch_scatter(
        out, values, target, base, meta, idx_numel, suffix_numel, suf_ndim, accumulate
    ):
        # Extreme shapes beyond the Triton path: native fallback.
        return _native_fallback(inp, indices, values, accumulate)

    if use_scratch:
        out.copy_(scratch)

    return out
