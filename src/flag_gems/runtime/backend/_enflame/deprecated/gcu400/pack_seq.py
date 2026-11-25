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

from __future__ import annotations

import logging
import math
import struct

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# L600 (GCU400): 48 SPs; a program whose DSM footprint stays <= ~448KB is
# co-resident as dual program per SP, so at most 48 balanced one-chunk
# programs keep every SP busy for any input size.
_NUM_SPC = 48
# Wide flat copy op width in lanes (blessed 1D streaming form, mirrors the
# unpack_seq kernel): rescan on the decode shapes picked 16384 lanes over
# 32768 for every dtype; 16384 lanes of fp32 keep the per-program DSM peak
# (load + store staging) at 128KB, far below the 448KB dual-program budget.
_COPY_BLOCK = 16384
_NUM_STAGES = 2
# Host-side memo for triton.next_power_of_2(B): B is a host integer, so the
# memo is a pure function cache with no tensor-state involvement.
_NEXT_POW2: dict = {}

# torch fp8 dtype -> (mant_bits, exp_bits, bias, max_e_field, max_m_at_top)
# for the pad encoder; max_* delimit the largest finite encoding.
_FP8_ELEMENT_FORMATS: dict = {}
for _dt, _fmt in (
    (getattr(torch, "float8_e4m3fn", None), (3, 4, 7, 15, 6)),
    (getattr(torch, "float8_e5m2", None), (2, 5, 15, 30, 3)),
):
    if _dt is not None:
        _FP8_ELEMENT_FORMATS[_dt] = _fmt
del _dt, _fmt


def _fp8_bits_from_value(
    value: float,
    mant_bits: int,
    exp_bits: int,
    bias: int,
    max_e_field: int,
    max_m_at_top: int,
) -> int:
    # GCU fp8 payloads stay on a uint8 view: it keeps the pad byte exact
    # (encoded on the host with pure bit math, no torch ops) and does not
    # rely on the backend's fp32->fp8 store lowering, which saturates
    # differently than CUDA (GCU maps -inf to NaN 0x7F instead of the
    # cvt.rn.satfinite -448 = 0xFE the generic CUDA path and the fp8 tests
    # expect).
    try:
        raw = struct.pack("<f", float(value))
    except (OverflowError, ValueError):
        raw = struct.pack("<f", math.inf if value > 0 else -math.inf)
    fbits = struct.unpack("<I", raw)[0]
    sign = (fbits >> 24) & 0x80
    f32 = struct.unpack("<f", raw)[0]
    if math.isnan(f32):
        return 0x7F if exp_bits == 4 else 0x7E
    a = abs(f32)
    if a == 0.0:
        return sign
    if math.isinf(a):
        e_field, m_field = max_e_field, max_m_at_top
        return sign | (e_field << mant_bits) | m_field
    min_norm = 2.0 ** (1 - bias)
    if a < min_norm:
        # subnormal range: uniform grid of 2^(1-bias-mant_bits)
        ri = round(a / 2.0 ** (1 - bias - mant_bits))
        if ri >= (1 << mant_bits):
            e_field, m_field = 1, 0
        else:
            e_field, m_field = 0, ri
    else:
        # normal range: quantize the mantissa at the local exponent spacing
        e = max(math.floor(math.log2(a)), 1 - bias)
        q = round(a / 2.0 ** (e - mant_bits))
        if q == (1 << (mant_bits + 1)):
            q >>= 1
            e += 1
        e_field = e + bias
        m_field = q - (1 << mant_bits)
        if e_field > max_e_field or (e_field == max_e_field and m_field > max_m_at_top):
            e_field, m_field = max_e_field, max_m_at_top
    return sign | (e_field << mant_bits) | m_field


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D] (uint8 view for 1-byte element paths)
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    rows_per_chunk,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_B: tl.constexpr,  # rounded-up batch entries for prefix sum
    BLOCK: tl.constexpr,  # flat elements per maskless load/store
    WIDE: tl.constexpr,
):
    # One program per contiguous output-row chunk of the flat [B*Lmax, D]
    # row space (the 2D [BLOCK_T, BLOCK_D] masked tile form and any
    # fp8-typed access fail to compile on the enflame toolchain; flat 1D
    # accesses are the blessed form). The host ceil-divides the rows into
    # <= NUM_SPC balanced chunks, so program pid owns exactly chunk pid.
    # Within a chunk, batch rows are moved as flat runs:
    #   - valid rows  [t, min(len_b, Lmax)): src and dst are both contiguous
    #     -> maskless full blocks + a single masked tail block
    #   - pad rows    [len_b, Lmax): constant fill, no loads at all
    #   - a full batch (len_b == Lmax) is source-contiguous with every
    #     following full batch, so consecutive full batches merge into one
    #     long flat run (bounded by the chunk's row budget)
    chunk = tl.program_id(0)
    TR = B * Lmax
    off_b = tl.arange(0, BLOCK_B)
    if PAD_IS_UINT8:
        pad_vec = tl.full((BLOCK,), PAD_VALUE, tl.uint8)
    else:
        pad_vec = tl.full((BLOCK,), PAD_VALUE, tl.float32)
    # A chunk covers a contiguous output-row range [r0, r_end); one
    # masked prefix load yields the packed source row of the chunk start.
    r0 = chunk * rows_per_chunk
    r_end = tl.minimum(r0 + rows_per_chunk, TR)
    b = r0 // Lmax
    t = r0 - b * Lmax
    prev = tl.load(lengths_ptr + off_b, mask=off_b < b, other=0)
    in_row = tl.sum(prev, axis=0) + t
    r = r0
    while r < r_end:
        len_b = tl.load(lengths_ptr + b)
        copy_rows = tl.minimum(len_b, Lmax) - t
        if copy_rows > 0:
            run = copy_rows
            if len_b == Lmax:
                # Batch b is completely full, so its remaining rows are
                # source-contiguous with every following full batch:
                # merge them into one flat run. The peek is bounded by
                # the chunk's row budget — runs are clamped to the chunk
                # anyway, so scanning past it is wasted work.
                nb = b + 1
                go = 1
                budget = r_end - r
                while (nb < B) & (go == 1) & (copy_rows + (nb - b) * Lmax <= budget):
                    len_nb = tl.load(lengths_ptr + nb)
                    go = tl.where(len_nb == Lmax, 1, 0)
                    nb += go
                run = copy_rows + (nb - b - 1) * Lmax
            run = tl.minimum(run, r_end - r)
            if WIDE:
                n_elem = run.to(tl.int64) * D
                src = in_row.to(tl.int64) * D
                dst = r.to(tl.int64) * D
            else:
                n_elem = run * D
                src = in_row * D
                dst = r * D
            # Block count always fits i32: >= 2^31 blocks would need a
            # > 2^46-element tensor, impossible on this hardware.
            n_blocks = (n_elem // BLOCK).to(tl.int32)
            for j in range(0, n_blocks):
                if WIDE:
                    offs = j.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                else:
                    offs = j * BLOCK + tl.arange(0, BLOCK)
                tl.store(out_ptr + (dst + offs), tl.load(x_ptr + (src + offs)))
            if WIDE:
                base_full = n_blocks.to(tl.int64) * BLOCK
            else:
                base_full = n_blocks * BLOCK
            rem = n_elem - base_full
            if rem > 0:
                offs = base_full + tl.arange(0, BLOCK)
                m = offs < n_elem
                tl.store(
                    out_ptr + (dst + offs),
                    tl.load(x_ptr + (src + offs), mask=m),
                    mask=m,
                )
            r += run
            in_row += run
        else:
            # Rows [t, Lmax) of batch b are pure padding: constant fill.
            # A chunk may start inside the pad region (t > len_b), so
            # clamp the tracked source row back to the batch boundary.
            in_row += len_b - t
            fill_rows = tl.minimum(Lmax - t, r_end - r)
            if WIDE:
                n_elem = fill_rows.to(tl.int64) * D
                dst = r.to(tl.int64) * D
            else:
                n_elem = fill_rows * D
                dst = r * D
            n_blocks = (n_elem // BLOCK).to(tl.int32)
            for j in range(0, n_blocks):
                if WIDE:
                    offs = j.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                else:
                    offs = j * BLOCK + tl.arange(0, BLOCK)
                tl.store(out_ptr + (dst + offs), pad_vec)
            if WIDE:
                base_full = n_blocks.to(tl.int64) * BLOCK
            else:
                base_full = n_blocks * BLOCK
            rem = n_elem - base_full
            if rem > 0:
                offs = base_full + tl.arange(0, BLOCK)
                m = offs < n_elem
                tl.store(out_ptr + (dst + offs), pad_vec, mask=m)
            r += fill_rows
        b_next = r // Lmax
        if b_next > b:
            # Left batch b: skip its source rows clipped past Lmax
            # (explicit max_length smaller than len_b).
            in_row += tl.maximum(len_b - Lmax, 0)
        b = b_next
        t = r - b * Lmax


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
    *,
    max_length: int | None = None,
) -> torch.Tensor:
    # block_t/block_d are accepted for signature compatibility with the
    # generic implementation; the fixed-grid flat-run kernel derives its
    # own tiling and ignores them.
    logger.debug("GEMS_ENFLAMEPACK_SEQ_TRITON")
    dt = x.dtype
    is_uint8 = dt is torch.uint8
    fp8_format = None if is_uint8 else _FP8_ELEMENT_FORMATS.get(dt)
    use_byte_path = is_uint8 or fp8_format is not None
    if is_uint8:
        assert (
            isinstance(pad_value, int) and 0 <= pad_value <= 255
        ), f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        pad_constexpr: int | float = int(pad_value)
    elif fp8_format is not None:
        pad_constexpr = _fp8_bits_from_value(float(pad_value), *fp8_format)
    else:
        pad_constexpr = float(pad_value)

    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    pow2_b = _NEXT_POW2.get(B)
    if pow2_b is None:
        pow2_b = triton.next_power_of_2(B)
        _NEXT_POW2[B] = pow2_b

    # The kernel indexes lengths linearly and reads int32 entries; the hot
    # path (bench/tests already pass int32 contiguous) is a zero-cost pass
    # through, and only mismatched callers pay a metadata-normalizing copy.
    if lengths.dtype is torch.int32 and lengths.is_contiguous():
        lengths_i32 = lengths
    else:
        lengths_i32 = lengths.to(torch.int32)
        if not lengths_i32.is_contiguous():
            lengths_i32 = lengths_i32.contiguous()

    # Launch the Lmax reduction before the remaining host prep so its device
    # execution overlaps the python below; only the .item() waits.
    mx = lengths.max() if max_length is None else None

    if fp8_format is not None:
        if not x_reshaped.is_contiguous():
            x_reshaped = x_reshaped.contiguous()
        x_kernel = x_reshaped.view(torch.uint8)
        elem_size = 1
    else:
        x_kernel = x_reshaped
        elem_size = x_reshaped.element_size()

    Lmax = int(mx.item()) if mx is not None else max_length
    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    total_rows = B * Lmax
    if total_rows == 0 or D == 0:
        if len(original_shape) > 2:
            return out.reshape((B, Lmax) + original_shape[1:])
        return out
    if fp8_format is not None:
        out_kernel = out.view(torch.uint8)
    else:
        out_kernel = out

    # Balanced chunking: ceil-divide the output rows into <= 48 chunks and
    # launch exactly one program per chunk (equal chunk sizes, single wave,
    # one O(B) prefix per chunk, and no idle programs for small inputs).
    rows_per_chunk = -(-total_rows // _NUM_SPC)
    n_chunks = -(-total_rows // rows_per_chunk)
    # Element offsets can exceed i32 for very large tensors; widen the whole
    # address chain (single addptr per trap-note) instead of wrapping.
    wide = 1 if (total_rows * D > 2**31 - 1) or (N * D > 2**31 - 1) else 0

    # Rescan on the decode shapes: 4 warps win for <=2-byte elements, 2 warps
    # for 4-byte elements (fp32 nw4 regressed +14%).
    num_warps = 4 if elem_size <= 2 else 2

    # Small chunks pay mostly predicated-off lanes on the masked tail block;
    # drop the block width to the chunk footprint there.
    chunk_elems = rows_per_chunk * D
    if chunk_elems >= _COPY_BLOCK:
        block = _COPY_BLOCK
    else:
        block = min(_COPY_BLOCK, max(256, triton.next_power_of_2(chunk_elems)))

    _pack_seq_kernel[(n_chunks,)](
        x_kernel,
        out_kernel,
        lengths_i32,
        rows_per_chunk,
        B=B,
        Lmax=Lmax,
        D=D,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=use_byte_path,
        BLOCK_B=pow2_b,
        BLOCK=block,
        WIDE=wide,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out
