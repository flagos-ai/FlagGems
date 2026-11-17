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

"""Kunlunxin (XPU) implementation of ``max_pool1d``.

The generic kernel relies on ``tl.load(..., mask=..., other=min_val)`` to
implement the -inf padding; this backend silently ignores the ``other`` clause
of masked loads (KERNEL_OPT_EXPERIENCE #4.3-6), so padded lanes read
neighbouring memory and produce wrong maxima at output positions that touch the
padding (S1: 39/54 passed, all 15 failures were padding-value mismatches).

Structure
---------
* ``max_pool1d_forward_flat_kernel`` mirrors the validated
  max_pool1d_with_indices flat kernel (clamp address + explicit ``valid``
  check + integer-order key; bf16 byte loads/stores for the fallback path;
  no indices output).
* ``max_pool1d_forward_wide_kernel`` handles the common downsampling branch
  (``stride % 2 == 0``, ``dilation == 1``, even ``in_l`` and
  ``(stride//2) * out_l == in_l//2``).  Its packed-word addresses are
  **affine** in the flat output offset::

      word(s) = A * offs + floor((s - padding) / 2),   A = stride // 2

  where ``offs = nc * out_l + ol``.  The max_pool1d_with_indices wide kernel
  instead clamps every word address with ``min/max``; on this stack such
  per-lane select/clamp addresses are lowered as a discrete gather and the
  same kernel is ~4.5x slower than the affine form (measured
  ``harness/perf_ir/probe_max_pool1d_affine.py``: fp32 (1024,1024) 648us ->
  146us).  Edge lanes (word < 0 or >= total_packed) are predicated off with an
  explicit bounds mask and their (garbage) values are never selected because
  selection is guarded by ``valid``.
* 16-bit dtypes load as packed uint32 words and store through a **uint16
  pointer**.  The blue-print bf16 byte-store path (`two uint8 stores per
  element`) measured ~20x slower than the uint16-pointer store (2794us ->
  138us for (1024,1024) bf16) and is only kept in the flat fallback kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def max_pool1d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode: the last window must
        # start strictly inside the (padded) input, not in the padding.
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1
    return output_size


@triton.jit
def _pool_key(
    bits, ELEM16: tl.constexpr, IS_BF16: tl.constexpr, ELEM64: tl.constexpr = False
):
    if ELEM64:
        signed = bits.to(tl.int64)
        sign_mask: tl.constexpr = 0x7FFFFFFFFFFFFFFF
        inf: tl.constexpr = 0x7FF0000000000000
    else:
        sign_mask: tl.constexpr = 0x7FFFFFFF
        # Shift 16-bit encodings into the high half for signed int32 keys.
        if ELEM16:
            signed = (bits.to(tl.uint32) << 16).to(tl.int32)
            if IS_BF16:
                inf: tl.constexpr = 0x7F800000
            else:
                inf: tl.constexpr = 0x7C000000
        else:
            signed = bits.to(tl.int32)
            inf: tl.constexpr = 0x7F800000
    # Negative encodings reverse order; signed zeros must compare equal.
    magnitude = signed & sign_mask
    key = tl.where(signed < 0, signed ^ sign_mask, signed)
    key = tl.where(magnitude == 0, 0, key)
    # All NaNs sort above numbers. The caller replaces equal NaN keys, keeping
    # the last NaN, while equal numeric keys keep the first sampled value.
    return tl.where(magnitude > inf, sign_mask, key)


@triton.jit
def _pool_value(
    bits, ELEM16: tl.constexpr, IS_BF16: tl.constexpr, ELEM64: tl.constexpr = False
):
    if ELEM64:
        return bits.to(tl.float64, bitcast=True)
    elif ELEM16:
        # Store the winning encoding directly: no rounding, subnormal flush or
        # NaN-payload conversion is needed for a selection operation.
        if IS_BF16:
            return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        else:
            return bits.to(tl.uint16).to(tl.float16, bitcast=True)
    else:
        return bits.to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def max_pool1d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    total: tl.constexpr,
    in_l: tl.constexpr,
    out_l: tl.constexpr,
    kernel_l: tl.constexpr,
    stride_l: tl.constexpr,
    padding_l: tl.constexpr,
    dilation_l: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ELEM16: tl.constexpr = (
        input_ptr.dtype.element_ty == tl.float16
        or input_ptr.dtype.element_ty == tl.bfloat16
    )
    ELEM64: tl.constexpr = input_ptr.dtype.element_ty == tl.float64
    IS_BF16: tl.constexpr = input_ptr.dtype.element_ty == tl.bfloat16
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if total > 0x7FFFFFFF or (total // out_l) * in_l > 0x7FFFFFFF:
        offsets = offsets.to(tl.int64)
    output_mask = offsets < total
    nc_idx = offsets // out_l
    ol = offsets % out_l
    nc_safe = tl.where(output_mask, nc_idx, 0)
    start = ol * stride_l - padding_l
    if ELEM64:
        neg_inf: tl.constexpr = 0xFFF0000000000000
        bits_type: tl.constexpr = tl.uint64
        nan_key: tl.constexpr = 0x7FFFFFFFFFFFFFFF
    else:
        bits_type: tl.constexpr = tl.uint32
        nan_key: tl.constexpr = 0x7FFFFFFF
        if ELEM16:
            if IS_BF16:
                neg_inf: tl.constexpr = 0xFF80
            else:
                neg_inf: tl.constexpr = 0xFC00
        else:
            neg_inf: tl.constexpr = 0xFF800000
    max_bits = tl.full((BLOCK,), neg_inf, bits_type)
    max_key = _pool_key(max_bits, ELEM16, IS_BF16, ELEM64)

    for kl in tl.static_range(0, kernel_l):
        il = start + kl * dilation_l
        valid = output_mask & (il >= 0) & (il < in_l)
        il_safe = tl.where(valid, il, 0)
        input_offset = nc_safe * in_l + il_safe
        if ELEM16:
            # uint16-pointer load: one memory transaction per element and no
            # float<->int bitcast convert (bf16/fp16 scalar bitcast raises
            # XPU `TritonXPUDtypeConvert` size-mismatch errors).
            in16 = input_ptr.to(tl.pointer_type(tl.uint16))
            bits = tl.load(in16 + input_offset).to(tl.uint32)
        else:
            value = tl.load(input_ptr + input_offset)
            if ELEM16:
                bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
            else:
                bits = value.to(bits_type, bitcast=True)
        key = _pool_key(bits, ELEM16, IS_BF16, ELEM64)
        is_new = valid & ((key > max_key) | (key == nan_key))
        max_bits = tl.where(is_new, bits, max_bits)
        max_key = tl.where(is_new, key, max_key)

    if ELEM16:
        # uint16-pointer store: no float bitcast in the kernel, measured ~20x
        # faster than the bf16 byte-store path on this stack.
        out16 = output_ptr.to(tl.pointer_type(tl.uint16))
        tl.store(out16 + offsets, max_bits.to(tl.uint16), mask=output_mask)
    else:
        tl.store(
            output_ptr + offsets,
            _pool_value(max_bits, ELEM16, IS_BF16, ELEM64),
            mask=output_mask,
        )


@libentry()
@triton.jit
def max_pool1d_forward_wide_kernel(
    input_ptr,
    output_ptr,
    total: tl.constexpr,
    total_packed: tl.constexpr,
    in_l: tl.constexpr,
    out_l: tl.constexpr,
    kernel_l: tl.constexpr,
    stride_l: tl.constexpr,
    padding_l: tl.constexpr,
    ELEM16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Affine packed-word wide kernel (stride even, dilation == 1).

    Host dispatches here only when ``(stride_l // 2) * out_l == in_l // 2`` so
    the word address of tap ``s`` collapses to ``A * offs + floor((s-p)/2)``
    with ``A = stride_l // 2`` -- affine in the flat output offset.  Bounds
    masks predicate the (at most a couple of) out-of-range word lanes off;
    their garbage values are never selected because every ``is_new`` update is
    guarded by ``valid``.
    """
    A: tl.constexpr = stride_l // 2
    BMIN: tl.constexpr = (-padding_l) // 2
    NLOAD: tl.constexpr = (kernel_l - 1 - padding_l) // 2 - BMIN + 1

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if total > 0x7FFFFFFF or total_packed > 0x3FFFFFFF:
        offsets = offsets.to(tl.int64)
    output_mask = offsets < total
    nc_idx = offsets // out_l
    ol = offsets % out_l
    nc_safe = tl.where(output_mask, nc_idx, 0)

    if ELEM16:
        ptr = input_ptr.to(tl.pointer_type(tl.uint32))
        if IS_BF16:
            neg_inf: tl.constexpr = 0xFF80
        else:
            neg_inf: tl.constexpr = 0xFC00
    else:
        ptr = input_ptr.to(tl.pointer_type(tl.uint64))
        neg_inf: tl.constexpr = 0xFF800000

    ol_base = ol * stride_l - padding_l
    max_bits = tl.full((BLOCK,), neg_inf, tl.uint32)
    max_key = _pool_key(max_bits, ELEM16, IS_BF16)

    for k in tl.static_range(NLOAD):
        w = offsets * A + (BMIN + k)
        words_mask = (w >= 0) & (w < total_packed)
        raw = tl.load(ptr + w, mask=words_mask)
        for s in tl.static_range(kernel_l):
            if ((s - padding_l) // 2 - BMIN) != k:
                pass
            else:
                if ELEM16:
                    bits = ((raw >> (16 * ((s - padding_l) & 1))) & 0xFFFF).to(
                        tl.uint32
                    )
                else:
                    bits = (raw >> ((32 * ((s - padding_l) & 1)) & 63)).to(tl.uint32)
                e = ol_base + s
                valid = output_mask & (e >= 0) & (e < in_l)
                key = _pool_key(bits, ELEM16, IS_BF16)
                is_new = valid & ((key > max_key) | (key == 0x7FFFFFFF))
                max_bits = tl.where(is_new, bits, max_bits)
                max_key = tl.where(is_new, key, max_key)

    if ELEM16:
        # uint16-pointer store: no float bitcast in the kernel, measured ~20x
        # faster than the bf16 byte-store path on this stack.
        out16 = output_ptr.to(tl.pointer_type(tl.uint16))
        tl.store(out16 + offsets, max_bits.to(tl.uint16), mask=output_mask)
    else:
        tl.store(
            output_ptr + offsets,
            max_bits.to(tl.float32, bitcast=True),
            mask=output_mask,
        )


def _parse_pool_params(kernel_size, stride, padding, dilation):
    def _single(value, name):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a single int for 1D pooling")
            return int(value[0])
        return int(value)

    kernel_l = _single(kernel_size, "kernel_size")

    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride_l = kernel_l
    else:
        stride_l = _single(stride, "stride")

    padding_l = _single(padding, "padding")
    dilation_l = _single(dilation, "dilation")

    if kernel_l <= 0:
        raise ValueError(f"kernel_size must be positive, but got {kernel_l}")
    if stride_l <= 0:
        raise ValueError(f"stride must be positive, but got {stride_l}")
    if padding_l < 0:
        raise ValueError(f"padding must be non-negative, but got {padding_l}")
    if dilation_l <= 0:
        raise ValueError(f"dilation must be positive, but got {dilation_l}")

    return kernel_l, stride_l, padding_l, dilation_l


def max_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL1D")

    kernel_l, stride_l, padding_l, dilation_l = _parse_pool_params(
        kernel_size, stride, padding, dilation
    )

    # Input may be (C, L) or (N, C, L). Normalize to 3D for the kernel launch.
    unbatched = input.dim() == 2
    if unbatched:
        input = input.unsqueeze(0)

    input = input.contiguous()
    in_n, in_c, in_l = input.shape
    out_l = max_pool1d_output_size(
        in_l, kernel_l, stride_l, padding_l, dilation_l, ceil_mode
    )

    output = torch.empty((in_n, in_c, out_l), device=input.device, dtype=input.dtype)

    if output.numel() == 0:
        return output.squeeze(0) if unbatched else output

    total = output.numel()
    block = 2048
    grid = (triton.cdiv(total, block),)

    wide_eligible = (
        input.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and dilation_l == 1
        and stride_l % 2 == 0
        and in_l % 2 == 0
        and (stride_l // 2) * out_l == in_l // 2
        and input.data_ptr() % (2 * input.element_size()) == 0
    )
    if wide_eligible:
        with torch_device_fn.device(input.device):
            max_pool1d_forward_wide_kernel[grid](
                input,
                output,
                total,
                input.numel() // 2,
                in_l,
                out_l,
                kernel_l,
                stride_l,
                padding_l,
                input.dtype != torch.float32,
                input.dtype == torch.bfloat16,
                block,
                num_warps=2,
                buffer_size_limit=2048,
                isCloseVectorization=False,
            )
    else:
        with torch_device_fn.device(input.device):
            max_pool1d_forward_flat_kernel[grid](
                input,
                output,
                total,
                in_l,
                out_l,
                kernel_l,
                stride_l,
                padding_l,
                dilation_l,
                BLOCK=block,
                num_warps=2,
                buffer_size_limit=2048,
                isCloseVectorization=False,
            )

    if unbatched:
        output = output.squeeze(0)
    return output
