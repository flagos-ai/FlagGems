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

"""Kunlunxin (XPU) implementation of ``max_pool1d_with_indices``.

The generic kernel relies on ``tl.load(..., mask=..., other=min_val)`` to
implement the -inf padding; this backend silently ignores the ``other`` clause
of masked loads (see KERNEL_OPT_EXPERIENCE #4.3-6), so padded lanes read
neighbouring memory and produce wrong maxima at output positions that touch the
left padding.  This override follows the proven max_pool2d flat kernel: clamp
the input address to an in-bounds lane first, load unconditionally, then apply
an explicit validity check. Integer ordering preserves NaNs, infinities and
first-occurrence numeric ties without backend float-comparison lowering.
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
        # PyTorch-compatible adjustment for ceil_mode
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
    indices_ptr,
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
    # Match ATen even for a dilated window with no in-bounds samples.
    first = start + tl.maximum(tl.cdiv(-start, dilation_l), 0) * dilation_l
    if in_l + dilation_l <= 0x7FFFFFFF:
        max_idx = first.to(tl.int32)
    else:
        max_idx = first.to(tl.int64)
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
        if IS_BF16:
            # Avoid bf16 and u16 loads on the fallback path too. Two byte loads
            # do not overread odd lengths or require packed-word alignment.
            ptr = input_ptr.to(tl.pointer_type(tl.uint8))
            lo = tl.load(ptr + input_offset * 2).to(tl.uint32)
            hi = tl.load(ptr + input_offset * 2 + 1).to(tl.uint32)
            bits = lo | (hi << 8)
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
        max_idx = tl.where(is_new, il.to(max_idx.dtype), max_idx)

    if IS_BF16:
        # bf16 bitcast store hits XPU `TritonXPUDtypeConvert` "Cannot bitcast
        # size 16 to size 32". Store the two bytes directly, symmetric to the
        # byte loads above, to keep bit-exact NaNs/signed-zero.
        out8 = output_ptr.to(tl.pointer_type(tl.uint8))
        tl.store(out8 + offsets * 2, (max_bits & 0xFF).to(tl.uint8), mask=output_mask)
        tl.store(
            out8 + offsets * 2 + 1,
            ((max_bits >> 8) & 0xFF).to(tl.uint8),
            mask=output_mask,
        )
    else:
        tl.store(
            output_ptr + offsets,
            _pool_value(max_bits, ELEM16, IS_BF16, ELEM64),
            mask=output_mask,
        )
    tl.store(indices_ptr + offsets, max_idx.to(tl.int64), mask=output_mask)


@libentry()
@triton.jit
def max_pool1d_forward_wide_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
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
    """Pack two consecutive fp16/fp32 elements into one int32/int64 load.

    For stride % 2 == 0 and dilation == 1 the ``kernel_l`` taps of an output
    are consecutive input elements, so they live in ``(kernel_l + MIS + 1)//2``
    packed words (MIS = padding_l % 2) instead of ``kernel_l`` separate loads.
    This follows the proven max_pool2d wide kernel: the packed loads are
    contiguous and roughly halve the load count, which is the dominant cost of
    the flat kernel on this backend.
    """
    MIS: tl.constexpr = padding_l % 2
    NLOAD: tl.constexpr = (kernel_l + MIS + 1) // 2

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

    max_bits = tl.full((BLOCK,), neg_inf, tl.uint32)
    max_key = _pool_key(max_bits, ELEM16, IS_BF16)
    ol_base = ol * stride_l - padding_l
    if in_l <= 0x7FFFFFFF:
        max_idx = tl.maximum(ol_base, 0).to(tl.int32)
    else:
        max_idx = tl.maximum(ol_base, 0).to(tl.int64)

    e0 = nc_safe * in_l + ol_base
    b0 = e0 >> 1
    for k in tl.static_range(NLOAD):
        bk = tl.minimum(tl.maximum(b0 + k, 0), total_packed - 1)
        raw = tl.load(ptr + bk)
        for s in tl.static_range(kernel_l):
            if (s + MIS) // 2 != k:
                pass
            else:
                if ELEM16:
                    bits = ((raw >> (16 * ((s + MIS) % 2))) & 0xFFFF).to(tl.uint32)
                else:
                    bits = (raw >> (32 * ((s + MIS) % 2))).to(tl.uint32)
                e = ol_base + s
                valid = output_mask & (e >= 0) & (e < in_l)
                key = _pool_key(bits, ELEM16, IS_BF16)
                is_new = valid & ((key > max_key) | (key == 0x7FFFFFFF))
                max_bits = tl.where(is_new, bits, max_bits)
                max_key = tl.where(is_new, key, max_key)
                max_idx = tl.where(is_new, e.to(max_idx.dtype), max_idx)

    if IS_BF16:
        out8 = output_ptr.to(tl.pointer_type(tl.uint8))
        tl.store(out8 + offsets * 2, (max_bits & 0xFF).to(tl.uint8), mask=output_mask)
        tl.store(
            out8 + offsets * 2 + 1,
            ((max_bits >> 8) & 0xFF).to(tl.uint8),
            mask=output_mask,
        )
    else:
        tl.store(
            output_ptr + offsets,
            _pool_value(max_bits, ELEM16, IS_BF16),
            mask=output_mask,
        )
    tl.store(indices_ptr + offsets, max_idx.to(tl.int64), mask=output_mask)


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


def max_pool1d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_KUNLUNXIN MAX_POOL1D_WITH_INDICES")

    kernel_l, stride_l, padding_l, dilation_l = _parse_pool_params(
        kernel_size, stride, padding, dilation
    )

    # Input may be (C, L) or (N, C, L). Normalize to 3D for the kernel launch.
    squeeze_batch = input.dim() == 2
    if squeeze_batch:
        input = input.unsqueeze(0)

    input = input.contiguous()
    in_n, in_c, in_l = input.shape
    out_l = max_pool1d_output_size(
        in_l, kernel_l, stride_l, padding_l, dilation_l, ceil_mode
    )

    output = torch.empty((in_n, in_c, out_l), device=input.device, dtype=input.dtype)
    indices = torch.empty((in_n, in_c, out_l), device=input.device, dtype=torch.int64)

    if output.numel() == 0:
        if squeeze_batch:
            output = output.squeeze(0)
            indices = indices.squeeze(0)
        return output, indices

    total = output.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)

    if (
        input.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and dilation_l == 1
        and stride_l % 2 == 0
        and in_l % 2 == 0
        and input.data_ptr() % (2 * input.element_size()) == 0
    ):
        with torch_device_fn.device(input.device):
            max_pool1d_forward_wide_kernel[grid](
                input,
                output,
                indices,
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
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=False,
            )
    else:
        with torch_device_fn.device(input.device):
            max_pool1d_forward_flat_kernel[grid](
                input,
                output,
                indices,
                total,
                in_l,
                out_l,
                kernel_l,
                stride_l,
                padding_l,
                dilation_l,
                BLOCK=block,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=False,
            )

    if squeeze_batch:
        output = output.squeeze(0)
        indices = indices.squeeze(0)

    return output, indices
