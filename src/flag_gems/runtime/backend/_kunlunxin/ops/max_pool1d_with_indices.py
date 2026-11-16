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

Fast path: for the dominant geometry ``k=3, s=2, p=1, dilation=1,
ceil_mode=False, L % 2 == 0`` (out_len == L // 2) the window of output ``o`` is
``[2o-1, 2o, 2o+1]``.  With the input viewed as pairs (fp32 -> int64, fp16/bf16
-> int32) the flat output index equals the flat pair index, so the three taps
are recovered from two unit-stride pair streams:

    self = pair[q]    (lo=x[2o], hi=x[2o+1])   -> xp + q
    left = pair[q-1]  (hi=x[2o-1])             -> xp + q - 1

``pair[-1]`` of the very first output is out of the allocation, so the head of
the output runs on a clamped (de-vectorised) launch while the rest stays purely
affine -- the same split the proven ``max_pool1d`` / ``avg_pool1d`` pair kernels
use.  The pair path is 1.7-1.9x faster than the packed-word kernel and removes
the per-lane address clamp from the hot loop.

bf16 detail: the previous implementation stored bf16 as two ``uint8`` stores
(``2*offset`` / ``2*offset+1``) because the backend rejects a bf16 bitcast
store.  That per-byte, stride-2 store was measured **16x** slower than a single
``uint16`` store (2890 us vs 173 us on the benchmark's smallest cell), so both
16-bit dtypes now store through a ``uint16`` pointer.  The bf16 flat path loads
through a ``uint16`` pointer as well (1.7x faster than two ``uint8`` loads).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# outputs covered by the clamped head launch of the pair kernel.  Only lane 0
# needs it (its left tap is pair[-1]); a *clamped* block is de-vectorised and
# costs ~90us when it is the only block in flight, so this stays tiny.
HEAD_OUT = 64
# below this output count the whole operator is launch bound, so a single
# clamped launch is faster than head + affine body
SINGLE_LAUNCH_MAX = 4096


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


def _pair_block(total: int) -> int:
    # Measured on P800/XPU3 (same table as the max_pool1d pair kernel).
    if total <= 4096:
        return 1024
    if total <= 262144:
        return 2048
    return 4096


def _flat_block(total: int, elem_size: int) -> int:
    """Measured medians on P800/XPU3 for the flat / wide kernels.

    Small totals stay at 1024 (a 4096-lane tile costs 8-10% there); fp32 prefers
    2048 once it has work to hide, while 16-bit loads pack two elements per
    word and only want 4096 past ~2M outputs (1M outputs: 1024 is 7% faster).
    """
    if total <= 2048:
        return 512
    if total <= 16384:
        return 1024
    if elem_size >= 4:
        return 1024 if total <= 163840 else 2048
    if total > 2000000:
        return 4096
    return 1024


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


@triton.jit
def _store16(output_ptr, offsets, bits, mask):
    """Single 16-bit store; a byte-pair store is 16x slower on this backend."""
    out16 = output_ptr.to(tl.pointer_type(tl.uint16))
    tl.store(out16 + offsets, bits.to(tl.uint16), mask=mask)


@libentry()
@triton.jit
def max_pool1d_forward_pair_kernel(
    xp,
    yp,
    ip,
    total: tl.constexpr,
    off: tl.constexpr,
    OW: tl.constexpr,
    CLAMP: tl.constexpr,
    FULL: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``k=3, s=2, p=1, dilation=1, ceil_mode=False, L % 2 == 0`` fast path.

    MODE 0: fp32 input viewed as int64 pairs.
    MODE 1: fp16 input viewed as int32 pairs.
    MODE 2: bf16 input viewed as int32 pairs.

    ``CLAMP`` bounds both pair indices to the last pair: lane 0 of the head
    would otherwise read ``pair[-1]``, and the tail lanes of a single clamped
    launch (small outputs) would read past the last pair.  This backend still
    performs the access of a *masked* load (only the ``other=`` select is
    dropped), so the clamp -- not the mask -- is what keeps those launches
    inside the allocation.  ``FULL`` marks the interior body blocks and the
    boundary-aligned last block (no mask at all, nothing to clamp).
    """
    lan = tl.arange(0, BLOCK)
    q0 = tl.program_id(0) * BLOCK + lan + off
    m = q0 < total
    if CLAMP:
        q = tl.minimum(q0, total - 1)
        left = tl.maximum(q - 1, 0)
    else:
        q = q0
        left = q - 1
    if FULL:
        v0 = tl.load(xp + q)
        v1 = tl.load(xp + left)
    else:
        v0 = tl.load(xp + q, mask=m, other=0)
        v1 = tl.load(xp + left, mask=m, other=0)

    if MODE == 0:
        v0 = v0.to(tl.uint64, bitcast=True)
        v1 = v1.to(tl.uint64, bitcast=True)
        lo_bits = (v0 & 0xFFFFFFFF).to(tl.uint32)
        hi_bits = (v0 >> 32).to(tl.uint32)
        led_bits = (v1 >> 32).to(tl.uint32)
    else:
        v0 = v0.to(tl.uint32, bitcast=True)
        v1 = v1.to(tl.uint32, bitcast=True)
        lo_bits = v0 & 0xFFFF
        hi_bits = v0 >> 16
        led_bits = v1 >> 16

    ELEM16: tl.constexpr = MODE != 0
    IS_BF16: tl.constexpr = MODE == 2
    ol = q % OW

    # Integer sort keys, never a float comparison: this backend flushes fp32
    # subnormals in compares (measured: bf16 0x0001 vs 0x0000 compare equal),
    # which would return the wrong value and index.  Window order is
    # left -> lo -> hi; ATen keeps the first maximum and always updates on a
    # NaN (so the last NaN wins).  Seeding with the left (or lo) tap removes
    # one full update chain.
    key_led = _pool_key(led_bits, ELEM16, IS_BF16)
    key_lo = _pool_key(lo_bits, ELEM16, IS_BF16)
    key_hi = _pool_key(hi_bits, ELEM16, IS_BF16)
    has_left = ol > 0
    mx_bits = tl.where(has_left, led_bits, lo_bits)
    mx_key = tl.where(has_left, key_led, key_lo)
    mx_idx = tl.where(has_left, 2 * ol - 1, 2 * ol).to(tl.int32)

    is_new = (key_lo > mx_key) | (key_lo == 0x7FFFFFFF)
    mx_bits = tl.where(is_new, lo_bits, mx_bits)
    mx_key = tl.where(is_new, key_lo, mx_key)
    mx_idx = tl.where(is_new, (2 * ol).to(mx_idx.dtype), mx_idx)

    is_new = (key_hi > mx_key) | (key_hi == 0x7FFFFFFF)
    mx_bits = tl.where(is_new, hi_bits, mx_bits)
    mx_idx = tl.where(is_new, (2 * ol + 1).to(mx_idx.dtype), mx_idx)

    if FULL:
        # Interior / boundary-aligned blocks: every lane is a real output, so
        # the stores are plain (measured neutral vs masked stores; the win in
        # this kernel comes from the affine addressing, not from the mask).
        if ELEM16:
            out16 = yp.to(tl.pointer_type(tl.uint16))
            tl.store(out16 + q, mx_bits.to(tl.uint16))
        else:
            tl.store(yp + q, mx_bits.to(tl.float32, bitcast=True))
        tl.store(ip + q, mx_idx.to(tl.int64))
    else:
        if ELEM16:
            _store16(yp, q, mx_bits, m)
        else:
            tl.store(yp + q, mx_bits.to(tl.float32, bitcast=True), mask=m)
        tl.store(ip + q, mx_idx.to(tl.int64), mask=m)


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
    if IS_BF16:
        # 16-bit gather through a uint16 pointer: one load per tap instead of
        # two byte loads (1.7x faster); every address is element aligned.
        ptr16 = input_ptr.to(tl.pointer_type(tl.uint16))
    for kl in tl.static_range(0, kernel_l):
        il = start + kl * dilation_l
        valid = output_mask & (il >= 0) & (il < in_l)
        il_safe = tl.where(valid, il, 0)
        input_offset = nc_safe * in_l + il_safe
        if IS_BF16:
            bits = tl.load(ptr16 + input_offset).to(tl.uint32)
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
        _store16(output_ptr, offsets, max_bits, output_mask)
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

    if ELEM16:
        _store16(output_ptr, offsets, max_bits, output_mask)
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


def _pair_fast_path(inp, kernel_l, stride_l, padding_l, dilation_l, ceil_mode, out_l):
    if ceil_mode or kernel_l != 3 or stride_l != 2 or padding_l != 1:
        return False
    if dilation_l != 1:
        return False
    if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if inp.shape[-1] % 2 != 0 or inp.shape[-1] // 2 != out_l:
        return False
    align = 8 if inp.dtype == torch.float32 else 4
    if inp.data_ptr() % align != 0:
        return False
    # the pair kernel indexes with int32
    if inp.numel() // 2 > 0x7FFFFFFF:
        return False
    return True


def _launch_pair(inp, output, indices, out_l, dtype):
    total = output.numel()
    if dtype == torch.float32:
        xv, mode = inp.view(torch.int64), 0
    else:
        xv, mode = inp.view(torch.int32), (2 if dtype == torch.bfloat16 else 1)
    block = _pair_block(total)
    head = min(HEAD_OUT, total)
    launches = []
    if total <= SINGLE_LAUNCH_MAX:
        launches.append((triton.cdiv(total, block), 0, True, False, block))
    else:
        # head: clamped so lane 0 never reads pair[-1]
        launches.append((1, 0, True, False, head))
        rest = total - head
        full = rest // block
        if full:
            launches.append((full, head, False, True, block))
        if rest - full * block:
            # Tail without a clamped (de-vectorised) launch: an overlapping
            # last block that ends exactly at `total` stays fully in bounds and
            # only repeats lanes with identical values.
            launches.append((1, total - block, False, True, block))
    with torch_device_fn.device(inp.device):
        for grid, off, clamp, full, blk in launches:
            max_pool1d_forward_pair_kernel[(grid,)](
                xv,
                output,
                indices,
                total,
                off,
                OW=out_l,
                CLAMP=clamp,
                FULL=full,
                MODE=mode,
                BLOCK=blk,
                num_warps=4,
                buffer_size_limit=2048,
                isCloseVectorization=False,
            )


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

    if _pair_fast_path(
        input, kernel_l, stride_l, padding_l, dilation_l, ceil_mode, out_l
    ):
        _launch_pair(input, output, indices, out_l, input.dtype)
    elif (
        input.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and dilation_l == 1
        and stride_l % 2 == 0
        and in_l % 2 == 0
        and input.data_ptr() % (2 * input.element_size()) == 0
    ):
        total = output.numel()
        block = _flat_block(total, input.element_size())
        grid = (triton.cdiv(total, block),)
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
        total = output.numel()
        block = _flat_block(total, input.element_size())
        grid = (triton.cdiv(total, block),)
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
