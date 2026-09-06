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
def adaptive_max_pool2d_forward_flat_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    numel,
    # Flat 1-D layout over (n, c, oh, ow). All pooling geometry is constexpr so
    # the per-lane decomposition compiles to cheap ALU, gather addresses stay
    # affine and @libentry caches one kernel per shape. Taps are loaded
    # unconditionally from clamped addresses (the XPU backend treats compound
    # i1 masked loads as a slow path and `other=` is unreliable there); window
    # membership is settled value-wise with tl.where.
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    MAX_KH: tl.constexpr,
    MAX_KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    store_mask = offsets < numel
    safe_offsets = tl.where(store_mask, offsets, 0)

    ow = safe_offsets % OW
    ohw = safe_offsets // OW
    oh = ohw % OH
    nc = ohw // OH

    # Adaptive window: start = floor(o * I / O), end = ceil((o + 1) * I / O).
    h_start = (oh * IH) // OH
    h_end = ((oh + 1) * IH + OH - 1) // OH
    w_start = (ow * IW) // OW
    w_end = ((ow + 1) * IW + OW - 1) // OW

    plane_base = input_ptr + nc * (IH * IW)

    acc_val = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    acc_idx = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)

    # h-major tap scan keeps the window max on the *first* (h, w) tap that
    # attains it and the *last* NaN, matching ATen's adaptive_max_pool2d.
    for kh in range(MAX_KH):
        h_in = h_start + kh
        h_ok = h_in < h_end
        h_safe = tl.where(h_ok, h_in, h_start)
        for kw in range(MAX_KW):
            w_in = w_start + kw
            w_ok = w_in < w_end
            w_safe = tl.where(w_ok, w_in, w_start)
            value = tl.load(plane_base + h_safe * IW + w_safe).to(tl.float32)
            active = h_ok & w_ok
            is_new = active & ((value > acc_val) | (value != value) | (acc_idx < 0))
            acc_val = tl.where(is_new, value, acc_val)
            acc_idx = tl.where(is_new, h_in * IW + w_in, acc_idx)

    tl.store(
        output_ptr + offsets,
        acc_val.to(output_ptr.dtype.element_ty),
        mask=store_mask,
    )
    tl.store(indices_ptr + offsets, acc_idx.to(tl.int64), mask=store_mask)


@libentry()
@triton.jit
def adaptive_max_pool2d_forward_wide64_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    numel,
    # Divisible-window fast path: every window is exactly KH x KW and tiles the
    # input, so tap addresses are base + const and, when the window-start is a
    # multiple of 2 (fp32) or 4 (fp16/bf16) elements, the taps can be fetched as
    # i64 loads (2x fp32 / 4x fp16 per lane). The wider load raises the memory
    # transaction size from 256B (2B/lane) to 1KB (8B/lane) at the same 128-lane
    # tile, which measured ~2.7-3.0x on XPU (0.94 -> 0.345ms fp16,
    # 3.65 -> 1.23ms fp16 shape2). BLOCK > 128 is a hard uni_sram limit for any
    # accumulator kernel on this backend, so lanes stay at 128.
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SUB: tl.constexpr,  # 4 for fp16/bf16, 2 for fp32
    IS_F16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    store_mask = offsets < numel
    safe_offsets = tl.where(store_mask, offsets, 0)

    ow = safe_offsets % OW
    ohw = safe_offsets // OW
    oh = ohw % OH
    nc = ohw // OH

    plane = KH * OH * IW
    base_off = (oh * KH) * IW + ow * KW  # fp16/fp32 plane idx of window's top-left
    base = (nc * plane + base_off) // SUB  # wide-scalar (i64) index

    acc_val = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    acc_idx = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)
    input64 = input_ptr.to(tl.pointer_type(tl.int64))
    if SUB == 4:
        # 16-bit types: 4 halfs per i64. fp16 goes through a native fp16->fp32
        # convert (fp16 is NOT a plain <<16 of its bits); bf16 bits ARE the top
        # 16 bits of an fp32, so bf16->fp32 is exactly <<16 (and the bf16
        # bitcast path triggers a uni_sram OOR on this backend).
        for kh in range(KH):
            for kw2 in range(KW // 4):
                raw = tl.load(input64 + base + (kh * IW // 4) + kw2)
                for s in tl.static_range(4):
                    if IS_F16:
                        v = (
                            ((raw >> (16 * s)) & 0xFFFF)
                            .to(tl.uint16)
                            .to(tl.float16, bitcast=True)
                            .to(tl.float32)
                        )
                    else:
                        v = (
                            (((raw >> (16 * s)) & 0xFFFF).to(tl.uint32) << 16).to(
                                tl.float32, bitcast=True
                            )
                        )
                    idx = base_off + kh * IW + 4 * kw2 + s
                    is_new = (v > acc_val) | (v != v) | (acc_idx < 0)
                    acc_val = tl.where(is_new, v, acc_val)
                    acc_idx = tl.where(is_new, idx, acc_idx)
    else:
        for kh in range(KH):
            for kw2 in range(KW // 2):
                raw = tl.load(input64 + base + (kh * IW // 2) + kw2)
                for s in tl.static_range(2):
                    v = ((raw >> (32 * s)) & 0xFFFFFFFF).to(tl.uint32).to(
                        tl.float32, bitcast=True
                    )
                    idx = base_off + kh * IW + 2 * kw2 + s
                    is_new = (v > acc_val) | (v != v) | (acc_idx < 0)
                    acc_val = tl.where(is_new, v, acc_val)
                    acc_idx = tl.where(is_new, idx, acc_idx)

    tl.store(
        output_ptr + offsets,
        acc_val.to(output_ptr.dtype.element_ty),
        mask=store_mask,
    )
    tl.store(indices_ptr + offsets, acc_idx.to(tl.int64), mask=store_mask)


def _parse_output_size(output_size):
    if isinstance(output_size, int):
        return output_size, output_size
    if isinstance(output_size, (list, tuple)) and len(output_size) == 2:
        return output_size[0], output_size[1]
    raise ValueError(f"Invalid output_size: {output_size}")


def _max_window(in_size: int, out_size: int) -> int:
    # Exact longest adaptive window instead of the loose cdiv(in, out) + 1
    # bound: every saved tap is a saved unconditional load per output lane.
    longest = 1
    for o in range(out_size):
        start = (o * in_size) // out_size
        end = -((-(o + 1) * in_size) // out_size)
        longest = max(longest, end - start)
    return longest


def adaptive_max_pool2d(
    input: torch.Tensor,
    output_size,
):
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL2D")
    input = input.contiguous()

    out_h, out_w = _parse_output_size(output_size)

    if input.dim() == 3:
        # Unbatched (C, H, W): add a batch dim so the flat decomposition holds.
        input = input.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    in_n, in_c, in_h, in_w = input.shape

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0 or input.numel() == 0:
        if squeeze_batch:
            return output.squeeze(0), indices.squeeze(0)
        return output, indices

    numel = output.numel()
    # 128 lanes is the largest tile this kernel compiles at: the loop body keeps
    # ~10 live BLOCK-sized vectors (window bounds, accumulators, addresses) and
    # BLOCK_SIZE >= 256 makes TritonXPUUnrollControl fail with `uni_sram` OOR on
    # every shape measured, independent of buffer_size_limit / num_warps.
    # 128-lane tiles only pay off once there is enough work to hand every one of
    # the 64 cores a full tile (64 * 128); below that the 64-lane tile wins
    # because the masked tail is a smaller share of the launch.
    block_size = 128 if numel >= 8192 else 64
    grid = (triton.cdiv(numel, block_size),)

    # Wide-i64 fast path: windows must tile the input exactly (divisible) and
    # every window start must sit on a 2/4-element boundary so the taps can be
    # packed into i64 loads (2x fp32 / 4x fp16+bf16 per lane). This raises the
    # memory transaction size 256B -> 1KB at the same 128-lane tile and measured
    # 2.7-3.0x faster on the benchmark shapes. Non-divisible / misaligned / fp64
    # inputs keep the general flat kernel.
    if (
        in_h % out_h == 0
        and in_w % out_w == 0
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        k_w = in_w // out_w
        if input.dtype in (torch.float16, torch.bfloat16) and in_w % 4 == 0 and k_w % 4 == 0:
            sub = 4
        elif input.dtype == torch.float32 and in_w % 2 == 0 and k_w % 2 == 0:
            sub = 2
        else:
            sub = 0
    else:
        sub = 0

    with torch_device_fn.device(input.device):
        if sub:
            adaptive_max_pool2d_forward_wide64_kernel[grid](
                input,
                output,
                indices,
                numel,
                IW=in_w,
                OH=out_h,
                OW=out_w,
                KH=in_h // out_h,
                KW=in_w // out_w,
                SUB=sub,
                IS_F16=(input.dtype == torch.float16),
                BLOCK_SIZE=block_size,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
        else:
            adaptive_max_pool2d_forward_flat_kernel[grid](
                input,
                output,
                indices,
                numel,
                IH=in_h,
                IW=in_w,
                OH=out_h,
                OW=out_w,
                MAX_KH=_max_window(in_h, out_h),
                MAX_KW=_max_window(in_w, out_w),
                BLOCK_SIZE=block_size,
                num_warps=1,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )

    if squeeze_batch:
        return output.squeeze(0), indices.squeeze(0)
    return output, indices
