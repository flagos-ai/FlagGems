# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of col2im.
#
# Root cause: the generic 2D-tiled col2im kernel
# (flag_gems/ops/col2im.py) uses `triton.autotune` over BLOCK_H/BLOCK_W with
# multiple num_stages, plus a 2D `(BLOCK_H, BLOCK_W)` accumulator and a
# runtime `h_num % stride_h / h_num // stride_h` computed BEFORE masking
# invalid contributions. On kunlunxin XPU this combination miscompiles
# for configs with stride>1/padding>0/dilation>1 -- baseline 12F/3P with
# max abs diff ~512.5 (100% mismatch on the affected configs).
#
# Fix: replace with a flat, output-position-parallel kernel modeled on the
# reflection_pad3d override. Each program handles one 1D BLOCK of output
# elements. Decode (n, c, h, w) via div/mod (all non-negative). Loop kh, kw
# with tl.static_range and accumulate gathers into a fp32 accumulator.
# All arithmetic on valid contributions is over non-negative indices, so
# `h_num % stride_h` and `h_num // stride_h` are safe. Fixed BLOCK=1024
# + CodeGenConfig(isCloseVectorization, buffer_size_limit=2048) avoids the
# XPU tiling/vectorize pipelines that miscompile.
import logging
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def col2im_kernel_flat(
    input_ptr,
    output_ptr,
    channels,
    out_h,
    out_w,
    L_h,
    L_w,
    total_out,
    HW_out,
    CHW_out,
    L_all,
    KHW,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # Decode flat -> (n, c, h, w)
    n_idx = o // CHW_out
    rem = o % CHW_out
    c_idx = rem // HW_out
    rem2 = rem % HW_out
    h_idx = rem2 // out_w
    w_idx = rem2 % out_w

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            # h_num = h_idx + padding_h - kh*dilation_h
            h_num = h_idx.to(tl.int32) + padding_h - kh * dilation_h
            w_num = w_idx.to(tl.int32) + padding_w - kw * dilation_w

            # Both must be non-negative multiples of stride, and quotient in [0, L)
            # Compute quotient only when h_num, w_num are non-negative to avoid
            # negative floor-div / % semantics on XPU.
            h_pos = h_num >= 0
            w_pos = w_num >= 0
            h_num_c = tl.where(h_pos, h_num, 0)
            w_num_c = tl.where(w_pos, w_num, 0)
            l_h = h_num_c // stride_h
            l_w = w_num_c // stride_w
            h_mod = h_num_c - l_h * stride_h
            w_mod = w_num_c - l_w * stride_w

            valid = (
                h_pos & w_pos & (h_mod == 0) & (w_mod == 0) & (l_h < L_h) & (l_w < L_w)
            )

            # Clamp indices for offset computation so masked-out lanes never
            # dereference out-of-range memory on XPU (masked loads on this
            # backend sometimes miscompile with negative/OOB addresses).
            l_h_s = tl.where(valid, l_h, 0)
            l_w_s = tl.where(valid, l_w, 0)
            c_k = c_idx * KHW + kh * kernel_w + kw
            l_idx = l_h_s * L_w + l_w_s
            in_offset = n_idx * (channels * KHW * L_all) + c_k * L_all + l_idx

            v = tl.load(input_ptr + in_offset, mask=mask & valid, other=0.0)
            # XPU may ignore `other` on masked loads; force invalid lanes to 0.
            v = tl.where(valid, v, 0.0)
            acc += v.to(tl.float32)

    tl.store(output_ptr + o, acc.to(output_ptr.type.element_ty), mask=mask)


def _to_pair(val, name):
    if isinstance(val, int):
        return val, val
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return tuple(val)
    raise ValueError(f"Invalid {name}: {val}")


def col2im(
    input: torch.Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COL2IM")

    out_h, out_w = _to_pair(output_size, "output_size")
    kernel_h, kernel_w = _to_pair(kernel_size, "kernel_size")
    dilation_h, dilation_w = _to_pair(dilation, "dilation")
    padding_h, padding_w = _to_pair(padding, "padding")
    stride_h, stride_w = _to_pair(stride, "stride")

    if input.dim() != 3:
        raise ValueError(f"Expected 3D input, got {input.dim()}D")

    batch_size, ck, L = input.shape
    L_h = (out_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    L_w = (out_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if L != L_h * L_w:
        raise ValueError(f"Input size mismatch: expected L={L_h * L_w}, got L={L}")
    kernel_total = kernel_h * kernel_w
    if ck % kernel_total != 0:
        raise ValueError(
            f"Input dim1 {ck} must be divisible by kernel_size {kernel_total}"
        )
    channels = ck // kernel_total

    input = input.contiguous()
    output = torch.empty(
        (batch_size, channels, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )
    if output.numel() == 0:
        return output

    total_out = output.numel()
    HW_out = out_h * out_w
    CHW_out = channels * HW_out

    BLOCK = 1024
    grid = (triton.cdiv(total_out, BLOCK),)
    with torch_device_fn.device(input.device):
        col2im_kernel_flat[grid](
            input,
            output,
            channels,
            out_h,
            out_w,
            L_h,
            L_w,
            total_out,
            HW_out,
            CHW_out,
            L,
            kernel_total,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            BLOCK=BLOCK,
            num_warps=1,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )

    return output
