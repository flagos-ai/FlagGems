# Copyright 2026, The FlagOS Contributors.
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

"""Kunlunxin(XPU) backend override for aten::_dyn_quant_pack_4bit_weight.

The op packs dynamic INT4 weights into PyTorch's ATen "portable fallback"
format: a single float32 buffer holding [upcast weights bytes, scales/zeros,
bias] in that order.  Semantics are identical to the generic implementation
in flag_gems.ops._dyn_quant_pack_4bit_weight (validated bit-for-bit against
the ATen CPU reference).

XPU 性能设计（2026-08-19 测量驱动）:
- 原通用实现是单 kernel、BLOCK=256 的平坦 3 段合并拷贝。XPU 上 launch
  开销 ~0.3-0.6us/program，BLOCK=256 使 (1024,1024,128) 产生 2k+ 个
  micro-program，launch-bound 导致 0.65ms（比 torch ref cat ~22x 慢）。
  增大 BLOCK 至 8192 + 每段一个程序组的无掩码块（段尾才掩码）将该 case
  降到 ~19us（与 torch cat 15-16us 同一量级）。
- 小 total 时单 launch 的平坦 kernel（BLOCK=512/4096）反而优于多段
  多 launch，保留 flat 快速路径；按 total_elements 分档。

限制记录（XPU Triton 后端实测 hard cap）:
- tl.load 掩码向量 > 8192（16384/32768/65536）编译失败（CoruseTiling 崩）；
  故大块一律 8192。
- u32/u64 向量提取字节 + 交错 4/8 段 store 在本后端产生离散 gather
  （620-1143us），禁止使用。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _pack_u8_seg_kernel(
    src,
    dst,
    n_elements,
    dst_base,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(src + offs, mask=mask, other=0)
    tl.store(dst + dst_base + offs, x.to(tl.float32), mask=mask)


@triton.jit
def _pack_u8_seg_kernel_unmasked(
    src,
    dst,
    dst_base: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offs)
    tl.store(dst + dst_base + offs, x.to(tl.float32))


@triton.jit
def _pack_f32_seg_kernel(
    src,
    dst,
    n_elements,
    dst_base,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(src + offs, mask=mask, other=0.0)
    tl.store(dst + dst_base + offs, x, mask=mask)


@triton.jit
def _pack_f32_seg_kernel_unmasked(
    src,
    dst,
    dst_base: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offs)
    tl.store(dst + dst_base + offs, x)


@triton.jit
def _pack_flat_kernel(
    weights,
    scales_zeros,
    bias,
    output,
    weight_elements,
    scale_elements,
    total_elements,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    weight_mask = mask & (offsets < weight_elements)
    values = tl.where(
        weight_mask,
        tl.load(weights + offsets, mask=weight_mask, other=0).to(tl.float32),
        values,
    )

    scale_offsets = offsets - weight_elements
    scale_mask = mask & (scale_offsets >= 0) & (scale_offsets < scale_elements)
    values = tl.where(
        scale_mask,
        tl.load(scales_zeros + scale_offsets, mask=scale_mask, other=0.0).to(
            tl.float32
        ),
        values,
    )

    if HAS_BIAS:
        bias_offsets = scale_offsets - scale_elements
        bias_mask = mask & (bias_offsets >= 0)
        values = tl.where(
            bias_mask,
            tl.load(bias + bias_offsets, mask=bias_mask, other=0.0).to(tl.float32),
            values,
        )

    tl.store(output + offsets, values, mask=mask)


# XPU 上实测：B 越大宽段块效益越好但 8192 是掩码/编译上限；小 total 用 flat。
SEG_BLOCK = 8192
FLAT_SMALL_BLOCK = 512
FLAT_MID_BLOCK = 4096
FLAT_MAX_TOTAL = 1 << 17  # 131072，更大走分段 unmasked 路径


def _emit_segment(kernel_masked, kernel_unmasked, src, dst, n, base, block):
    if n <= 0:
        return
    if n % block == 0:
        kernel_unmasked[(n // block,)](src, dst, base, BLOCK_SIZE=block)
    else:
        kernel_masked[(triton.cdiv(n, block),)](
            src, dst, n, base, BLOCK_SIZE=block
        )


def _dyn_quant_pack_4bit_weight(
    weights: torch.Tensor,
    scales_zeros: torch.Tensor,
    bias: torch.Tensor | None,
    block_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Pack dynamic INT4 weights in ATen's portable fallback format."""
    logger.debug("GEMS _DYN_QUANT_PACK_4BIT_WEIGHT (kunlunxin)")
    if weights.dtype != torch.uint8:
        raise RuntimeError("_dyn_quant_pack_4bit_weight expects uint8 weights")
    if block_size != in_features and (
        block_size % 32 != 0 or in_features % block_size != 0
    ):
        raise RuntimeError(
            "group size must equal in_features or divide it as a multiple of 32"
        )
    if scales_zeros.device != weights.device or (
        bias is not None and bias.device != weights.device
    ):
        raise RuntimeError("weights, scales_zeros, and bias must be on the same device")

    weights = weights.contiguous()
    scales_zeros = scales_zeros.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    weight_elements = weights.numel()
    scale_elements = scales_zeros.numel()
    bias_elements = 0 if bias is None else bias.numel()
    total_elements = weight_elements + scale_elements + bias_elements
    output = torch.empty(total_elements, device=weights.device, dtype=torch.float32)
    if total_elements == 0:
        return output

    device = weights.device
    with torch_device_fn.device(device):
        if total_elements <= FLAT_MAX_TOTAL:
            block = FLAT_MID_BLOCK if total_elements > 8192 else FLAT_SMALL_BLOCK
            grid = (triton.cdiv(total_elements, block),)
            bias_arg = weights if bias is None else bias
            _pack_flat_kernel[grid](
                weights,
                scales_zeros,
                bias_arg,
                output,
                weight_elements,
                scale_elements,
                total_elements,
                HAS_BIAS=bias is not None,
                BLOCK_SIZE=block,
            )
            return output

        _emit_segment(
            _pack_u8_seg_kernel,
            _pack_u8_seg_kernel_unmasked,
            weights,
            output,
            weight_elements,
            0,
            SEG_BLOCK,
        )
        _emit_segment(
            _pack_f32_seg_kernel,
            _pack_f32_seg_kernel_unmasked,
            scales_zeros,
            output,
            scale_elements,
            weight_elements,
            SEG_BLOCK,
        )
        if bias is not None and bias_elements > 0:
            _emit_segment(
                _pack_f32_seg_kernel,
                _pack_f32_seg_kernel_unmasked,
                bias,
                output,
                bias_elements,
                weight_elements + scale_elements,
                SEG_BLOCK,
            )
    return output