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
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

# grid.x hardware limit; larger launches are rejected by the backend.
_MAX_GRID = 65535
# Elements per CTA for large tensors. Measured on GCU400 at 2**30 elements:
# 30 ms with this tile vs 104 ms with a 512-element tile, since one CTA per
# small tile spends most of its time on launch/scheduling instead of streaming.
_BLOCK_SIZE = 8192
# Tile for tensors too small to fill the machine at _BLOCK_SIZE.
_MIN_BLOCK_SIZE = 512


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_kernel(
    x_ptr,
    scale_ptr,
    zero_point_ptr,
    fake_quant_enabled_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    QUANT_MIN: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    # Scalar qparams stay on device: one load per program, then cache hits.
    scale = tl.load(scale_ptr).to(tl.float32)
    inv = tl.div_rn(1.0, scale)
    zero_point = tl.load(zero_point_ptr).to(tl.float32)
    enabled = tl.load(fake_quant_enabled_ptr) != 0

    # Capped grid with a grid-stride loop: one CTA per tile would overflow
    # grid.x (65535) past ~537M elements at this tile size, e.g. the
    # 2**30-element benchmark shape.
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(tl.program_id(0), num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        load_mask = offsets < n_elements

        x_raw = tl.load(x_ptr + offsets, mask=load_mask, other=0.0)
        x = x_raw.to(tl.float32)
        # ATen's tensor_qparams path multiplies by the fp32 reciprocal of scale
        # and rounds to nearest even before adding zero_point; dividing by scale
        # differs by an ulp on exact ties.
        q = tl_extra_shim.nearbyint(x * inv) + zero_point
        in_range = (q >= QUANT_MIN) & (q <= QUANT_MAX)
        q_clamped = tl.minimum(tl.maximum(q, QUANT_MIN), QUANT_MAX)
        # NaN does not propagate through ATen's clamp; it lands on QUANT_MIN.
        q_clamped = tl.where(x != x, QUANT_MIN * 1.0, q_clamped)
        quantized = scale * (q_clamped - zero_point)
        # When fake quantization is disabled, pass the input through untouched.
        # Store the native-dtype value rather than the fp32-widened one: ATen
        # returns the input bit-for-bit here, and not relying on the
        # fp16/bf16 -> fp32 -> fp16 round trip leaves NaN payloads and signed
        # zeros to a plain copy.
        output = tl.where(enabled, quantized.to(out_ptr.dtype.element_ty), x_raw)
        mask = tl.where(enabled, in_range, True)

        tl.store(out_ptr + offsets, output, mask=load_mask)
        tl.store(mask_ptr + offsets, mask, mask=load_mask)


def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    logger.debug(
        "GEMS_ENFLAME FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS"
    )
    self = self.contiguous()
    out = torch.empty_like(self)
    mask = torch.empty_like(self, dtype=torch.bool)

    n_elements = self.numel()
    if n_elements == 0:
        return out, mask

    if fake_quant_enabled.dtype == torch.int64:
        # This backend's Triton int64 loads silently read as 0 unless the
        # int64-enablement env is set. The enabled flag is a 0/1 int64 scalar,
        # so read its little-endian low 32 bits instead; pure metadata rebind.
        fake_quant_enabled = fake_quant_enabled.reshape(-1).view(torch.int32)

    # Small tensors cannot fill the machine at a big tile, so keep the 512
    # tile that gives them one CTA per tile; past that, big tiles amortize
    # launch/scheduling, worth up to 4.9x at 16M elements and 3.4x at 2**30
    # (measured on GCU400, fp16/fp32/bf16).
    block_size = _BLOCK_SIZE if n_elements >= _BLOCK_SIZE * 8 else _MIN_BLOCK_SIZE
    num_warps = 4 if self.dtype == torch.float32 else 8
    num_tiles = triton.cdiv(n_elements, block_size)
    # Cap grid.x at the 65535 hardware limit and let each CTA walk a strided
    # set of tiles, instead of growing the tile until it fits or spilling into
    # a second grid dimension.
    grid_size = min(num_tiles, _MAX_GRID)
    grid = (grid_size,)
    with torch_device_fn.device(self.device):
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_kernel[grid](
            self,
            scale,
            zero_point,
            fake_quant_enabled,
            out,
            mask,
            n_elements,
            QUANT_MIN=quant_min,
            QUANT_MAX=quant_max,
            BLOCK_SIZE=block_size,
            GRID_SIZE=grid_size,
            num_warps=num_warps,
        )
    return out, mask
