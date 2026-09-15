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
#
# XPU vendor implementation of torch.dequantize for quantized tensors.
#
# Why a vendor override is needed at all:
#   * The generic implementation (src/flag_gems/ops/dequantize.py) reads the
#     quantized payload through `a.int_repr()`, whose native kernel is not
#     present in the XPU build ("CUDA error: invalid device function").  We
#     instead build a zero-copy int8 view of the quantized storage
#     (`untyped_storage()` + `set_`) and run the arithmetic in a Triton
#     kernel.
#   * Quantized tensors dispatch through the `QuantizedCUDA` backend key; a
#     plain `CUDA` registration (what the generic `_FULL_CONFIG`/registrar
#     path uses) is never consulted for them, and the XPU-native
#     `dequantize.self` kernel is missing as well.  This module therefore
#     registers its implementation for the `QuantizedCUDA` key at import time
#     (process-wide) and keeps the `torch.library.Library` object alive to
#     prevent GC-unregistration.  The `CUDA`-key registration performed by
#     `use_gems()` stays inert for quantized tensors.
#
# Performance note: a plain 1-D triton kernel with an int8 load tops out at
# ~146 GB/s on XPU (the int8 -> float conversion breaks the block-DMA
# packing).  The `pointwise_dynamic` codegen with `isCloseVectorization` +
# `unroll_num=16` + `prefer_1d_tile` reaches ~627 GB/s (5.2 ms for 655M
# elements vs 48.5 ms naive; the XPU-native int8->float cast itself runs at
# ~1486 GB/s), so an unrolled vectorized 1-D tile is used here.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

_quantized_lib = None  # keep reference alive to prevent GC

_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    num_outputs=1,
    config=_config,
)
@triton.jit
def dequantize_func(inp, zero_point, scale):
    return (inp - zero_point) * scale


def dequantize(a):
    """Dequantize a quantized tensor (e.g. torch.qint8) to float32 on XPU.

    Args:
        a: A quantized tensor (torch.qint8 / ...) on the XPU device.

    Returns:
        A float32 tensor with the dequantized values.
    """
    logger.debug("GEMS DEQUANTIZE")

    scale = float(a.q_scale())
    zero_point = int(a.q_zero_point())

    if a.numel() == 0:
        return torch.empty(a.shape, dtype=torch.float32, device=a.device)

    # `a.int_repr()` has no kernel in the XPU build; build a zero-copy int8
    # view of the quantized payload instead (handles storage offset/strides,
    # materializing a contiguous copy only if the payload is not contiguous).
    raw = torch.empty(0, dtype=torch.int8, device=a.device)
    raw.set_(a.untyped_storage(), a.storage_offset(), a.size(), a.stride())
    if not raw.is_contiguous():
        raw = raw.contiguous()

    return dequantize_func(raw, zero_point, scale)


def _register_quantized_dequantize():
    """Register the XPU dequantize for the QuantizedCUDA dispatch key.

    Idempotent.  Quantized tensors (`torch.quantize_per_tensor(...)` on the
    XPU device) carry the `QuantizedCUDA` keyset; the `CUDA`-key registration
    used by the generic config is never consulted for them and the XPU-native
    `dequantize` kernel is absent, so `torch.dequantize` would otherwise fail
    with "invalid device function".
    """
    global _quantized_lib
    if _quantized_lib is not None:
        return
    try:
        _quantized_lib = torch.library.Library("aten", "IMPL")
        _quantized_lib.impl("dequantize.self", dequantize, "QuantizedCUDA")
        _quantized_lib.impl("dequantize", dequantize, "QuantizedCUDA")
        logger.debug(
            "FlagGems Kunlunxin: registered QuantizedCUDA dequantize override"
        )
    except Exception as e:
        logger.warning(
            f"GEMS_KL3 failed to register dequantize QuantizedCUDA override: {e}"
        )


_register_quantized_dequantize()