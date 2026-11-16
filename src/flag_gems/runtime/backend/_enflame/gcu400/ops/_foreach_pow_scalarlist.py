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
"""GCU400-specific fix for ``_foreach_pow.ScalarList``.

The common codegen executor's scalar pow kernel makes the GCU LLVM backend emit an unselectable
``select_cc f16`` node for fp16 / bf16 and fails to compile; the 2D ``grid.y`` on large shapes
also exceeds the hardware limit of 255. Here input / output addresses, element counts
and each float32 exponent bit pattern are packed into one int64 metadata array, and a 1D
fixed-grid ``@triton.jit`` kernel processes the whole list in one launch: both pow operands are
promoted to fp32 before calling libdevice powf, completely avoiding low-precision scalar comparisons.

Non-float, non-contiguous and scalar-count-mismatch shapes fall back to the common executor, with dtype policy
and exception semantics unchanged; metadata is read-only, contains no tensor data, and can be cached by address and
scalar.
"""

import logging
import struct
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from ._foreach_binary import _apply_scalar
from ._foreach_flat import _TL_DTYPES, MODE_SCALAR_LIST, can_use_flat

logger = logging.getLogger(__name__)

# Triton 3.6 tl tensors have no __pow__; reuse the same GCU libdevice powf as the common implementation.
_pow = tl_extra_shim.pow

# --- TUNING PARAMS START ---
# Compile-time / launch-time config, adjusted per optimization round; semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 2
USE_DEVICE_CONTEXT = 0
FALLBACK_TO_PUBLIC = 0
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
USE_SCALAR_CONSTEXPR = 0
GRID_MODE = 0
PACK_NUMEL_SCALAR = 0
USE_SQRT_HALF_EXPONENT = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_pow_scalarlist_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    USE_CONST_SCALAR: tl.constexpr,
    SCALAR_VALUE: tl.constexpr,
    PACKED_META: tl.constexpr,
    USE_SQRT_HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Out-of-place ``self ** scalars[i]`` for the whole list in one launch.

    ``meta_ptr`` holds input addresses, output addresses (NT entries each), element counts and
    float32 exponent bit patterns. ``grid.x`` is linearized over ``(tensor, part)``.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    numel_scalar_raw = tl.load(meta_ptr + 2 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))

    if PACKED_META:
        # The low 32 bits are the float32 exponent bit pattern; the high bits are the element count.
        num_elems = numel_scalar_raw >> 32
        scalar_raw = numel_scalar_raw & 0xFFFFFFFF
    else:
        num_elems = numel_scalar_raw
        scalar_raw = tl.load(meta_ptr + 3 * NT + tensor_id)
    if USE_CONST_SCALAR:
        exponent = SCALAR_VALUE
    else:
        exponent = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        # powf accepts only fp32 operands; store implicitly casts back to the output tensor's dtype.
        base_fp32 = a.to(tl.float32)
        if USE_SQRT_HALF:
            # pow(x, 0.5) matches sqrt(x) point-by-point under IEEE semantics (negative bases give NaN in both),
            # so one sqrt instruction replaces the whole libdevice powf, avoiding log2/exp2 and edge corrections.
            result = tl.sqrt(base_fp32)
        else:
            result = _pow(base_fp32, exponent)
        tl.store(out_ptr + offs, result, mask=mask)


# Read-only metadata cache: the key contains device, input/output addresses, element counts and exponent bit patterns,
# with no tensor data; the kernel still reads and computes fully each time, only repeated synchronous H2D copies are
# saved.
_META_CACHE = {}


def _get_cached_meta(values, device):
    cache_key = (str(device),) + tuple(values)
    cached = _META_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if len(_META_CACHE) >= META_CACHE_LIMIT:
        _META_CACHE.clear()
    cached = torch.tensor(values, dtype=torch.int64, device=device)
    _META_CACHE[cache_key] = cached
    return cached


def _foreach_pow_ScalarList(self, scalars):
    """Out-of-place ``[self[i] ** scalars[i] for i in range(len(self))]``."""
    logger.debug("GEMS_GCU400 _FOREACH_POW_SCALARLIST")

    def fallback():
        return _apply_scalar("pow", self, scalars, inplace=False)

    if FALLBACK_TO_PUBLIC or not can_use_flat(self, scalars, MODE_SCALAR_LIST):
        return fallback()

    num_tensors = len(self)
    outputs = [torch.empty_like(tensor) for tensor in self]
    exponent_bits = [
        struct.unpack("<i", struct.pack("<f", float(item)))[0] for item in scalars
    ]
    if PACK_NUMEL_SCALAR:
        # Pack element count and float32 exponent bit pattern into one int64, shrinking metadata from 4 segments to 3.
        packed = [
            (tensor.numel() << 32) | (bits & 0xFFFFFFFF)
            for tensor, bits in zip(self, exponent_bits)
        ]
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in outputs]
            + packed
        )
    else:
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in outputs]
            + [tensor.numel() for tensor in self]
            + exponent_bits
        )
    device = self[0].device
    if USE_META_CACHE:
        meta = _get_cached_meta(meta_values, device)
    else:
        meta = torch.tensor(meta_values, dtype=torch.int64, device=device)

    blocks_per_tensor = triton.cdiv(max(tensor.numel() for tensor in self), BLOCK_SIZE)
    if GRID_MODE == 1:
        grid_size = max(1, num_tensors * blocks_per_tensor)
    else:
        grid_size = max(1, min(FIXED_GRID, num_tensors * blocks_per_tensor))

    # When all exponent list values are identical a compile-time constant can be used, saving one metadata read and bit
    # cast.
    use_const_scalar = bool(USE_SCALAR_CONSTEXPR and len(set(scalars)) == 1)
    scalar_value = float(scalars[0]) if use_const_scalar else 0.0
    # Only when the whole scalar list is exactly 0.5 can sqrt replace pow; other exponents still use generic powf.
    use_sqrt_half = bool(
        USE_SQRT_HALF_EXPONENT and len(set(scalars)) == 1 and float(scalars[0]) == 0.5
    )

    device_context = (
        torch_device_fn.device(device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        fused_pow_scalarlist_kernel[(grid_size,)](
            meta,
            num_tensors,
            _TL_DTYPES[self[0].dtype],
            use_const_scalar,
            scalar_value,
            PACK_NUMEL_SCALAR,
            use_sqrt_half,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return outputs
