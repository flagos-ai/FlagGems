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
"""GCU400-specific fix for ``_foreach_addcmul_.ScalarList``.

The common codegen executor carries the tensor list in a 2D ``grid=(block, tensor)``; GCU400's
``grid.y <= 255`` limit throws ``OutOfResources`` for the 16x(1024x1024) benchmark shape,
making the operator unusable. Here the three address groups of self / tensor1 /
tensor2, each tensor's element count, and each float32 scalar bit pattern are packed into
one int64 metadata array, and a 1D fixed-grid ``@triton.jit`` kernel completes the whole
in-place ``self += value * tensor1 * tensor2`` list in one launch.

Non-float, non-contiguous, mismatched list lengths or element counts fall back to the common executor,
keeping dtype policy and exception semantics unchanged; metadata is read-only and contains no tensor data,
so it can be cached by (device, address, element count, scalar bit pattern), removing repeated synchronous H2D copies.
"""

import logging
import struct
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_binary import _apply_ternary
from ._foreach_flat import _TL_DTYPES, FLOAT_DTYPES

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time / launch-time config, adjusted per optimization round; semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 2
FP32_PROMOTE = 1
USE_DEVICE_CONTEXT = 0
FALLBACK_TO_PUBLIC = 0
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
USE_SCALAR_CONSTEXPR = 0
GRID_MODE = 0
PACK_NUMEL_SCALAR = 0
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_addcmul_scalarlist_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    USE_CONST_SCALAR: tl.constexpr,
    SCALAR_VALUE: tl.constexpr,
    PACKED_META: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place addcmul over the whole list in one launch.

    ``meta_ptr`` holds the addresses of self / tensor1 / tensor2 (NT entries each),
    element counts (NT entries), and float32 scalar bit patterns (NT entries, carried as int64).
    ``grid.x`` is linearized over ``(tensor, part)`` to avoid the ``grid.y`` limit.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    b_raw = tl.load(meta_ptr + NT + tensor_id)
    c_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    numel_scalar_raw = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(A_DT))
    c_ptr = c_raw.to(tl.pointer_type(A_DT))

    if PACKED_META:
        # The low 32 bits are the float32 scalar bit pattern; the high bits are the element count.
        num_elems = numel_scalar_raw >> 32
        scalar_raw = numel_scalar_raw & 0xFFFFFFFF
    else:
        num_elems = numel_scalar_raw
        scalar_raw = tl.load(meta_ptr + 4 * NT + tensor_id)
    if USE_CONST_SCALAR:
        value = SCALAR_VALUE
    else:
        value = tl.cast(scalar_raw.to(tl.int32), tl.float32, bitcast=True)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        c = tl.load(c_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            # Promote low precision to fp32 for computation, then implicitly cast back to the output dtype on store.
            a = a.to(tl.float32)
            b = b.to(tl.float32)
            c = c.to(tl.float32)
        tl.store(a_ptr + offs, a + value * b * c, mask=mask)


# Read-only metadata cache: the key consists of device, three address groups, element counts and scalar bit patterns,
# with no tensor data; the kernel still reads and computes everything each time, only repeated synchronous H2D copies
# are saved.
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


def _can_use_fused(self, tensor1, tensor2, scalars):
    """Use the fast path only when all three lists are contiguous float tensors of the same dtype and the scalar list
    is valid."""
    if not isinstance(scalars, (list, tuple)) or len(scalars) != len(self):
        return False
    if not self or len(tensor1) != len(self) or len(tensor2) != len(self):
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES:
        return False
    for tensor in self:
        if tensor.dtype != dtype or not tensor.is_contiguous() or tensor.numel() == 0:
            return False
    for first, second, third in zip(self, tensor1, tensor2):
        if (
            second.dtype != dtype
            or third.dtype != dtype
            or not second.is_contiguous()
            or not third.is_contiguous()
            or second.numel() != first.numel()
            or third.numel() != first.numel()
        ):
            return False
    return all(isinstance(value, (int, float)) for value in scalars)


def _foreach_addcmul__ScalarList(self, tensor1, tensor2, value=1):
    """In-place ``self += scalars[i] * tensor1[i] * tensor2[i]``."""
    logger.debug("GEMS_GCU400 _FOREACH_ADDCMUL__SCALARLIST")

    def fallback():
        return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)

    if FALLBACK_TO_PUBLIC or not _can_use_fused(self, tensor1, tensor2, value):
        return fallback()

    num_tensors = len(self)
    scalar_bits = [
        struct.unpack("<i", struct.pack("<f", float(item)))[0] for item in value
    ]
    if PACK_NUMEL_SCALAR:
        # Pack element count and float32 scalar bit pattern into one int64, shrinking metadata from 5 segments to 4.
        packed = [
            (tensor.numel() << 32) | (bits & 0xFFFFFFFF)
            for tensor, bits in zip(self, scalar_bits)
        ]
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in tensor1]
            + [tensor.data_ptr() for tensor in tensor2]
            + packed
        )
    else:
        meta_values = (
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in tensor1]
            + [tensor.data_ptr() for tensor in tensor2]
            + [tensor.numel() for tensor in self]
            + scalar_bits
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

    # When all scalar list values are identical a compile-time constant can be used, saving one metadata read and bit
    # cast;
    # changing a constexpr triggers a new Triton specialization, so this is only an optional strategy.
    use_const_scalar = bool(USE_SCALAR_CONSTEXPR and len(set(value)) == 1)
    scalar_value = float(value[0]) if use_const_scalar else 0.0

    device_context = (
        torch_device_fn.device(device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        fused_addcmul_scalarlist_kernel[(grid_size,)](
            meta,
            num_tensors,
            _TL_DTYPES[self[0].dtype],
            FP32_PROMOTE,
            use_const_scalar,
            scalar_value,
            PACK_NUMEL_SCALAR,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return None
