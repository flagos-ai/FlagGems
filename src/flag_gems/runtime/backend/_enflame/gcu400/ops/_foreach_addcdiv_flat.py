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
"""GCU400-specific 1D fused implementation of ``_foreach_addcdiv_.Tensor``.

The common ``foreach_ternary`` executor launches a 2D ``grid=(tensor, tile)``; with
16 tensors of 512x512 and BLOCK=1024 the second dimension reaches 256, exceeding the GCU400 ``grid.y``
hardware limit of 255 and making the benchmark fail with ``OutOfResources``. Here ``grid.x`` is
linearized over ``(tensor, part)`` in a single launch, and the kernel strides over remaining tiles.
Mixed dtype, non-contiguous, broadcast, scalar value, and invalid dtype cases still fall back to the
common executor, keeping the original dtype policy and exception semantics.
"""

import logging
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}

# Returned when the fast path is not applicable, so the caller delegates to the common executor.
FALLBACK = object()

# --- TUNING PARAMS START (foreach_addcdiv_tensor_) ---
# Injected per round by the optimization sweep; all compile-time / launch-time config, semantics unchanged.
ADDC_DIV_TENSOR_TUNING = {
    "enabled": 1,
    "fixed_grid": 24,
    "block_size": 16384,
    "num_warps": 4,
    "num_stages": 1,
    "fp32_promote": 1,
    "cache_meta": 1,
}
# --- TUNING PARAMS END (foreach_addcdiv_tensor_) ---

# do_bench repeatedly calls the same group of tensors; metadata depends only on addresses / numel / value.
# A cache hit avoids two host-to-device small-tensor constructions; the key includes all addresses and values,
# so reused addresses with changed contents still rebuild it, and the cache cap avoids long-term residency.
_META_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
_META_CACHE_MAX = 64


@libentry()
@triton.jit
def fused_addcdiv_tensor_kernel(
    meta_ptr,
    value_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute ``self + value * (tensor1 / tensor2)`` in one launch.

    ``meta_ptr`` is int64 and holds self / output / tensor1 / tensor2 pointers plus
    element counts (NT entries each); ``value_ptr`` holds one fp32 value per tensor.
    ``grid.x`` is linearized over ``(tensor, part)`` to bypass the grid.y limit of 255.
    """
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = program_id % NT
    part_id = program_id // NT
    num_parts = tl.cdiv(num_programs, NT)

    self_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    tensor1_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    tensor2_raw = tl.load(meta_ptr + 3 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 4 * NT + tensor_id)

    self_ptr = self_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    tensor1_ptr = tensor1_raw.to(tl.pointer_type(A_DT))
    tensor2_ptr = tensor2_raw.to(tl.pointer_type(A_DT))
    value = tl.load(value_ptr + tensor_id)

    element_arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offsets = block_id * BLOCK + element_arange
        mask = offsets < num_elems
        self_value = tl.load(self_ptr + offsets, mask=mask, other=0)
        tensor1_value = tl.load(tensor1_ptr + offsets, mask=mask, other=0)
        tensor2_value = tl.load(tensor2_ptr + offsets, mask=mask, other=0)
        if FP32_PROMOTE:
            # Low-precision division is promoted to fp32 first, then implicitly cast back to the output dtype on store.
            self_value = self_value.to(tl.float32)
            tensor1_value = tensor1_value.to(tl.float32)
            tensor2_value = tensor2_value.to(tl.float32)
            value = value.to(tl.float32)
        result = self_value + value * (tensor1_value / tensor2_value)
        tl.store(out_ptr + offsets, result, mask=mask)


def _build_launch_data(self, tensor1, tensor2, out_tensors, values):
    """Return (int64 metadata, fp32 value); reuse device-side tensors on a cache hit."""
    key = (
        tuple(tensor.data_ptr() for tensor in self),
        tuple(tensor.data_ptr() for tensor in out_tensors),
        tuple(tensor.data_ptr() for tensor in tensor1),
        tuple(tensor.data_ptr() for tensor in tensor2),
        tuple(tensor.numel() for tensor in self),
        tuple(float(value) for value in values),
        self[0].dtype,
        self[0].device,
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    meta = torch.tensor(
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in out_tensors]
        + [tensor.data_ptr() for tensor in tensor1]
        + [tensor.data_ptr() for tensor in tensor2]
        + [tensor.numel() for tensor in self],
        dtype=torch.int64,
        device=self[0].device,
    )
    device_values = torch.tensor(
        [float(value) for value in values],
        dtype=torch.float32,
        device=self[0].device,
    )
    if len(_META_CACHE) >= _META_CACHE_MAX:
        _META_CACHE.clear()
    _META_CACHE[key] = (meta, device_values)
    return meta, device_values


def can_use_flat(self, tensor1, tensor2, scalars) -> bool:
    """Use the fast path only for same-dtype, contiguous, equal-length lists with per-tensor numeric values."""
    if not self:
        return False
    if not isinstance(scalars, torch.Tensor) or scalars.numel() != len(self):
        return False
    if not isinstance(tensor1, (list, tuple)) or len(tensor1) != len(self):
        return False
    if not isinstance(tensor2, (list, tuple)) or len(tensor2) != len(self):
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES:
        return False
    device = self[0].device
    for base, first, second in zip(self, tensor1, tensor2):
        if (
            base.dtype != dtype
            or first.dtype != dtype
            or second.dtype != dtype
            or base.device != device
            or first.device != device
            or second.device != device
        ):
            return False
        if (
            not base.is_contiguous()
            or not first.is_contiguous()
            or not second.is_contiguous()
        ):
            return False
        if (
            base.numel() == 0
            or first.numel() != base.numel()
            or second.numel() != base.numel()
        ):
            return False
    return True


def run_addcdiv_tensor(self, tensor1, tensor2, scalars, inplace: bool):
    """Execute the .Tensor overload of addcdiv; return the ``FALLBACK`` sentinel when the fast path does not apply."""
    config = ADDC_DIV_TENSOR_TUNING
    if not config["enabled"] or not can_use_flat(self, tensor1, tensor2, scalars):
        return FALLBACK

    num_tensors = len(self)
    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    out_tensors = self if inplace else outputs
    values = scalars.tolist()
    if config["cache_meta"]:
        meta, device_values = _build_launch_data(
            self, tensor1, tensor2, out_tensors, values
        )
    else:
        meta = torch.tensor(
            [tensor.data_ptr() for tensor in self]
            + [tensor.data_ptr() for tensor in out_tensors]
            + [tensor.data_ptr() for tensor in tensor1]
            + [tensor.data_ptr() for tensor in tensor2]
            + [tensor.numel() for tensor in self],
            dtype=torch.int64,
            device=self[0].device,
        )
        device_values = torch.tensor(
            [float(value) for value in values],
            dtype=torch.float32,
            device=self[0].device,
        )

    block_size = config["block_size"]
    blocks_per_tensor = triton.cdiv(max(tensor.numel() for tensor in self), block_size)
    # Give every tensor at least one program without exceeding the fixed grid limit.
    grid = max(1, min(config["fixed_grid"], num_tensors * blocks_per_tensor))
    fused_addcdiv_tensor_kernel[(grid,)](
        meta,
        device_values,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        config["fp32_promote"],
        block_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return None if inplace else outputs
