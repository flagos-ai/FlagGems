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

# GCU400 customization (2026-11-17):
# 1) Fix the hard benchmark crash: the official bench calls this op with device='gcu' (no index string),
#    so `inp.device == device` is always False and it falls into generic `_resize_output` -> generic
#    copy_ with a 1D unbounded grid (at the 2^30-element scale grid.x=131072 > the hard limit of 65535, OutOfResources).
#    Device normalization: an index-less device of the same type means "the current device of that type"; for the same
#    physical device, go directly
#    through the in-place metadata path; genuinely cross-device cases go through the local enflame `_resize_output`
#    (whose internal copy_ uses a 48-CTA grid-stride loop, so no scale hits the limit).
# 2) The in-place path goes straight to `set_.source_Storage_storage_offset` (FlagGems does not register this
#    overload, so behavior is deterministic in both dispatch states): zero aten calls in the same-shape steady state
#    (inductor
#    output-buffer reuse mostly involves repeated shapes; this mirrors native resize_'s same-shape early exit).
#    torch_gcu's set_ has no capacity check - an out-of-bounds set_ silently succeeds and out-of-bounds reads risk
#    MMU/RAS errors,
#    so grow performs a (offset+numel)*itemsize capacity check; if insufficient, following native resize_ semantics it
#    allocates new storage + copies the linear prefix (local copy_).
#
# GCU400 perf (2026-11-19): this op is metadata-only in steady state (bench and inductor buffer reuse both replay
# repeated same-numel shapes), so per-call host Python overhead dominates (~1.06us measured, zero kernels). Slimmed
# the hot path without changing semantics:
#   - removed the unconditional `logger.debug` call (0.44us/call);
#   - flattened device matching inline: single `inp.device.type` / `.index` reads (was 2-3 each), str devices parsed
#     via `partition` instead of building a `torch.device` object, `torch_device_fn.current_device()` still consulted
#     only for index-less same-type devices;
#   - shape guard switched from `inp.shape == torch.Size(size)` (Size construction per call) to
#     `tuple(size) == inp.shape` (tuple(size) is cheap; torch.Size is a tuple subclass).
# Measured fast path (same-shape early return): 1.06us -> 0.81us on gcu400; bench do_bench level
# avg 2.13us -> 2.02us (the fixed do_bench harness overhead is outside the op).
import math
from typing import List

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops._resize_output import _resize_output
from flag_gems.runtime.backend._enflame.gcu400.ops.copy import copy_

_MISMATCH_INDEX = -2  # sentinel: never equal to any real tensor device index (or None)


def _resize_tensor_inplace(inp: torch.Tensor, size) -> torch.Tensor:
    """In-place equivalent of aten::resize_ that skips aten dispatch (under use_gems
    resize_ routes to the local set_, which has no capacity check; this version guards it)."""
    new_numel = math.prod(size)
    old_numel = inp.numel()
    storage = inp.untyped_storage()
    offset = inp.storage_offset()
    if (
        new_numel <= old_numel
        or (offset + new_numel) * inp.element_size() <= storage.nbytes()
    ):
        inp.set_(storage, offset, size)
        return inp
    # grow beyond capacity: allocate new storage + copy the linear prefix (native resize_ semantics; for non-contiguous
    # inp, reshape(-1) materializes logical order, consistent with generic _resize_output)
    new = torch.empty(size, dtype=inp.dtype, device=inp.device)
    if old_numel:
        copy_(new.reshape(-1)[:old_numel], inp.reshape(-1))
    inp.set_(new.untyped_storage(), 0, size)
    return inp


def _resize_output_(inp: torch.Tensor, size: List[int], device: torch.device):
    tensor_device = inp.device
    tensor_type = tensor_device.type
    tensor_index = tensor_device.index
    if isinstance(device, torch.device):
        target_type, target_index = device.type, device.index
    elif isinstance(device, str):
        target_type, sep, target_index = device.partition(":")
        if sep:
            try:
                target_index = int(target_index)
            except ValueError:
                # Invalid device string: raise the canonical RuntimeError the
                # torch.device(device) construction used to raise here.
                torch.device(device)
                target_index = _MISMATCH_INDEX
        else:
            target_index = None
    else:
        target_type, target_index = None, _MISMATCH_INDEX
    same_device = target_type == tensor_type and (
        (
            target_index is None
            and (
                tensor_index is None or tensor_index == torch_device_fn.current_device()
            )
        )
        or target_index == tensor_index
    )
    if same_device:
        if inp.is_contiguous() and tuple(size) == inp.shape:
            return inp
        return _resize_tensor_inplace(inp, size)
    out = _resize_output(inp, size, device)
    _resize_tensor_inplace(inp, size)
    inp.copy_(out)
    return inp
