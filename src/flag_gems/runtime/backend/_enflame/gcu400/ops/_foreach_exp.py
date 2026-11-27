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

"""GCU400-specific _foreach_exp / _foreach_exp_ (mostly in-place).

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.utils import libentry

from ._foreach_flat import _TL_DTYPES, can_use_flat

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization driver script; all compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_META_CACHE = 1
META_CACHE_LIMIT = 32
# out-of-place output allocation: 1 = one contiguous allocation + output base address passed as a kernel scalar argument
# (zero H2D); 0 = per-tensor empty_like (addresses provided by the 4th segment of kernel metadata).
PACK_OUTPUTS = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


@libentry()
@triton.jit
def fused_exp_kernel(
    meta_ptr,
    out_base,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_OUT: tl.constexpr,
    PACK_OUT: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
):
    """One launch applies ``exp`` to the whole tensor list (in place or written out).

    ``meta_ptr`` holds, in order, the input addresses, the element counts and each output's
    offset inside the contiguous output buffer (used only with ``HAS_OUT``). With
    ``HAS_OUT=1`` and ``PACK_OUT=1`` the output base is a kernel scalar argument; with
    ``PACK_OUT=0`` outputs come from a 4th metadata segment; with ``HAS_OUT=0`` the input is
    written in place. grid.x is flattened as ``(tensor, part)`` to stay inside ``grid.y <= 255``.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    in_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    if HAS_OUT:
        if PACK_OUT == 1:
            # The output base address is a kernel scalar argument; static metadata stores only each tensor's offset.
            out_offset = tl.load(meta_ptr + 2 * NT + tensor_id)
            out_raw = out_base + out_offset * ELEM_BYTES
        else:
            out_raw = tl.load(meta_ptr + 3 * NT + tensor_id)
    else:
        out_raw = in_raw
    in_ptr = in_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        value = tl.load(in_ptr + offs, mask=mask)
        # exp on the GCU backend accepts only fp32/fp64, so lower precision must be promoted first;
        # on store it is implicitly downcast back to the output tensor's dtype.
        result = tl.exp(value.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


# Read-only metadata cache: the key contains only device, addresses, numels and offsets, no tensor data;
# the kernel still reads the full input and recomputes each time, so address reuse or in-place modification of contents
# doesn't affect correctness.
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


def _run_exp(self, inplace):
    """Zero-H2D fast path; the caller guards the input shape via ``can_use_flat``."""
    num_tensors = len(self)
    numels = [tensor.numel() for tensor in self]
    out_offsets = []
    running_offset = 0
    for numel in numels:
        out_offsets.append(running_offset)
        running_offset += numel
    # The third segment holds each tensor's element offset within the contiguous output buffer (determined by numels,
    # and doesn't change across calls); when PACK_OUTPUTS=0, a fourth segment with per-tensor output addresses is
    # appended.
    static_values = [tensor.data_ptr() for tensor in self] + numels + out_offsets
    if USE_META_CACHE:
        meta = _get_cached_meta(static_values, self[0].device)
    else:
        meta = torch.tensor(static_values, dtype=torch.int64, device=self[0].device)

    if inplace:
        outputs = None
        out_ptr_meta = meta
        out_base = 0
        has_out = 0
        pack_out = 0
    elif PACK_OUTPUTS:
        # One contiguous allocation replaces 16 empty_like calls; outputs are independent
        # views on the same buffer with non-overlapping segments, so modifying one output in place doesn't affect the
        # others.
        # Views are built with a single as_strided call (narrow+view is two dispatches; for 16
        # tensors the host overhead is 66µs -> 19µs, directly visible in do_bench for small-shape host-bound
        # settings).
        flat_output = torch.empty(
            running_offset, dtype=self[0].dtype, device=self[0].device
        )
        outputs = []
        element_offset = 0
        for tensor in self:
            outputs.append(
                torch.as_strided(
                    flat_output, tensor.shape, tensor.stride(), element_offset
                )
            )
            element_offset += tensor.numel()
        if PACK_OUTPUTS == 1:
            out_ptr_meta = meta
            out_base = flat_output.data_ptr()
            pack_out = 1
        else:
            out_ptr_meta = torch.tensor(
                [tensor.data_ptr() for tensor in outputs],
                dtype=torch.int64,
                device=self[0].device,
            )
            out_base = 0
            pack_out = 0
        has_out = 1
    else:
        outputs = [torch.empty_like(tensor) for tensor in self]
        out_ptr_meta = torch.tensor(  # noqa: F841
            [tensor.data_ptr() for tensor in outputs],
            dtype=torch.int64,
            device=self[0].device,
        )
        out_base = 0
        has_out = 1
        pack_out = 0

    blocks_per_tensor = triton.cdiv(max(numels), BLOCK_SIZE)
    grid_size = max(1, min(FIXED_GRID, num_tensors * blocks_per_tensor))
    fused_exp_kernel[(grid_size,)](
        meta,
        out_base,
        num_tensors,
        _TL_DTYPES[self[0].dtype],
        BLOCK_SIZE,
        has_out,
        pack_out,
        self[0].element_size(),
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return None if inplace else outputs


def _foreach_exp_(self):
    logger.debug("GEMS_GCU400 _FOREACH_EXP_")

    if FALLBACK_TO_PUBLIC or not can_use_flat(self, None, None):
        return apply_unary("exp", self, inplace=True)
    return _run_exp(self, True)


def _foreach_exp(self):
    logger.debug("GEMS_GCU400 _FOREACH_EXP")

    if FALLBACK_TO_PUBLIC or not can_use_flat(self, None, None):
        return apply_unary("exp", self, inplace=False)
    return _run_exp(self, False)
