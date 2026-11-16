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

"""GCU400-specific ``_foreach_log2`` (out-of-place).

The common ``foreach_unary`` executor carries the chunk table in a 2D grid, which on GCU400 hits the
grid.y<=255 limit and needs 64-bit offsets in metadata. Two Triton paths are provided here:

* ``FUSED_LAUNCH=1``: pack the tensor list's (input pointer, output pointer, element count) into
  int64 metadata and process all tensors in one launch with a 1D grid, removing per-tensor
  launch overhead; this is the main path for this operator on GCU400.
* ``FUSED_LAUNCH=0``: per-tensor launches of a 1D fixed-grid striding kernel, kept as
  a fallback and performance reference.

Non-float, complex or mixed dtype inputs fall back to common ``apply_unary`` to preserve ATen's
dtype promotion policy (integral / bool promoted to float32).
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time config, adjusted per optimization round; semantics unchanged.
FUSED_LAUNCH = 1
BLOCK_SIZE = 16384
FIXED_GRID = 48
NUM_WARPS = 4
MAX_CHUNKS_CAP = 0
USE_DEVICE_CONTEXT = 0
USE_MULTIPLE_OF = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_log2_out_kernel(
    src_ptr,
    dst_ptr,
    N,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """Per-tensor launch: 1D fixed-grid striding that reads input and writes log2."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
        tl.store(dst_ptr + offs, tl.log2(value), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["NT", "CHUNKS"])
def _foreach_log2_fused_kernel(
    meta_ptr,
    NT,
    CHUNKS,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_MULTIPLE_OF: tl.constexpr,
):
    """Single launch: metadata layout is [input pointers x NT, output pointers x NT, element counts x NT].

    The 1D grid flattens ``(tensor_id, chunk_id)`` to avoid GCU400's grid.y
    limit; programs beyond a tensor's element count are fully masked off.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for task_id in tl.range(pid, NT * CHUNKS, num_programs):
        tensor_id = task_id // CHUNKS
        chunk_id = task_id % CHUNKS
        in_v = tl.load(meta_ptr + tensor_id)
        out_v = tl.load(meta_ptr + NT + tensor_id)
        n_elements = tl.load(meta_ptr + 2 * NT + tensor_id)
        idx = chunk_id * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        if USE_MULTIPLE_OF and (in_v & 15) == 0 and (out_v & 15) == 0:
            in_ptr = tl.multiple_of(in_v.to(tl.pointer_type(IN_DT)), 16)
            out_ptr = tl.multiple_of(out_v.to(tl.pointer_type(IN_DT)), 16)
        else:
            in_ptr = in_v.to(tl.pointer_type(IN_DT))
            out_ptr = out_v.to(tl.pointer_type(IN_DT))
        value = tl.load(in_ptr + idx, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + idx, tl.log2(value), mask=mask)


def _foreach_log2(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2")
    try:
        all_float = all(
            tensor.dtype in _FLOAT_DTYPES and not tensor.dtype.is_complex
            for tensor in self
        )
        homogeneous = len({tensor.dtype for tensor in self}) == 1
        # The fused kernel addresses memory flat via data_ptr()/numel(), so non-contiguous views would write to wrong
        # locations.
        contiguous = all(tensor.is_contiguous() for tensor in self)
    except AttributeError:
        # Non-Tensor elements: fall back to the common executor to match ATen's TypeError/RuntimeError.
        return apply_unary("log2", self, inplace=False)
    if FALLBACK_TO_PUBLIC or not (all_float and homogeneous and contiguous):
        return apply_unary("log2", self, inplace=False)

    # Out-of-place needs a separate same-dtype / same-shape output for each input tensor;
    # the output list order must match self.
    outputs = [torch.empty_like(tensor) for tensor in self]
    device = self[0].device
    device_context = (
        torch_device_fn.device(device) if USE_DEVICE_CONTEXT else nullcontext()
    )

    if FUSED_LAUNCH:
        tensor_count = len(self)
        # Metadata stores device pointers as int64, so GCU 64-bit support must be enabled, otherwise
        # torch_gcu downgrades int64 to int32 and truncates the pointers.
        metadata = torch.empty(3 * tensor_count, dtype=torch.int64)
        for index, (src_tensor, dst_tensor) in enumerate(zip(self, outputs)):
            metadata[index] = src_tensor.data_ptr()
            metadata[tensor_count + index] = dst_tensor.data_ptr()
            metadata[2 * tensor_count + index] = src_tensor.numel()
        metadata = metadata.to(device)
        max_numel = max(tensor.numel() for tensor in self)
        chunks = triton.cdiv(max_numel, BLOCK_SIZE)
        if MAX_CHUNKS_CAP:
            chunks = min(chunks, MAX_CHUNKS_CAP)
        grid_size = min(tensor_count * chunks, FIXED_GRID * tensor_count)
        with device_context:
            _foreach_log2_fused_kernel[(grid_size,)](
                metadata,
                tensor_count,
                chunks,
                _TL_DTYPES[self[0].dtype],
                BLOCK_SIZE,
                USE_MULTIPLE_OF,
                num_warps=NUM_WARPS,
            )
        return outputs

    launch_pairs = list(zip(self, outputs))
    with device_context:
        for src_tensor, dst_tensor in launch_pairs:
            num_elems = src_tensor.numel()
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_log2_out_kernel[grid](
                src_tensor,
                dst_tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
            )
    return outputs
