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
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Fixed Grid on GCU400 (48 CTAs). The generic implementation launches
# cdiv(num_elements, BLOCK_SIZE) CTAs which exceeds the 65535 grid.x
# hardware limit for large tensors (2**28 elements -> 262144 CTAs ->
# OutOfResources). Striding from a fixed grid keeps every launch within
# the limit regardless of tensor size.
_GRID_SIZE = 48

# One kernel launch processes up to _NT_SLOTS tensors of the same
# (device, dtype) group; longer groups are chunked. Each program visits
# every slot's tile space with the same grid stride, so the per-program
# iteration count stays balanced across differently sized tensors.
_NT_SLOTS = 8

# IEEE-754 float32: a value is inf/nan iff its exponent bits (masked
# with 0x7F800000) are all set. The kernel checks this on the PRE-scale
# value, matching aten semantics: found_inf is never set by finite values
# that only overflow after scaling (found_inf is also never cleared).


@triton.jit
def _unscale_detect_tensor(
    inp_ptr,
    found_inf_ptr,
    scale_fp32,
    pid,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MASKED: tl.constexpr,
):
    # Grid-stride over the tiles of a single tensor, scaling in place and
    # accumulating the exponent-bit maximum for the fused non-finite check.
    # MASKED=0 requires num_elements % BLOCK_SIZE == 0 (host guarantees it)
    # and removes every mask predicate from the hot loop.
    if MASKED:
        num_tile = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        num_tile = num_elements // BLOCK_SIZE
    nf_acc = 0
    for tile_id in tl.range(pid, num_tile, GRID_SIZE, num_stages=NUM_STAGES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if MASKED:
            mask = offsets < num_elements
            inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
        else:
            inp = tl.load(inp_ptr + offsets)
        inp_fp32 = inp.to(tl.float32)
        bits = inp_fp32.to(tl.int32, bitcast=True) & 0x7F800000
        scaled = (inp_fp32 * scale_fp32).to(inp.dtype)
        if MASKED:
            tl.store(inp_ptr + offsets, scaled, mask=mask)
        else:
            tl.store(inp_ptr + offsets, scaled)
        nf_acc = tl.maximum(nf_acc, tl.max(bits, axis=0))
    # Only ever set to 1.0 (aten semantics: found_inf is never cleared).
    # Concurrent programs storing the same value race benignly.
    if nf_acc == 0x7F800000:
        tl.store(found_inf_ptr, 1.0)


@libentry()
@triton.jit(do_not_specialize=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
def _amp_foreach_non_finite_check_and_unscale_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    inv_scale_ptr,
    found_inf_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MASKED: tl.constexpr,
):
    # In-place scale + fused non-finite detection for up to 8 tensors in a
    # single launch. Padded slots get size 0, which makes their tile loop a
    # zero-trip loop, so a single compiled variant serves any list length <= 8.
    # Scaling every value unconditionally (instead of passing non-finite
    # values through) is bit-identical for inf/nan and matches aten.
    pid = ext.program_id(0)
    scale = tl.load(inv_scale_ptr)
    scale_fp32 = scale.to(tl.float32)
    _unscale_detect_tensor(
        p0,
        found_inf_ptr,
        scale_fp32,
        pid,
        s0,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p1,
        found_inf_ptr,
        scale_fp32,
        pid,
        s1,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p2,
        found_inf_ptr,
        scale_fp32,
        pid,
        s2,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p3,
        found_inf_ptr,
        scale_fp32,
        pid,
        s3,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p4,
        found_inf_ptr,
        scale_fp32,
        pid,
        s4,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p5,
        found_inf_ptr,
        scale_fp32,
        pid,
        s5,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p6,
        found_inf_ptr,
        scale_fp32,
        pid,
        s6,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )
    _unscale_detect_tensor(
        p7,
        found_inf_ptr,
        scale_fp32,
        pid,
        s7,
        BLOCK_SIZE,
        GRID_SIZE,
        NUM_STAGES,
        MASKED,
    )


def _amp_foreach_non_finite_check_and_unscale_(
    tensors: List[torch.Tensor],
    found_inf: torch.Tensor,
    inv_scale: torch.Tensor,
):
    """
    Check for non-finite values in tensors and unscale them.

    For each floating-point tensor in the list, in one fused kernel pass:
    - Scale every value by inv_scale (inf/nan are invariant under scaling)
    - Set found_inf to 1.0 if the PRE-scale value was inf or nan (aten
      semantics; found_inf is never cleared)

    Complex/integer/empty tensors are skipped, matching the previous
    behavior of this backend implementation.
    """
    logger.debug("GEMS_ENFLAME AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE")

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected list or tuple of tensors, got {type(tensors)}")

    if len(tensors) == 0:
        return

    # Ensure inv_scale is a float32 tensor as PyTorch expects
    inv_scale = inv_scale.to(dtype=torch.float32)

    # Collect the floating-point tensors (in-place: no temporary + copy back)
    work = []
    for tensor in tensors:
        if not tensor.is_floating_point():
            # Skip non-floating point tensors (complex/integer)
            continue
        if tensor.numel() == 0:
            continue
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        work.append(tensor)
    if not work:
        return

    # Group by (device, dtype): one kernel handles one group in a single
    # launch; groups longer than _NT_SLOTS are processed in chunks.
    groups = {}
    for tensor in work:
        groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)

    for (device, _dtype), group in groups.items():
        # Tuned on GCU400 (probe sweep): 2-byte dtypes peak with BLOCK=65536
        # (2.61 TB/s), 4-byte dtypes with BLOCK=32768 (3.16 TB/s); both at
        # num_warps=4 / num_stages=4. BLOCK=131072 exceeds the 917504B DSM
        # limit and is rejected.
        block_even = 65536 if group[0].element_size() <= 2 else 32768
        with torch_device_fn.device(device):
            for start in range(0, len(group), _NT_SLOTS):
                chunk = group[start : start + _NT_SLOTS]
                numels = [t.numel() for t in chunk]
                if all(n % block_even == 0 for n in numels):
                    # Exact-multiple shapes: mask-free hot loop.
                    block, masked = block_even, 0
                else:
                    max_n = max(numels)
                    block = (
                        block_even
                        if max_n >= block_even
                        else triton.next_power_of_2(max_n)
                    )
                    masked = 1
                total_tiles = sum((n + block - 1) // block for n in numels)
                grid = (min(_GRID_SIZE, total_tiles), 1, 1)
                pad = chunk[0]
                _amp_foreach_non_finite_check_and_unscale_kernel[grid](
                    *[chunk[i] if i < len(chunk) else pad for i in range(_NT_SLOTS)],
                    inv_scale,
                    found_inf,
                    *[numels[i] if i < len(numels) else 0 for i in range(_NT_SLOTS)],
                    BLOCK_SIZE=block,
                    GRID_SIZE=grid[0],
                    NUM_STAGES=4,
                    MASKED=masked,
                    num_warps=4,
                )
