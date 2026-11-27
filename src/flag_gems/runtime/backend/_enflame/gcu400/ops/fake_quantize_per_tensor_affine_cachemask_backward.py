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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_GRID_X = 65535
# Per-program on-chip (DSM) staging for this kernel is roughly
# BLOCK * (itemsize + 1) bytes (grad tile + byte mask; the store reuses the
# grad slot).  The device budget is 896KB: fp32 at 256K elements needs 1.25MB
# and aborts with "dsm memory Required: 1310720, Hardware limit: 917504", so
# block sizes must stay under this per-dtype cap.
_DSM_BYTES = 917504

# Tuned on GCU400 (L600) with the harness protocol (kernel-mode do_bench
# median, warmup 300ms / rep 100ms, min of 3 rounds).  Buckets are keyed by
# element count; entries are (upper_bound, block, unroll, warps).  Small
# tensors are launch-floor bound (~3.1us); mid tensors want many small-tile
# programs (a single-CTA 128K-element tile loses up to 3x there, e.g. 98304
# fp32); huge tensors are bandwidth bound (~3.0 TB/s ceiling, native ATen
# already at ~95% of it), where few large programs with 2 warps win.
_LADDER_2B = [
    (8192, 1024, 1, 2),
    (65536, 4096, 1, 2),
    (262144, 8192, 1, 8),
    (524288, 16384, 1, 4),
    (1048576, 16384, 2, 4),
    (8388608, 131072, 1, 4),
    (None, 65536, 1, 2),
]
_LADDER_4B = [
    (8192, 1024, 1, 2),
    (65536, 4096, 1, 4),
    (262144, 8192, 1, 4),
    (524288, 8192, 2, 4),
    (1048576, 8192, 4, 4),
    (8388608, 32768, 1, 2),
    (None, 65536, 1, 2),
]
_LADDERS = {2: _LADDER_2B, 4: _LADDER_4B}


def _dsm_block_cap(itemsize):
    cap = _DSM_BYTES // (itemsize + 1)
    return max(1024, 1 << (cap.bit_length() - 1))


def _pick_config(n, itemsize):
    ladder = _LADDERS.get(itemsize, _LADDER_4B)
    for bound, block, unroll, warps in ladder:
        if bound is None or n <= bound:
            break
    # Escalate the per-program span until grid.x fits the 65535 cap.  Growing
    # unroll keeps the per-iteration DSM staging unchanged; growing block is
    # bounded by the DSM budget.  Only element counts no real tensor reaches
    # (>2**31) exhaust the ladder, so this is a guard, not a tuned path.
    while (n + block * unroll - 1) // (block * unroll) > _MAX_GRID_X:
        if block < _dsm_block_cap(itemsize):
            block = min(block * 2, _dsm_block_cap(itemsize))
        else:
            unroll *= 2
    return block, unroll, warps


@libentry()
@triton.jit
def fake_quantize_per_tensor_affine_cachemask_backward_kernel(
    grad_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    # Once the grid.x cap stops bounding the program count, element offsets
    # past the int32 domain (n_elements + BLOCK * UNROLL > 2**31) must be
    # computed in int64 to avoid wrap-around loads/stores on tail lanes.
    if WIDE:
        base = pid.to(tl.int64) * (BLOCK * UNROLL)
    else:
        base = pid * (BLOCK * UNROLL)
    for u in tl.static_range(UNROLL):
        offsets = base + u * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            grad = tl.load(grad_ptr + offsets)
            quant_mask = tl.load(mask_ptr + offsets)
            output = grad * quant_mask
            tl.store(output_ptr + offsets, output)
        else:
            active = offsets < n_elements
            grad = tl.load(grad_ptr + offsets, mask=active, other=0.0)
            quant_mask = tl.load(mask_ptr + offsets, mask=active, other=0)
            output = grad * quant_mask
            tl.store(output_ptr + offsets, output, mask=active)


def fake_quantize_per_tensor_affine_cachemask_backward(grad, mask):
    logger.debug("GEMS_ENFLAME FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_BACKWARD")
    grad = grad.contiguous()
    mask = mask.contiguous()
    output = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return output

    # A bool mask is byte-addressed by the kernel, so it is viewed rather than
    # converted: the value 0/1 is exactly what the multiply needs (bitwise
    # exact for finite inputs, -0.0 sign and inf*0=NaN included).
    kernel_mask = mask.view(torch.uint8) if mask.dtype == torch.bool else mask

    block, unroll, warps = _pick_config(n_elements, grad.element_size())
    span = block * unroll
    nblocks = (n_elements + span - 1) // span
    with torch_device_fn.device(grad.device):
        fake_quantize_per_tensor_affine_cachemask_backward_kernel[(nblocks,)](
            grad,
            kernel_mask,
            output,
            n_elements,
            EVEN=n_elements % span == 0,
            WIDE=n_elements > 2**31 - span,
            BLOCK=block,
            UNROLL=unroll,
            num_warps=warps,
        )
    return output
