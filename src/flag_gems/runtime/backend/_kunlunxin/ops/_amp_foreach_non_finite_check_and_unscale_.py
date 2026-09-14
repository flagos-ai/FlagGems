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

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_isnanf = tl_extra_shim.isnan
_isinff = tl_extra_shim.isinf

# Fixed block sizes (no autotune: XPU recompiles per shape under autotune and
# the accumulated IR dumps explode). Two hand-dispatched tiles instead
# (HARNESS pattern: explicit per-shape dispatch beats autotune):
# - 4096 for larger tensors: cuts the program count 4x versus 1024; the large
#   comprehensive shapes were launch-bound (25k tiny programs for
#   4096x4096 + 2048x4096, 6.8ms -> 2.5ms fp16 / 1.9ms fp32).
# - 1024 for small tensors: a single 4096-wide program serializes small
#   tensors onto one program ([1024,4] fp32: 0.025ms -> 0.048ms).
# Both keep the per-program `tl.max` reduction within the XPU-safe block
# limit (<= 8192).
BLOCK_SIZE_SMALL = 1024
BLOCK_SIZE_LARGE = 4096
LARGE_TENSOR_NUMEL = 65536

# Block size of the tiny found_inf fixup kernel that scans the per-program
# bad-value flags of a whole foreach call (a few int32 loads per program).
FOUND_INF_REDUCE_BLOCK = 1024


@triton.jit
def _amp_foreach_non_finite_check_and_unscale_kernel(
    inp_ptr,
    inv_scale_ptr,
    bad_flags_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(inv_scale_ptr).to(tl.float32)

    inp_fp32 = inp.to(tl.float32)

    # NOTE (kunlunxin/XPU): `~tl_extra_shim.finitef(x)` is unreliable in this
    # kernel -- finite values get reported as non-finite, so every element
    # stays unscaled. Detect non-finite explicitly with libdevice isnan/isinf
    # (the same reliable primitives the backend isnan/isinf overrides use).
    is_non_finite = _isnanf(inp_fp32) | _isinff(inp_fp32)

    # Non-finite values stay as-is; finite values are scaled (ATen semantics).
    scaled_fp32 = tl.where(is_non_finite, inp_fp32, inp_fp32 * scale)

    scaled = scaled_fp32.to(inp.dtype)
    tl.store(output_ptr + offsets, scaled, mask=mask)

    # found_inf detection fused into the same kernel: check the *stored* value
    # (converted back to fp32), which is exactly what the previous separate
    # post-pass (`torch.isnan(t).any() or torch.isinf(t).any()`) observed --
    # including values that overflow to inf during the dtype conversion.
    # Masked lanes load `other=0.0` (finite) and never store, so they cannot
    # raise a false positive.
    #
    # NOTE (kunlunxin/XPU): the per-block flag cannot be reduced and fed into
    # an atomic on this backend (atomic_max/atomic_add with a block-derived
    # value operand fail to compile: "Failed to tune buffer size"; only
    # constant-value atomics compile, and masked atomics fail to parse).
    # Instead each program stores its 0/1 flag into its own slot of a small
    # flags buffer (unconditional store of every slot: no stale data across
    # calls), and a single tiny fixup kernel folds the flags into found_inf.
    stored_fp32 = scaled.to(tl.float32)
    is_bad = _isnanf(stored_fp32) | _isinff(stored_fp32)
    block_bad = tl.max(is_bad.to(tl.int32), axis=0)
    tl.store(bad_flags_ptr + pid, block_bad, mask=True)


@triton.jit
def _amp_foreach_found_inf_reduce_kernel(
    bad_flags_ptr,
    found_inf_ptr,
    num_flags,
    BLOCK_SIZE: tl.constexpr,
):
    # Single-program fold of the per-program flags into found_inf, mirroring
    # the previous host-side semantics: found_inf <- 1.0 if any non-finite
    # value was seen in any input tensor, otherwise found_inf is untouched.
    any_bad = 0
    for start in range(0, num_flags, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        flag_mask = offsets < num_flags
        flags = tl.load(bad_flags_ptr + offsets, mask=flag_mask, other=0)
        any_bad = any_bad | tl.max(flags, axis=0)

    current = tl.load(found_inf_ptr)
    updated = tl.where(any_bad > 0, 1.0, current)
    tl.store(found_inf_ptr, updated)


def _amp_foreach_non_finite_check_and_unscale_(
    tensors: List[torch.Tensor],
    found_inf: torch.Tensor,
    inv_scale: torch.Tensor,
):
    """
    Check for non-finite values in tensors and unscale them (in-place).

    For each tensor in the list:
    - Scale finite values by inv_scale; non-finite (inf, nan) stay unchanged.
    - If any element is non-finite, set found_inf to 1.0.
    """
    logger.debug("GEMS_KUNLUNXIN AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE")

    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"Expected list or tuple of tensors, got {type(tensors)}")

    if len(tensors) == 0:
        return

    # PyTorch expects inv_scale as a float32 tensor.
    inv_scale = inv_scale.to(dtype=torch.float32)

    # The unscale kernel and the found_inf detection are fused into a single
    # launch per tensor. The previous implementation needed ~6 launches per
    # tensor (empty_like alloc + unscale kernel + _copy_from write-back, then
    # isnan/isinf/any post-passes over the full data), which made this foreach
    # op launch- and traffic-bound (full-tensor bool materializations of
    # isnan/isinf alone quadrupled the memory traffic of the op call).
    jobs = []
    for tensor in tensors:
        if not tensor.is_floating_point():
            continue

        num_elements = tensor.numel()
        if num_elements == 0:
            continue

        block = (
            BLOCK_SIZE_LARGE
            if num_elements >= LARGE_TENSOR_NUMEL
            else BLOCK_SIZE_SMALL
        )
        num_programs = triton.cdiv(num_elements, block)
        jobs.append((tensor, num_elements, num_programs, block))

    if not jobs:
        return

    total_programs = sum(job[2] for job in jobs)
    # torch.empty only allocates (no kernel launch); every flag slot is
    # unconditionally written by its own program, so no zero-init is needed.
    bad_flags = torch.empty(
        total_programs, dtype=torch.int32, device=inv_scale.device
    )

    flag_offset = 0
    for tensor, num_elements, num_programs, block in jobs:
        grid = (num_programs,)

        if tensor.is_contiguous():
            # Fast path: the store target aliases the load source, which is
            # safe: every program loads its own disjoint region before
            # storing it.
            _amp_foreach_non_finite_check_and_unscale_kernel[grid](
                tensor,
                inv_scale,
                bad_flags[flag_offset:],
                tensor,
                num_elements,
                BLOCK_SIZE=block,
            )
        else:
            # Non-contiguous inputs: the kernel addresses elements by flat
            # offset, so write a dense temporary and route the strided
            # write-back through the ATen `_copy_from` primitive (gems never
            # overrides `_copy_from`, so this reaches the vendor's native
            # strided-copy engine).
            output = torch.empty_like(tensor)
            _amp_foreach_non_finite_check_and_unscale_kernel[grid](
                tensor,
                inv_scale,
                bad_flags[flag_offset:],
                output,
                num_elements,
                BLOCK_SIZE=block,
            )
            torch.ops.aten._copy_from(output, tensor, False)

        flag_offset += num_programs

    # One tiny fixup kernel per foreach call folds the flags into found_inf
    # (a few int32 loads per program slot; ordering is guaranteed by the
    # in-stream program order of the launches above).
    _amp_foreach_found_inf_reduce_kernel[(1,)](
        bad_flags,
        found_inf,
        total_programs,
        BLOCK_SIZE=FOUND_INF_REDUCE_BLOCK,
    )
