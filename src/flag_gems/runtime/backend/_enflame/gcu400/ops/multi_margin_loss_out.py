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
"""GCU400-specific ``multi_margin_loss.out``.

The common ``_compute_forward`` launches a row kernel with ``grid=min(N, 65535)``,
creating thousands of programs on GCU400 (4 DIE x 6 SIP), with scheduling and DTE launch overhead
far exceeding the actual compute. Keep the same Triton compute kernel and only change launch strategy to
a fixed grid + in-kernel striding, exposing ``BLOCK_C`` / ``num_warps`` / reduction block size
as tunable parameters; all other input validation, dtype policy and out semantics match the common implementation.
"""

import logging

import torch
import triton

from flag_gems.ops.multi_margin_loss import (
    _REDUCE_BLOCK_SIZE,
    _USE_DEVICE_ASSERT,
    _check_inputs,
    _empty_forward,
    _multi_margin_loss_kernel,
    _multi_margin_loss_reduce_kernel,
    _normalize_p,
    _normalize_reduction,
    _output_shape,
)
from flag_gems.ops.multi_margin_loss import (
    multi_margin_loss_out as _public_multi_margin_loss_out,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---


# Overwritten per round by the optimization driver script; all compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
NUM_WARPS = 4
BLOCK_C_CAP = 1024
REDUCE_BLOCK = 512
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


def _compute_forward_gcu400(
    input,
    target,
    weight,
    N,
    C,
    is_batched,
    p,
    margin,
    reduction,
    output_shape,
    destination,
):
    """Launch the row loss kernel with a fixed grid, then a reduce kernel if reduction is needed."""
    if N == 0:
        return _empty_forward(input, N, is_batched, reduction)

    output = (
        destination
        if destination is not None
        else torch.empty(output_shape, dtype=input.dtype, device=input.device)
    )
    # One program per row exceeds the SIP topology; fix the program count and stride over rows in-kernel.
    grid = (min(N, FIXED_GRID),)
    block_c = min(BLOCK_C_CAP, triton.next_power_of_2(C))
    weight_ptr = input if weight is None else weight
    has_weight = weight is not None

    needs_reduction = is_batched and N > 1 and reduction != 0
    if needs_reduction:
        # Per-row results accumulate in fp32 to avoid error amplification from a second reduction in low precision.
        row_losses = torch.empty((N,), dtype=torch.float32, device=input.device)
    else:
        row_losses = output

    with torch_device_fn.device(input.device):
        _multi_margin_loss_kernel[grid](
            input,
            target,
            weight_ptr,
            row_losses,
            N,
            C,
            margin,
            P=p,
            HAS_WEIGHT=has_weight,
            USE_DEVICE_ASSERT=_USE_DEVICE_ASSERT,
            BLOCK_C=block_c,
            num_warps=NUM_WARPS,
        )
        if needs_reduction:
            _multi_margin_loss_reduce_kernel[(1,)](
                row_losses,
                output,
                N,
                REDUCTION=reduction,
                BLOCK_SIZE=REDUCE_BLOCK or _REDUCE_BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
    return output


def multi_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    p=1,
    margin=1,
    weight=None,
    reduction=1,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_GCU400 MULTI_MARGIN_LOSS_OUT")
    if FALLBACK_TO_PUBLIC:
        return _public_multi_margin_loss_out(
            input, target, p, margin, weight, reduction, out=out
        )
    p = _normalize_p(p)
    reduction = _normalize_reduction(reduction)
    try:
        margin = float(margin)
    except (TypeError, ValueError) as error:
        raise RuntimeError("multi_margin_loss: margin must be a real scalar") from error
    input, target, weight, N, C, is_batched = _check_inputs(input, target, weight)
    if out.device != input.device:
        raise RuntimeError("multi_margin_loss.out: out must be on the input device")
    if out.dtype != input.dtype:
        raise RuntimeError("multi_margin_loss.out: out must have the input dtype")

    shape = _output_shape(N, is_batched, reduction)
    if tuple(out.shape) != shape:
        out.resize_(shape)
    destination = out if out.is_contiguous() else None
    result = _compute_forward_gcu400(
        input,
        target,
        weight,
        N,
        C,
        is_batched,
        p,
        margin,
        reduction,
        shape,
        destination,
    )
    if result is not out:
        out.copy_(result)
    return out
