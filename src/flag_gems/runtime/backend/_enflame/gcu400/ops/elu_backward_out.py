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

"""GCU400 specialization of ``aten::elu_backward``'s out variant.

``elu_backward_out`` is a three-stream elementwise kernel (grad_output load,
self_or_result load, grad_input store) and is therefore purely bandwidth bound.
The portable module launches ``ceil(numel / 1024)`` one-shot programs, which on
GCU400 leaves the tail SPC waves idle and grows grid.x with the problem size.
This copy keeps a persistent grid capped at two programs per SPC and marches each
program over the flat element range with a wide tile, plus an exact-tile variant
that drops the element predicate when ``numel`` divides the tile width.  The
non-contiguous fallback reuses the portable pointwise wrapper so strided inputs
keep working unchanged.
"""

import logging

import triton
import triton.language as tl

from flag_gems.ops.elu import (
    elu_backward_grad_input_kernel_with_result,
    elu_backward_grad_input_kernel_with_self,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
GRID_CAP = NUM_SIPS * 2
SMALL_MAX = 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements", "alpha", "scale", "input_scale"])
def _elu_backward_out_kernel(
    grad_output_ptr,
    self_or_result_ptr,
    grad_input_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    IS_RESULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offs = block_id * BLOCK + arange
        mask = offs < n_elements
        grad_output = tl.load(grad_output_ptr + offs, mask=mask).to(tl.float32)
        src = tl.load(self_or_result_ptr + offs, mask=mask).to(tl.float32)
        if IS_RESULT:
            # `src` holds the forward output y = elu(x); the slack slope is
            # recovered from y + alpha * scale, so no exp is needed.
            grad_input = tl.where(
                src <= 0,
                grad_output * input_scale * (src + alpha * scale),
                grad_output * scale,
            )
        else:
            # `src` holds the input x.  Clamping before exp keeps lanes that are
            # discarded by the select from overflowing to +inf.
            grad_input = tl.where(
                src <= 0,
                grad_output
                * input_scale
                * alpha
                * scale
                * tl.exp(tl.minimum(src, 0.0) * input_scale),
                grad_output * scale,
            )
        tl.store(
            grad_input_ptr + offs,
            grad_input.to(grad_input_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit(do_not_specialize=["n_elements", "alpha", "scale", "input_scale"])
def _elu_backward_out_kernel_exact(
    grad_output_ptr,
    self_or_result_ptr,
    grad_input_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    IS_RESULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = n_elements // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offs = block_id * BLOCK + arange
        grad_output = tl.load(grad_output_ptr + offs).to(tl.float32)
        src = tl.load(self_or_result_ptr + offs).to(tl.float32)
        if IS_RESULT:
            grad_input = tl.where(
                src <= 0,
                grad_output * input_scale * (src + alpha * scale),
                grad_output * scale,
            )
        else:
            grad_input = tl.where(
                src <= 0,
                grad_output
                * input_scale
                * alpha
                * scale
                * tl.exp(tl.minimum(src, 0.0) * input_scale),
                grad_output * scale,
            )
        tl.store(grad_input_ptr + offs, grad_input.to(grad_input_ptr.dtype.element_ty))


def _launch_params(n_elements, element_size):
    """Pick (BLOCK, num_warps) for the flat element range.

    Small ranges pack the whole span into few programs; large ranges use a wide
    tile whose element count is halved for 4-byte payloads so a tile keeps the
    same byte footprint.
    """
    if n_elements <= SMALL_MAX:
        tile = triton.next_power_of_2(triton.cdiv(n_elements, NUM_SIPS))
        block = max(1024, min(tile, 8192))
        return block, 2
    # Wide tiles amortise the strided loop bookkeeping, but a 4-byte tile must
    # stay at half the element count of a 2-byte tile: three live buffers of
    # 32768 * 4 B already take 384 KiB of the 896 KiB DSM budget, and 65536 *
    # 4 B tiles fail the dsm allocation.
    if element_size > 2:
        return 32768, 4
    return 65536, 4


def elu_backward_out(
    grad_output, alpha, scale, input_scale, is_result, self_or_result, *, grad_input
):
    logger.debug("GEMS_ENFLAME ELU_BACKWARD_OUT")
    n_elements = grad_output.numel()
    if n_elements == 0:
        return grad_input

    if (
        grad_output.is_contiguous()
        and self_or_result.is_contiguous()
        and grad_input.is_contiguous()
    ):
        block, num_warps = _launch_params(n_elements, grad_output.element_size())
        grid = (min(triton.cdiv(n_elements, block), GRID_CAP),)
        kernel = (
            _elu_backward_out_kernel_exact
            if n_elements % block == 0
            else _elu_backward_out_kernel
        )
        with torch_device_fn.device(grad_output.device):
            kernel[grid](
                grad_output,
                self_or_result,
                grad_input,
                n_elements,
                float(alpha),
                float(scale),
                float(input_scale),
                IS_RESULT=bool(is_result),
                BLOCK=block,
                num_warps=num_warps,
            )
        return grad_input

    if is_result:
        return elu_backward_grad_input_kernel_with_result(
            grad_output,
            alpha,
            scale,
            input_scale,
            self_or_result,
            out0=grad_input,
        )
    return elu_backward_grad_input_kernel_with_self(
        grad_output,
        alpha,
        scale,
        input_scale,
        self_or_result,
        out0=grad_input,
    )
