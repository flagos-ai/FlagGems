# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.ops._thnn_differentiable_gru_cell_backward import (
    _gru_bias_backward_kernel,
    _gru_cell_backward_kernel,
    _validate_inputs,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# GCU miscompiles this kernel's pair of cross-lane tl.sum reductions when the
# reduction tile has 256 lanes: the value stored for the first reduction
# (grad_input_bias) is corrupted (garbage partial sums, up to ~1e37) whenever
# more than 32 batch rows are active, while grad_hidden_bias stays correct.
# The same kernel launched with a 64- or 128-lane tile is exact, so reduce over
# 64 lanes. The elementwise kernel keeps its 256-lane tile: it never reduces
# across lanes, and every store there is masked.
BIAS_BLOCK_BATCH = 64


def _thnn_differentiable_gru_cell_backward(
    grad_hy: torch.Tensor,
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
):
    """Compute the five differentiable GRU-cell backward outputs."""
    logger.debug("GEMS _THNN_DIFFERENTIABLE_GRU_CELL_BACKWARD")
    _validate_inputs(grad_hy, input_gates, hidden_gates, hx)

    batch_size, hidden_size = hx.shape
    gate_size = 3 * hidden_size
    grad_input_gates = torch.empty(
        (batch_size, gate_size), dtype=input_gates.dtype, device=input_gates.device
    )
    grad_hidden_gates = torch.empty(
        (batch_size, gate_size), dtype=hidden_gates.dtype, device=hidden_gates.device
    )
    grad_hx = torch.empty((batch_size, hidden_size), dtype=hx.dtype, device=hx.device)

    n_elements = batch_size * hidden_size
    if n_elements != 0:
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        input_bias_arg = input_bias if input_bias is not None else input_gates
        hidden_bias_arg = hidden_bias if hidden_bias is not None else hidden_gates
        with torch_device_fn.device(grad_hy.device):
            _gru_cell_backward_kernel[grid](
                grad_hy,
                input_gates,
                hidden_gates,
                hx,
                input_bias_arg,
                hidden_bias_arg,
                grad_input_gates,
                grad_hidden_gates,
                grad_hx,
                batch_size,
                hidden_size,
                grad_hy.stride(0),
                grad_hy.stride(1),
                input_gates.stride(0),
                input_gates.stride(1),
                hidden_gates.stride(0),
                hidden_gates.stride(1),
                hx.stride(0),
                hx.stride(1),
                input_bias_arg.stride(0),
                hidden_bias_arg.stride(0),
                BLOCK_SIZE=block_size,
                HAS_INPUT_BIAS=input_bias is not None,
                HAS_HIDDEN_BIAS=hidden_bias is not None,
                IS_FP64=input_gates.dtype == torch.float64,
                IS_FP16=input_gates.dtype == torch.float16,
                IS_BF16=input_gates.dtype == torch.bfloat16,
            )

    # ATen intentionally keys both optional bias gradients on input_bias.
    if input_bias is None:
        return grad_input_gates, grad_hidden_gates, grad_hx, None, None

    grad_input_bias = torch.empty(
        (gate_size,), dtype=input_gates.dtype, device=input_gates.device
    )
    grad_hidden_bias = torch.empty(
        (gate_size,), dtype=hidden_gates.dtype, device=hidden_gates.device
    )
    if gate_size != 0:
        with torch_device_fn.device(grad_hy.device):
            _gru_bias_backward_kernel[(gate_size,)](
                grad_input_gates,
                grad_hidden_gates,
                grad_input_bias,
                grad_hidden_bias,
                batch_size,
                gate_size,
                BLOCK_BATCH=BIAS_BLOCK_BATCH,
                IS_FP64=input_gates.dtype == torch.float64,
            )
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )
