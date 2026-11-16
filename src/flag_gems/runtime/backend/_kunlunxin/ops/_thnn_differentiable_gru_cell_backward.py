# Copyright 2026 The FlagOS Contributors.
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
#
# Kunlunxin specialization of ``aten::_thnn_differentiable_gru_cell_backward``.
#
# The generic implementation launches an elementwise gate/hidden backward
# kernel and a per-gate bias reduction over the batch axis.  The elementwise
# kernel is fine on XPU3, but the bias reduction silently miscomputes both
# bias vectors with ``BLOCK_BATCH=256`` (and 128 for small batch): the masked
# tail poisons the loop-carried accumulators, so e.g. batch=1/hidden=4 returns
# values several units away from ``sum(dim=0)``.  This is the documented XPU3
# ``tl.sum``/loop-carried-accumulator miscompile
# (``wiki/patterns/xpu3-loop-carried-reduce-miscompiles.md``).
#
# A block-size sweep of the same kernel measured ``BLOCK_BATCH<=64`` exact for
# both accumulators at batch in {1,2,5,300} (max diff <= 1e-5 fp32) while 128
# and 256 are wrong for batch < ~128.  Only the reduction width changes;
# validation, the elementwise kernel and the rounding boundaries are reused
# from generic FlagGems (all Triton, no ATen/CPU compute here).

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

_ELEM_BLOCK = 256
# The bias reduction's masked tail miscompiles at 128/256 lanes on XPU3; 64 is
# exact for every probed batch size (see the header note).
_BIAS_BLOCK_BATCH = 64


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
        input_bias_arg = input_bias if input_bias is not None else input_gates
        hidden_bias_arg = hidden_bias if hidden_bias is not None else hidden_gates
        with torch_device_fn.device(grad_hy.device):
            _gru_cell_backward_kernel[(triton.cdiv(n_elements, _ELEM_BLOCK),)](
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
                BLOCK_SIZE=_ELEM_BLOCK,
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
                BLOCK_BATCH=_BIAS_BLOCK_BATCH,
                IS_FP64=input_gates.dtype == torch.float64,
            )
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )
