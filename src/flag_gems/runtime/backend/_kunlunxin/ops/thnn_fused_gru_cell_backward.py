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
# Kunlunxin specialization of ``aten::_thnn_fused_gru_cell_backward`` (and its
# ``.out`` variant).
#
# The generic implementation chooses between two bias reductions: a small-batch
# kernel (``BLOCK_BATCH=256``) and a two-stage coalesced reduction for
# ``batch >= 512``.  Both produce silently wrong bias gradients on XPU3: the
# masked tail poisons the loop-carried ``tl.sum`` accumulators
# (``wiki/patterns/xpu3-loop-carried-reduce-miscompiles.md``).  A block-size
# sweep of the small-batch kernel measured ``BLOCK_BATCH<=64`` exact for both
# accumulators at batch {1,2,8,33,513} (max diff <= 1.5e-5 fp32) while 128/256
# are wrong for batch 1/2/33.
#
# This override therefore always uses the (correct) small-batch reduction at
# ``BLOCK_BATCH=64`` for every batch size; the generic two-stage path is not
# used.  The elementwise kernel and validation are reused unchanged from
# generic FlagGems (all Triton, no ATen/CPU compute here).

import logging

import torch
import triton

from flag_gems.ops.thnn_fused_gru_cell_backward import (
    _gru_bias_backward_kernel_small_batch,
    _gru_cell_backward_kernel,
    _prepare_out,
    _validate_inputs,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_ELEM_BLOCK = 256
# The bias reduction's masked tail miscompiles at 128/256 lanes; 64 is exact
# for every probed batch size (see the header note).
_BIAS_BLOCK_BATCH = 64


def _launch_gru_cell_backward(
    grad_hy,
    workspace,
    has_bias,
    grad_input_gates,
    grad_hidden_gates,
    grad_hx,
    grad_input_bias,
    grad_hidden_bias,
):
    batch_size, hidden_size = grad_hy.shape
    n_elements = batch_size * hidden_size
    with torch_device_fn.device(grad_hy.device):
        if n_elements != 0:
            _gru_cell_backward_kernel[(triton.cdiv(n_elements, _ELEM_BLOCK),)](
                grad_hy,
                workspace,
                grad_input_gates,
                grad_hidden_gates,
                grad_hx,
                batch_size,
                hidden_size,
                grad_hy.stride(0),
                grad_hy.stride(1),
                workspace.stride(0),
                workspace.stride(1),
                grad_input_gates.stride(0),
                grad_input_gates.stride(1),
                grad_hidden_gates.stride(0),
                grad_hidden_gates.stride(1),
                grad_hx.stride(0),
                grad_hx.stride(1),
                BLOCK_SIZE=_ELEM_BLOCK,
                IS_FP64=grad_hy.dtype == torch.float64,
            )
        gate_size = 3 * hidden_size
        if has_bias and gate_size != 0:
            _gru_bias_backward_kernel_small_batch[(gate_size,)](
                grad_input_gates,
                grad_hidden_gates,
                grad_input_bias,
                grad_hidden_bias,
                batch_size,
                grad_input_gates.stride(0),
                grad_input_gates.stride(1),
                grad_hidden_gates.stride(0),
                grad_hidden_gates.stride(1),
                grad_input_bias.stride(0),
                grad_hidden_bias.stride(0),
                BLOCK_BATCH=_BIAS_BLOCK_BATCH,
                IS_FP64=grad_hy.dtype == torch.float64,
            )


def _thnn_fused_gru_cell_backward(
    grad_hy: torch.Tensor, workspace: torch.Tensor, has_bias: bool
):
    logger.debug("GEMS _THNN_FUSED_GRU_CELL_BACKWARD")
    _validate_inputs(grad_hy, workspace)
    batch_size, hidden_size = grad_hy.shape
    gate_shape = (batch_size, 3 * hidden_size)
    options = {"dtype": grad_hy.dtype, "device": grad_hy.device}
    grad_input_gates = torch.empty(gate_shape, **options)
    grad_hidden_gates = torch.empty(gate_shape, **options)
    grad_hx = torch.empty((batch_size, hidden_size), **options)
    if has_bias:
        bias_shape = (3 * hidden_size,)
        grad_input_bias = torch.empty(bias_shape, **options)
        grad_hidden_bias = torch.empty(bias_shape, **options)
    else:
        grad_input_bias = None
        grad_hidden_bias = None

    _launch_gru_cell_backward(
        grad_hy,
        workspace,
        has_bias,
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )
    return (
        grad_input_gates,
        grad_hidden_gates,
        grad_hx,
        grad_input_bias,
        grad_hidden_bias,
    )


def _thnn_fused_gru_cell_backward_out(
    grad_hy: torch.Tensor,
    workspace: torch.Tensor,
    has_bias: bool,
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor,
    out4: torch.Tensor,
):
    logger.debug("GEMS _THNN_FUSED_GRU_CELL_BACKWARD_OUT")
    _validate_inputs(grad_hy, workspace)
    if not has_bias:
        raise RuntimeError("the out overload requires has_bias=True")
    batch_size, hidden_size = grad_hy.shape
    gate_shape = (batch_size, 3 * hidden_size)
    bias_shape = (3 * hidden_size,)
    for out, shape, name in (
        (out0, gate_shape, "out0"),
        (out1, gate_shape, "out1"),
        (out2, (batch_size, hidden_size), "out2"),
        (out3, bias_shape, "out3"),
        (out4, bias_shape, "out4"),
    ):
        _prepare_out(out, shape, grad_hy.dtype, grad_hy.device, name)

    _launch_gru_cell_backward(
        grad_hy,
        workspace,
        True,
        out0,
        out1,
        out2,
        out3,
        out4,
    )
    return out0, out1, out2, out3, out4
