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
# Kunlunxin specialization of ``aten::_thnn_differentiable_lstm_cell_backward``.
#
# The generic implementation launches two Triton kernels: an elementwise
# gate/cell backward pass and a per-gate bias reduction over the batch axis.
# The elementwise kernel is numerically correct on XPU3 (outputs 0-2 match
# ATen to ~2e-7 fp32), but the bias reduction is miscompiled: with
# ``BLOCK_BATCH=256`` (and 128) the loop-carried accumulator reduction returns
# garbage whenever most of the lanes are masked off, e.g. ``batch=2,
# hidden=17`` gives ``max|grad_bias - ref| = 4.8`` with per-element ratios that
# are not a constant scale factor.  This is the documented XPU3
# ``tl.sum``/loop-carried-accumulator miscompile
# (``wiki/patterns/xpu3-loop-carried-reduce-miscompiles.md``): the reduction
# width must stay small enough that the masked tail cannot poison it.
#
# A block-size sweep of the *same* kernel measured ``BLOCK_BATCH<=64`` exact
# (``max|diff| <= 2.3e-5 fp32`` for batch in {1,2,5,300,1024,2048}) while 128/256
# are wrong for batch < ~128.  64 is therefore used for the bias reduction;
# everything else (validation, kernel bodies, rounding boundaries, the
# differentiable/composite path needed for higher-order graphs) is reused from
# the generic FlagGems implementation -- no ATen/CPU compute runs here.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.thnn_differentiable_lstm_cell_backward import (
    _differentiable_lstm_cell_backward as _composite_backward,
    _lstm_bias_backward_kernel,
    _validate_inputs,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_ELEM_BLOCK = 256
# The bias reduction's masked tail miscompiles at 128/256 lanes on XPU3; 64 is
# exact for every probed batch size (see the header note).
_BIAS_BLOCK_BATCH = 64


@libentry()
@triton.jit
def _lstm_cell_backward_full_kernel(
    grad_hy,
    grad_cy,
    input_gates,
    hidden_gates,
    input_bias,
    hidden_bias,
    cx,
    cy,
    grad_gates,
    grad_cx,
    batch_size,
    hidden_size,
    grad_hy_stride_0,
    grad_hy_stride_1,
    grad_cy_stride_0,
    grad_cy_stride_1,
    input_gates_stride_0,
    input_gates_stride_1,
    hidden_gates_stride_0,
    hidden_gates_stride_1,
    input_bias_stride,
    hidden_bias_stride,
    cx_stride_0,
    cx_stride_1,
    cy_stride_0,
    cy_stride_1,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD_HY: tl.constexpr,
    HAS_GRAD_CY: tl.constexpr,
    HAS_INPUT_BIAS: tl.constexpr,
    HAS_HIDDEN_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    raw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = batch_size * hidden_size
    mask = raw < n_elements
    # Clamp the tail lanes onto the last element so every load stays in bounds
    # without a mask; only the stores are masked.
    offsets = tl.minimum(raw, n_elements - 1)
    rows = offsets // hidden_size
    cols = offsets % hidden_size

    input_offsets = rows * input_gates_stride_0 + cols * input_gates_stride_1
    hidden_offsets = rows * hidden_gates_stride_0 + cols * hidden_gates_stride_1

    gate_i = tl.load(input_gates + input_offsets)
    gate_f = tl.load(
        input_gates + input_offsets + hidden_size * input_gates_stride_1
    )
    gate_c = tl.load(
        input_gates + input_offsets + 2 * hidden_size * input_gates_stride_1
    )
    gate_o = tl.load(
        input_gates + input_offsets + 3 * hidden_size * input_gates_stride_1
    )
    # Sum the four gates in fp32 so the pre-activation matches the fp64 oracle;
    # the following `+=` then promotes the fp16/bf16 operand automatically.
    if not IS_FP64:
        gate_i = gate_i.to(tl.float32)
        gate_f = gate_f.to(tl.float32)
        gate_c = gate_c.to(tl.float32)
        gate_o = gate_o.to(tl.float32)
    gate_i += tl.load(hidden_gates + hidden_offsets)
    gate_f += tl.load(
        hidden_gates + hidden_offsets + hidden_size * hidden_gates_stride_1
    )
    gate_c += tl.load(
        hidden_gates + hidden_offsets + 2 * hidden_size * hidden_gates_stride_1
    )
    gate_o += tl.load(
        hidden_gates + hidden_offsets + 3 * hidden_size * hidden_gates_stride_1
    )

    # All math below runs in fp32 (the reference is an fp64 CPU oracle); the
    # bf16/fp16 intermediate roundings of the generic kernel are deliberately
    # dropped because they pushed single bf16 elements past the test tolerance.

    if HAS_INPUT_BIAS:
        gate_i += tl.load(input_bias + cols * input_bias_stride)
        gate_f += tl.load(
            input_bias + (hidden_size + cols) * input_bias_stride
        )
        gate_c += tl.load(
            input_bias + (2 * hidden_size + cols) * input_bias_stride
        )
        gate_o += tl.load(
            input_bias + (3 * hidden_size + cols) * input_bias_stride
        )

    if HAS_HIDDEN_BIAS:
        gate_i += tl.load(hidden_bias + cols * hidden_bias_stride)
        gate_f += tl.load(
            hidden_bias + (hidden_size + cols) * hidden_bias_stride
        )
        gate_c += tl.load(
            hidden_bias + (2 * hidden_size + cols) * hidden_bias_stride
        )
        gate_o += tl.load(
            hidden_bias + (3 * hidden_size + cols) * hidden_bias_stride
        )

    if not IS_FP64:
        gate_i = gate_i.to(tl.float32)
        gate_f = gate_f.to(tl.float32)
        gate_c = gate_c.to(tl.float32)
        gate_o = gate_o.to(tl.float32)
    input_gate = tl.sigmoid(gate_i)
    forget_gate = tl.sigmoid(gate_f)
    cell_gate = tl_extra_shim.tanh(gate_c)
    output_gate = tl.sigmoid(gate_o)

    state_offsets = rows * cx_stride_0 + cols * cx_stride_1
    cx_value = tl.load(cx + state_offsets)
    cy_value = tl.load(
        cy + rows * cy_stride_0 + cols * cy_stride_1
    )
    cy_tanh = tl_extra_shim.tanh(cy_value if IS_FP64 else cy_value.to(tl.float32))

    if HAS_GRAD_HY:
        grad_hy_value = tl.load(
            grad_hy + rows * grad_hy_stride_0 + cols * grad_hy_stride_1
        )
        grad_output_seed = grad_hy_value * cy_tanh
        grad_cell_seed = grad_hy_value * output_gate
        if not IS_FP64:
            grad_output_seed = grad_output_seed.to(tl.float32)
            grad_cell_seed = grad_cell_seed.to(tl.float32)
            output_gate_fp = output_gate.to(tl.float32)
            cy_tanh_fp = cy_tanh.to(tl.float32)
        else:
            output_gate_fp = output_gate
            cy_tanh_fp = cy_tanh
        grad_output = grad_output_seed * (1.0 - output_gate_fp) * output_gate_fp
        grad_cell = grad_cell_seed * (1.0 - cy_tanh_fp * cy_tanh_fp)
        if HAS_GRAD_CY:
            grad_cy_value = tl.load(
                grad_cy + rows * grad_cy_stride_0 + cols * grad_cy_stride_1
            )
            grad_cell += grad_cy_value
    else:
        grad_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        grad_cell = tl.load(
            grad_cy + rows * grad_cy_stride_0 + cols * grad_cy_stride_1
        )

    grad_input_seed = grad_cell * cell_gate
    grad_forget_seed = grad_cell * cx_value
    grad_candidate_seed = grad_cell * input_gate
    next_grad_cx = grad_cell * forget_gate

    if not IS_FP64:
        grad_input_seed = grad_input_seed.to(tl.float32)
        grad_forget_seed = grad_forget_seed.to(tl.float32)
        grad_candidate_seed = grad_candidate_seed.to(tl.float32)
        input_gate_fp = input_gate.to(tl.float32)
        forget_gate_fp = forget_gate.to(tl.float32)
        cell_gate_fp = cell_gate.to(tl.float32)
    else:
        input_gate_fp = input_gate
        forget_gate_fp = forget_gate
        cell_gate_fp = cell_gate
    grad_input = grad_input_seed * (1.0 - input_gate_fp) * input_gate_fp
    grad_forget = grad_forget_seed * (1.0 - forget_gate_fp) * forget_gate_fp
    grad_candidate = grad_candidate_seed * (1.0 - cell_gate_fp * cell_gate_fp)

    output_offsets = rows * (4 * hidden_size) + cols
    tl.store(grad_gates + output_offsets, grad_input, mask=mask)
    tl.store(grad_gates + output_offsets + hidden_size, grad_forget, mask=mask)
    tl.store(
        grad_gates + output_offsets + 2 * hidden_size,
        grad_candidate,
        mask=mask,
    )
    tl.store(
        grad_gates + output_offsets + 3 * hidden_size,
        grad_output,
        mask=mask,
    )
    tl.store(grad_cx + rows * hidden_size + cols, next_grad_cx, mask=mask)


def _thnn_differentiable_lstm_cell_backward(
    grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy
):
    """Compute the five differentiable LSTM-cell backward outputs."""
    logger.debug("GEMS _THNN_DIFFERENTIABLE_LSTM_CELL_BACKWARD")

    # Native returns a tuple of undefined tensors without inspecting the other
    # arguments when neither output gradient is defined.
    if grad_hy is None and grad_cy is None:
        return None, None, None, None, None

    _validate_inputs(
        grad_hy,
        grad_cy,
        input_gates,
        hidden_gates,
        input_bias,
        hidden_bias,
        cx,
        cy,
    )
    differentiable_inputs = (
        grad_hy,
        grad_cy,
        input_gates,
        hidden_gates,
        input_bias,
        hidden_bias,
        cx,
        cy,
    )
    if torch.is_grad_enabled() and any(
        value is not None and value.requires_grad for value in differentiable_inputs
    ):
        # The higher-order graph needs the composite autograd path; it is the
        # generic FlagGems implementation, not an ATen/CPU fallback.
        return _composite_backward(*differentiable_inputs)

    batch_size, hidden_size = cx.shape
    gate_size = 4 * hidden_size
    grad_gates = torch.empty(
        (batch_size, gate_size),
        dtype=input_gates.dtype,
        device=input_gates.device,
    )
    grad_cx = torch.empty_like(cx, memory_format=torch.contiguous_format)

    n_elements = batch_size * hidden_size
    if n_elements != 0:
        input_bias_arg = input_bias if input_bias is not None else input_gates
        hidden_bias_arg = hidden_bias if hidden_bias is not None else hidden_gates
        grad_hy_arg = grad_hy if grad_hy is not None else cx
        grad_cy_arg = grad_cy if grad_cy is not None else cx
        # Mask-free loads with a clamped tail and masked stores: the
        # always-masked load path is several times slower on XPU3, and the
        # clamp+mask combination is safe outside dot kernels.
        with torch_device_fn.device(input_gates.device):
            _lstm_cell_backward_full_kernel[
                (triton.cdiv(n_elements, _ELEM_BLOCK),)
            ](
                grad_hy_arg,
                grad_cy_arg,
                input_gates,
                hidden_gates,
                input_bias_arg,
                hidden_bias_arg,
                cx,
                cy,
                grad_gates,
                grad_cx,
                batch_size,
                hidden_size,
                grad_hy_arg.stride(0),
                grad_hy_arg.stride(1),
                grad_cy_arg.stride(0),
                grad_cy_arg.stride(1),
                input_gates.stride(0),
                input_gates.stride(1),
                hidden_gates.stride(0),
                hidden_gates.stride(1),
                input_bias_arg.stride(0),
                hidden_bias_arg.stride(0),
                cx.stride(0),
                cx.stride(1),
                cy.stride(0),
                cy.stride(1),
                BLOCK_SIZE=_ELEM_BLOCK,
                HAS_GRAD_HY=grad_hy is not None,
                HAS_GRAD_CY=grad_cy is not None,
                HAS_INPUT_BIAS=input_bias is not None,
                HAS_HIDDEN_BIAS=hidden_bias is not None,
                IS_FP64=input_gates.dtype == torch.float64,
                IS_FP16=input_gates.dtype == torch.float16,
                IS_BF16=input_gates.dtype == torch.bfloat16,
            )

    # ATen intentionally keys both optional bias gradients on input_bias.
    if input_bias is None:
        return grad_gates, grad_gates, grad_cx, None, None

    grad_bias = torch.empty(
        (gate_size,), dtype=input_gates.dtype, device=input_gates.device
    )
    if gate_size != 0:
        with torch_device_fn.device(input_gates.device):
            _lstm_bias_backward_kernel[(gate_size,)](
                grad_gates,
                grad_bias,
                batch_size,
                gate_size,
                BLOCK_BATCH=_BIAS_BLOCK_BATCH,
                IS_FP64=input_gates.dtype == torch.float64,
            )
    return grad_gates, grad_gates, grad_cx, grad_bias, grad_bias
