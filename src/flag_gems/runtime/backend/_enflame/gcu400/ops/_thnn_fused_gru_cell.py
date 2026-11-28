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
from flag_gems.utils import libentry, tl_extra_shim

from .copy import copy_

logger = logging.getLogger(__name__)

# Launch geometry for the fused GRU cell kernel.
# The grid is (row, column block): one program covers a contiguous column slice
# of a single row, so the kernel needs no runtime division/modulo to recover
# (row, column) from a flat linear index. That index arithmetic used to cost
# more than the arithmetic of the cell itself on GCU400.
# NUM_WARPS_GRU: warp count per program (kept as an explicit tuning knob).
NUM_WARPS_GRU = 4
# Row tile cap. One full row fits in a single program up to this width; wider
# rows are split along the column axis of the grid.
ROW_BLOCK_CAP = 2048
# Column tile used for rows narrower than the cap.
DEFAULT_BLOCK_SIZE = 1024

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _select_block_size(hidden_size):
    """Pick the column tile width for one GRU cell row."""
    return ROW_BLOCK_CAP if hidden_size >= ROW_BLOCK_CAP else DEFAULT_BLOCK_SIZE


@libentry()
@triton.jit
def _thnn_fused_gru_cell_kernel(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
    hidden_size,
    input_stride_0,
    input_stride_1,
    hidden_stride_0,
    hidden_stride_1,
    hx_stride_0,
    hx_stride_1,
    input_bias_stride,
    hidden_bias_stride,
    hy_stride_0,
    hy_stride_1,
    workspace_stride_0,
    workspace_stride_1,
    HAS_INPUT_BIAS: tl.constexpr,
    HAS_HIDDEN_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # row = batch index, columns = one contiguous slice of the 3 * hidden gates
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Tell the backend the column slice is a BLOCK_SIZE-aligned contiguous run so
    # it can emit wide, unpredicated vector accesses.
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_SIZE), BLOCK_SIZE)
    mask = cols < hidden_size

    # Gate planes are laid out as [reset | update | new] along the last axis, so
    # the three loads of one operand differ by hidden_size * stride_1.
    input_base = row * input_stride_0 + cols * input_stride_1
    hidden_base = row * hidden_stride_0 + cols * hidden_stride_1
    input_reset = tl.load(input_gates + input_base, mask=mask, other=0.0)
    input_gate = tl.load(
        input_gates + input_base + hidden_size * input_stride_1,
        mask=mask,
        other=0.0,
    )
    input_new = tl.load(
        input_gates + input_base + 2 * hidden_size * input_stride_1,
        mask=mask,
        other=0.0,
    )
    hidden_reset = tl.load(hidden_gates + hidden_base, mask=mask, other=0.0)
    hidden_gate = tl.load(
        hidden_gates + hidden_base + hidden_size * hidden_stride_1,
        mask=mask,
        other=0.0,
    )
    hidden_new = tl.load(
        hidden_gates + hidden_base + 2 * hidden_size * hidden_stride_1,
        mask=mask,
        other=0.0,
    )
    hx_value = tl.load(
        hx + row * hx_stride_0 + cols * hx_stride_1,
        mask=mask,
        other=0.0,
    )

    # Accumulate in fp32 for the reduced-precision dtypes: the activation
    # functions and the gate sum are then evaluated with fp32 accuracy, matching
    # ATen's reference which also upcasts.
    if not IS_FP64:
        input_reset = input_reset.to(tl.float32)
        input_gate = input_gate.to(tl.float32)
        input_new = input_new.to(tl.float32)
        hidden_reset = hidden_reset.to(tl.float32)
        hidden_gate = hidden_gate.to(tl.float32)
        hidden_new = hidden_new.to(tl.float32)
        hx_value = hx_value.to(tl.float32)

    reset_gate = input_reset + hidden_reset
    update_gate = input_gate + hidden_gate
    # The two biases are independent inputs: ATen applies hidden_bias even when
    # input_bias is omitted, so each gets its own constexpr guard.
    if HAS_INPUT_BIAS:
        bias_reset = tl.load(
            input_bias + cols * input_bias_stride, mask=mask, other=0.0
        )
        bias_update = tl.load(
            input_bias + hidden_size * input_bias_stride + cols * input_bias_stride,
            mask=mask,
            other=0.0,
        )
        bias_new = tl.load(
            input_bias + 2 * hidden_size * input_bias_stride + cols * input_bias_stride,
            mask=mask,
            other=0.0,
        )
        if not IS_FP64:
            bias_reset = bias_reset.to(tl.float32)
            bias_update = bias_update.to(tl.float32)
            bias_new = bias_new.to(tl.float32)
        reset_gate += bias_reset
        update_gate += bias_update
        input_new += bias_new
    if HAS_HIDDEN_BIAS:
        hidden_bias_reset = tl.load(
            hidden_bias + cols * hidden_bias_stride, mask=mask, other=0.0
        )
        hidden_bias_update = tl.load(
            hidden_bias + hidden_size * hidden_bias_stride + cols * hidden_bias_stride,
            mask=mask,
            other=0.0,
        )
        hidden_bias_new = tl.load(
            hidden_bias
            + 2 * hidden_size * hidden_bias_stride
            + cols * hidden_bias_stride,
            mask=mask,
            other=0.0,
        )
        if not IS_FP64:
            hidden_bias_reset = hidden_bias_reset.to(tl.float32)
            hidden_bias_update = hidden_bias_update.to(tl.float32)
            hidden_bias_new = hidden_bias_new.to(tl.float32)
        # Addition order is kept identical to ATen: input bias then hidden bias.
        reset_gate += hidden_bias_reset
        update_gate += hidden_bias_update
        hidden_new += hidden_bias_new

    reset_gate = tl.sigmoid(reset_gate)
    update_gate = tl.sigmoid(update_gate)
    new_gate = input_new + reset_gate * hidden_new
    new_gate = tl_extra_shim.tanh(new_gate)
    hy_value = new_gate + update_gate * (hx_value - new_gate)

    tl.store(
        hy + row * hy_stride_0 + cols * hy_stride_1,
        hy_value,
        mask=mask,
    )
    # Workspace layout consumed by the fused backward: five planes holding
    # r, z, n, hx and (r * hidden_new) respectively.
    workspace_base = row * workspace_stride_0 + cols * workspace_stride_1
    tl.store(workspace + workspace_base, reset_gate, mask=mask)
    tl.store(
        workspace + workspace_base + hidden_size * workspace_stride_1,
        update_gate,
        mask=mask,
    )
    tl.store(
        workspace + workspace_base + 2 * hidden_size * workspace_stride_1,
        new_gate,
        mask=mask,
    )
    tl.store(
        workspace + workspace_base + 3 * hidden_size * workspace_stride_1,
        hx_value,
        mask=mask,
    )
    tl.store(
        workspace + workspace_base + 4 * hidden_size * workspace_stride_1,
        hidden_new,
        mask=mask,
    )


def _validate_tensor(tensor, shape, dtype, device, name):
    if tensor.ndim != len(shape) or tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(f"{name} must have shape {tuple(shape)}")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}")
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on {device}")


def _validate_inputs(input_gates, hidden_gates, hx, input_bias, hidden_bias):
    if input_gates.ndim != 2:
        raise RuntimeError("input_gates must be a 2-D tensor")
    if input_gates.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "_thnn_fused_gru_cell only supports FP16, BF16, FP32, and FP64"
        )
    batch_size, gate_size = input_gates.shape
    if gate_size % 3 != 0:
        raise RuntimeError("input_gates.size(1) must be divisible by 3")
    hidden_size = gate_size // 3
    dtype = input_gates.dtype
    device = input_gates.device
    _validate_tensor(
        hidden_gates, (batch_size, gate_size), dtype, device, "hidden_gates"
    )
    _validate_tensor(hx, (batch_size, hidden_size), dtype, device, "hx")
    if input_bias is not None:
        _validate_tensor(input_bias, (gate_size,), dtype, device, "input_bias")
        if hidden_bias is None:
            raise RuntimeError("hidden_bias must be defined when input_bias is defined")
    if hidden_bias is not None:
        _validate_tensor(hidden_bias, (gate_size,), dtype, device, "hidden_bias")
    return batch_size, hidden_size


def _prepare_out(out, shape, dtype, device, name):
    if out.dtype != dtype:
        raise RuntimeError(f"Expected {name} to have dtype {dtype}, got {out.dtype}")
    if out.device != device:
        raise RuntimeError(f"Expected {name} to be on {device}, got {out.device}")
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)


def _shares_storage(left, right):
    return (
        left.device == right.device
        and left.untyped_storage().nbytes() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _launch_gru_cell(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
):
    batch_size, hidden_size = hx.shape
    n_elements = batch_size * hidden_size
    if n_elements == 0:
        return
    has_input_bias = input_bias is not None
    # `hidden_bias must be defined when input_bias is defined` (ATen): the
    # hidden bias is only meaningful alongside an input bias, and a
    # hidden-bias-only call must behave exactly like the biasless one
    # (`test_accuracy_thnn_fused_gru_cell_hidden_bias_without_input_bias`).
    has_hidden_bias = hidden_bias is not None and has_input_bias
    block_size = _select_block_size(hidden_size)
    # One program per (row, column tile); reads and writes stay coalesced along
    # the contiguous gate axis and no linear-index decomposition is needed.
    grid = (batch_size, triton.cdiv(hidden_size, block_size))
    with torch_device_fn.device(input_gates.device):
        _thnn_fused_gru_cell_kernel[grid](
            input_gates,
            hidden_gates,
            hx,
            input_bias if has_input_bias else hx,
            hidden_bias if has_hidden_bias else hx,
            hy,
            workspace,
            hidden_size,
            input_gates.stride(0),
            input_gates.stride(1),
            hidden_gates.stride(0),
            hidden_gates.stride(1),
            hx.stride(0),
            hx.stride(1),
            input_bias.stride(0) if has_input_bias else 0,
            hidden_bias.stride(0) if has_hidden_bias else 0,
            hy.stride(0),
            hy.stride(1),
            workspace.stride(0),
            workspace.stride(1),
            HAS_INPUT_BIAS=has_input_bias,
            HAS_HIDDEN_BIAS=has_hidden_bias,
            IS_FP64=input_gates.dtype == torch.float64,
            BLOCK_SIZE=block_size,
            num_warps=NUM_WARPS_GRU,
        )


def _thnn_fused_gru_cell(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
):
    logger.debug("GEMS_ENFLAME _THNN_FUSED_GRU_CELL")
    batch_size, hidden_size = _validate_inputs(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )
    options = {"dtype": input_gates.dtype, "device": input_gates.device}
    hy = torch.empty((batch_size, hidden_size), **options)
    workspace = torch.empty((batch_size, 5 * hidden_size), **options)
    _launch_gru_cell(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        hy,
        workspace,
    )
    return hy, workspace


def _thnn_fused_gru_cell_out(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
):
    logger.debug("GEMS_ENFLAME _THNN_FUSED_GRU_CELL.OUT")
    batch_size, hidden_size = _validate_inputs(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )
    if _shares_storage(out0, out1):
        raise RuntimeError("out0 and out1 must not share storage")
    aliases_input = any(
        operand is not None and _shares_storage(output, operand)
        for output in (out0, out1)
        for operand in (input_gates, hidden_gates, hx, input_bias, hidden_bias)
    )
    temporary = None
    if aliases_input:
        # An output overlapping an input must be written only after the inputs
        # are consumed, so compute into fresh buffers first.
        temporary = _thnn_fused_gru_cell(
            input_gates, hidden_gates, hx, input_bias, hidden_bias
        )
    _prepare_out(
        out0,
        (batch_size, hidden_size),
        input_gates.dtype,
        input_gates.device,
        "out0",
    )
    _prepare_out(
        out1,
        (batch_size, 5 * hidden_size),
        input_gates.dtype,
        input_gates.device,
        "out1",
    )
    if temporary is not None:
        copy_(out0, temporary[0])
        copy_(out1, temporary[1])
        return out0, out1
    _launch_gru_cell(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        out0,
        out1,
    )
    return out0, out1
