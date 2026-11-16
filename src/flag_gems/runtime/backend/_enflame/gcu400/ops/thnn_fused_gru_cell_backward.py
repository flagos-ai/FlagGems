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

"""GCU400-specific ``_thnn_fused_gru_cell_backward_out``.

The common implementation uses two steps: the main kernel writes 5 gradient tensors by (batch,hidden) element, then a
separate
bias reduction kernel (stage1+stage2 for large batch) rereads those tensors and
sums along batch, so the bias path rereads 2xBx3H elements. This implementation partitions
CTAs by hidden column block: one program computes all four gate gradients in registers, writes all tensors, and
reduces bias along batch in the same kernel, finishing all outputs in one launch.

To match the engineering CPU reference's numeric convention, each gate gradient is converted back to the output dtype
and then to the accumulation dtype before bias accumulation, reproducing the common "write back quantized tensors, then
read and sum" order.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.thnn_fused_gru_cell_backward import _prepare_out
from flag_gems.ops.thnn_fused_gru_cell_backward import (
    _thnn_fused_gru_cell_backward_out as _public_out,
)
from flag_gems.ops.thnn_fused_gru_cell_backward import _validate_inputs
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Compile-time config, adjusted per optimization round; semantics unchanged.
BASELINE_MODE = 0
FUSED_SINGLE_LAUNCH = 1
BLOCK_B = 128
BLOCK_H = 128
NUM_WARPS = 2
BIAS_QUANTIZE = 1
# --- TUNING PARAMS END ---


@libentry()
@triton.jit(do_not_specialize=["B", "H"])
def _gru_cell_backward_fused_kernel(
    grad_hy_ptr,
    workspace_ptr,
    grad_input_ptr,
    grad_hidden_ptr,
    grad_hx_ptr,
    grad_input_bias_ptr,
    grad_hidden_bias_ptr,
    B,
    H,
    grad_hy_stride_0,
    grad_hy_stride_1,
    workspace_stride_0,
    workspace_stride_1,
    grad_input_stride_0,
    grad_input_stride_1,
    grad_hidden_stride_0,
    grad_hidden_stride_1,
    grad_hx_stride_0,
    grad_hx_stride_1,
    grad_input_bias_stride,
    grad_hidden_bias_stride,
    IS_FP64: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BIAS_QUANTIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Each program owns a hidden column segment; rows loop over batch in BLOCK_B chunks,
    # so any batch_size is supported without cross-CTA atomic reduction.
    cols = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < H
    accumulator_dtype = tl.float64 if IS_FP64 else tl.float32
    output_dtype = grad_input_ptr.dtype.element_ty

    # With has_bias=False the whole bias reduction is eliminated at compile time and bias pointers are not dereferenced
    # (the non-out overload does not allocate out3/out4 without bias).
    bias_input_reset = tl.zeros((BLOCK_H,), accumulator_dtype)
    bias_input_input = tl.zeros((BLOCK_H,), accumulator_dtype)
    bias_input_new = tl.zeros((BLOCK_H,), accumulator_dtype)
    bias_hidden_reset = tl.zeros((BLOCK_H,), accumulator_dtype)
    bias_hidden_input = tl.zeros((BLOCK_H,), accumulator_dtype)
    bias_hidden_new = tl.zeros((BLOCK_H,), accumulator_dtype)

    for batch_start in range(0, B, BLOCK_B):
        rows = batch_start + tl.arange(0, BLOCK_B)
        valid = (rows[:, None] < B) & col_mask[None, :]
        grad_hy_offset = (
            rows[:, None] * grad_hy_stride_0 + cols[None, :] * grad_hy_stride_1
        )
        workspace_offset = (
            rows[:, None] * workspace_stride_0 + cols[None, :] * workspace_stride_1
        )
        grad_hy_value = tl.load(grad_hy_ptr + grad_hy_offset, mask=valid, other=0.0)
        reset_gate = tl.load(workspace_ptr + workspace_offset, mask=valid, other=0.0)
        input_gate = tl.load(
            workspace_ptr + workspace_offset + H * workspace_stride_1,
            mask=valid,
            other=0.0,
        )
        new_gate = tl.load(
            workspace_ptr + workspace_offset + 2 * H * workspace_stride_1,
            mask=valid,
            other=0.0,
        )
        hx = tl.load(
            workspace_ptr + workspace_offset + 3 * H * workspace_stride_1,
            mask=valid,
            other=0.0,
        )
        hidden_new = tl.load(
            workspace_ptr + workspace_offset + 4 * H * workspace_stride_1,
            mask=valid,
            other=0.0,
        )
        if not IS_FP64:
            grad_hy_value = grad_hy_value.to(tl.float32)
            reset_gate = reset_gate.to(tl.float32)
            input_gate = input_gate.to(tl.float32)
            new_gate = new_gate.to(tl.float32)
            hx = hx.to(tl.float32)
            hidden_new = hidden_new.to(tl.float32)

        # Keep the same multiplication association order as ATen / the common implementation.
        grad_input_gate = grad_hy_value * (hx - new_gate)
        grad_input_gate *= 1.0 - input_gate
        grad_input_gate *= input_gate
        grad_hx_value = grad_hy_value * input_gate
        grad_new_input = grad_hy_value * (1.0 - input_gate)
        grad_new_input *= 1.0 - new_gate * new_gate
        grad_hidden_new = grad_new_input * reset_gate
        grad_reset = grad_new_input * hidden_new
        grad_reset *= 1.0 - reset_gate
        grad_reset *= reset_gate

        grad_input_offset = (
            rows[:, None] * grad_input_stride_0 + cols[None, :] * grad_input_stride_1
        )
        grad_hidden_offset = (
            rows[:, None] * grad_hidden_stride_0 + cols[None, :] * grad_hidden_stride_1
        )
        tl.store(grad_input_ptr + grad_input_offset, grad_reset, mask=valid)
        tl.store(
            grad_input_ptr + grad_input_offset + H * grad_input_stride_1,
            grad_input_gate,
            mask=valid,
        )
        tl.store(
            grad_input_ptr + grad_input_offset + 2 * H * grad_input_stride_1,
            grad_new_input,
            mask=valid,
        )
        tl.store(grad_hidden_ptr + grad_hidden_offset, grad_reset, mask=valid)
        tl.store(
            grad_hidden_ptr + grad_hidden_offset + H * grad_hidden_stride_1,
            grad_input_gate,
            mask=valid,
        )
        tl.store(
            grad_hidden_ptr + grad_hidden_offset + 2 * H * grad_hidden_stride_1,
            grad_hidden_new,
            mask=valid,
        )
        grad_hx_offset = (
            rows[:, None] * grad_hx_stride_0 + cols[None, :] * grad_hx_stride_1
        )
        tl.store(grad_hx_ptr + grad_hx_offset, grad_hx_value, mask=valid)

        # The common implementation writes back quantized gradient tensors first, then the bias kernel reads and
        # accumulates;
        # here to(output_dtype).to(accumulator_dtype) reproduces that quantization step.
        if not HAS_BIAS:
            pass
        elif BIAS_QUANTIZE:
            sum_input_reset = tl.sum(
                grad_reset.to(output_dtype).to(accumulator_dtype), axis=0
            )
            sum_input_input = tl.sum(
                grad_input_gate.to(output_dtype).to(accumulator_dtype), axis=0
            )
            sum_input_new = tl.sum(
                grad_new_input.to(output_dtype).to(accumulator_dtype), axis=0
            )
            sum_hidden_reset = sum_input_reset
            sum_hidden_input = sum_input_input
            sum_hidden_new = tl.sum(
                grad_hidden_new.to(output_dtype).to(accumulator_dtype), axis=0
            )
        else:
            sum_input_reset = tl.sum(grad_reset.to(accumulator_dtype), axis=0)
            sum_input_input = tl.sum(grad_input_gate.to(accumulator_dtype), axis=0)
            sum_input_new = tl.sum(grad_new_input.to(accumulator_dtype), axis=0)
            sum_hidden_reset = sum_input_reset
            sum_hidden_input = sum_input_input
            sum_hidden_new = tl.sum(grad_hidden_new.to(accumulator_dtype), axis=0)
        if HAS_BIAS:
            bias_input_reset += sum_input_reset
            bias_input_input += sum_input_input
            bias_input_new += sum_input_new
            bias_hidden_reset += sum_hidden_reset
            bias_hidden_input += sum_hidden_input
            bias_hidden_new += sum_hidden_new

    if HAS_BIAS:
        tl.store(
            grad_input_bias_ptr + cols * grad_input_bias_stride,
            bias_input_reset,
            mask=col_mask,
        )
        tl.store(
            grad_input_bias_ptr + (H + cols) * grad_input_bias_stride,
            bias_input_input,
            mask=col_mask,
        )
        tl.store(
            grad_input_bias_ptr + (2 * H + cols) * grad_input_bias_stride,
            bias_input_new,
            mask=col_mask,
        )
        tl.store(
            grad_hidden_bias_ptr + cols * grad_hidden_bias_stride,
            bias_hidden_reset,
            mask=col_mask,
        )
        tl.store(
            grad_hidden_bias_ptr + (H + cols) * grad_hidden_bias_stride,
            bias_hidden_input,
            mask=col_mask,
        )
        tl.store(
            grad_hidden_bias_ptr + (2 * H + cols) * grad_hidden_bias_stride,
            bias_hidden_new,
            mask=col_mask,
        )


def _launch_fused_kernel(
    grad_hy: torch.Tensor,
    workspace: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    out3: torch.Tensor | None,
    out4: torch.Tensor | None,
    has_bias: bool,
):
    """Single-launch fused kernel, shared by the non-out and out overloads."""
    batch_size, hidden_size = grad_hy.shape
    # With has_bias=False the bias segment is eliminated at compile time; pointers still need to be dereferenceable, use
    # out2 as a placeholder.
    bias_ptr = out2 if not has_bias else out3
    bias_stride = 1 if not has_bias else out3.stride(0)
    hidden_bias_ptr = out2 if not has_bias else out4
    hidden_bias_stride = 1 if not has_bias else out4.stride(0)
    # On small problems (e.g. 1x32) a fixed 128x128 tile idles, so narrow it to a
    # power of two by actual size; large scale still uses the BLOCK_B/BLOCK_H cap to keep vectorization width.
    block_h = min(BLOCK_H, triton.next_power_of_2(max(hidden_size, 1)))
    block_b = min(BLOCK_B, triton.next_power_of_2(max(batch_size, 1)))
    grid = (triton.cdiv(hidden_size, block_h),)
    with torch_device_fn.device(grad_hy.device):
        _gru_cell_backward_fused_kernel[grid](
            grad_hy,
            workspace,
            out0,
            out1,
            out2,
            bias_ptr,
            hidden_bias_ptr,
            batch_size,
            hidden_size,
            grad_hy.stride(0),
            grad_hy.stride(1),
            workspace.stride(0),
            workspace.stride(1),
            out0.stride(0),
            out0.stride(1),
            out1.stride(0),
            out1.stride(1),
            out2.stride(0),
            out2.stride(1),
            bias_stride,
            hidden_bias_stride,
            IS_FP64=grad_hy.dtype == torch.float64,
            BLOCK_B=block_b,
            BLOCK_H=block_h,
            BIAS_QUANTIZE=BIAS_QUANTIZE,
            HAS_BIAS=has_bias,
            num_warps=NUM_WARPS,
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
    """out overload of GRU cell backward, signature matches the common implementation."""
    logger.debug("GEMS_ENFLAME_GCU400 _THNN_FUSED_GRU_CELL_BACKWARD_OUT")
    if BASELINE_MODE == 1 or not FUSED_SINGLE_LAUNCH:
        return _public_out(
            grad_hy,
            workspace,
            has_bias,
            out0=out0,
            out1=out1,
            out2=out2,
            out3=out3,
            out4=out4,
        )

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

    if batch_size * hidden_size == 0:
        return out0, out1, out2, out3, out4

    _launch_fused_kernel(
        grad_hy, workspace, out0, out1, out2, out3, out4, has_bias=True
    )
    return out0, out1, out2, out3, out4
