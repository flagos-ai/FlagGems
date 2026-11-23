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

"""GCU400-specific ``_thnn_fused_gru_cell_backward(_out)``.

Row-streaming redesign (2026-11-19 perf round).  Measured on GCU400 L600:

- Rank-2 tile memory ops carry a large fixed cost per program while rank-1
  contiguous loads/stores do not; a program that streams whole rows issues
  13 rank-1 ops per row (every workspace gate segment is contiguous inside a
  row), which is latency-friendly on tiny shapes and bandwidth-friendly on
  large ones (e.g. 256x1024 fp16 backward: 71.7us flat kernel / 26.8us
  column-tile kernel -> 8.5us row-streaming kernel).
- The device prefers <= 48 concurrent programs each moving large chunks, so
  the kernel launches ``min(48, B)`` programs that stride over batch rows.
- Bias gradients are accumulated in registers per program (partial buffer
  shape ``(nprog, 4, H)`` fp32) and finalized by a tiny ``(col_block, gate)``
  2-D-grid kernel; ``B == 1`` stores the bias directly from the single-row
  kernel and skips the finalize launch entirely.
- ``next_pow2(H) > 4096`` (DSM budget for the live fp32 row vectors) or fp64
  inputs fall back to the previous column-partitioned fused kernel below.

To match the reference numeric convention, gate gradients are quantized to
the output dtype before bias accumulation, reproducing the common
"write back quantized tensors, then read and sum" order.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.thnn_fused_gru_cell_backward import (
    _prepare_out,
)
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
GCU_NUM_GRID = 48
MAX_ROW_BLOCK_H = 4096
NUM_WARPS_SMALL = 1
NUM_WARPS_LARGE = 2
NUM_WARPS_LARGE_MIN_ELEMS = 32768
BIAS_FIN_BLOCK_H = 128
# --- TUNING PARAMS END ---


@triton.jit
def _gru_bwd_rows_body(
    grad_hy_ptr,
    workspace_ptr,
    grad_input_ptr,
    grad_hidden_ptr,
    grad_hx_ptr,
    bias_a_ptr,
    bias_b_ptr,
    B,
    H,
    NPROG,
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
    MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # MODE: 0 = no bias, 1 = per-program fp32 partials, 2 = direct bias store (B == 1).
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < H
    if MODE == 1:
        sum_reset = tl.zeros((BLOCK_H,), dtype=tl.float32)
        sum_input = tl.zeros((BLOCK_H,), dtype=tl.float32)
        sum_new = tl.zeros((BLOCK_H,), dtype=tl.float32)
        sum_hidden_new = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for row in range(pid, B, NPROG):
        grad_hy_value = tl.load(
            grad_hy_ptr + row * grad_hy_stride_0 + cols * grad_hy_stride_1,
            mask=col_mask,
            other=0.0,
        )
        ws_base = row * workspace_stride_0 + cols * workspace_stride_1
        reset_gate = tl.load(workspace_ptr + ws_base, mask=col_mask, other=0.0)
        input_gate = tl.load(
            workspace_ptr + ws_base + H * workspace_stride_1,
            mask=col_mask,
            other=0.0,
        )
        new_gate = tl.load(
            workspace_ptr + ws_base + 2 * H * workspace_stride_1,
            mask=col_mask,
            other=0.0,
        )
        hx = tl.load(
            workspace_ptr + ws_base + 3 * H * workspace_stride_1,
            mask=col_mask,
            other=0.0,
        )
        hidden_new = tl.load(
            workspace_ptr + ws_base + 4 * H * workspace_stride_1,
            mask=col_mask,
            other=0.0,
        )
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

        out_base = row * grad_input_stride_0 + cols * grad_input_stride_1
        hidden_base = row * grad_hidden_stride_0 + cols * grad_hidden_stride_1
        tl.store(grad_input_ptr + out_base, grad_reset, mask=col_mask)
        tl.store(
            grad_input_ptr + out_base + H * grad_input_stride_1,
            grad_input_gate,
            mask=col_mask,
        )
        tl.store(
            grad_input_ptr + out_base + 2 * H * grad_input_stride_1,
            grad_new_input,
            mask=col_mask,
        )
        tl.store(grad_hidden_ptr + hidden_base, grad_reset, mask=col_mask)
        tl.store(
            grad_hidden_ptr + hidden_base + H * grad_hidden_stride_1,
            grad_input_gate,
            mask=col_mask,
        )
        tl.store(
            grad_hidden_ptr + hidden_base + 2 * H * grad_hidden_stride_1,
            grad_hidden_new,
            mask=col_mask,
        )
        tl.store(
            grad_hx_ptr + row * grad_hx_stride_0 + cols * grad_hx_stride_1,
            grad_hx_value,
            mask=col_mask,
        )
        if MODE == 2:
            # B == 1: the single row's quantized values are the bias sums.
            output_dtype = grad_input_ptr.dtype.element_ty
            tl.store(bias_a_ptr + cols, grad_reset.to(output_dtype), mask=col_mask)
            tl.store(
                bias_a_ptr + H + cols, grad_input_gate.to(output_dtype), mask=col_mask
            )
            tl.store(
                bias_a_ptr + 2 * H + cols,
                grad_new_input.to(output_dtype),
                mask=col_mask,
            )
            tl.store(bias_b_ptr + cols, grad_reset.to(output_dtype), mask=col_mask)
            tl.store(
                bias_b_ptr + H + cols, grad_input_gate.to(output_dtype), mask=col_mask
            )
            tl.store(
                bias_b_ptr + 2 * H + cols,
                grad_hidden_new.to(output_dtype),
                mask=col_mask,
            )
        elif MODE == 1:
            output_dtype = grad_input_ptr.dtype.element_ty
            sum_reset += grad_reset.to(output_dtype).to(tl.float32)
            sum_input += grad_input_gate.to(output_dtype).to(tl.float32)
            sum_new += grad_new_input.to(output_dtype).to(tl.float32)
            sum_hidden_new += grad_hidden_new.to(output_dtype).to(tl.float32)
    if MODE == 1:
        partial_base = pid * (4 * H) + cols
        tl.store(bias_a_ptr + partial_base, sum_reset, mask=col_mask)
        tl.store(bias_a_ptr + partial_base + H, sum_input, mask=col_mask)
        tl.store(bias_a_ptr + partial_base + 2 * H, sum_new, mask=col_mask)
        tl.store(bias_a_ptr + partial_base + 3 * H, sum_hidden_new, mask=col_mask)


@libentry()
@triton.jit
def _gru_cell_backward_rows_kernel(
    grad_hy_ptr,
    workspace_ptr,
    grad_input_ptr,
    grad_hidden_ptr,
    grad_hx_ptr,
    bias_a_ptr,
    bias_b_ptr,
    B,
    H,
    NPROG,
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
    MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    _gru_bwd_rows_body(
        grad_hy_ptr,
        workspace_ptr,
        grad_input_ptr,
        grad_hidden_ptr,
        grad_hx_ptr,
        bias_a_ptr,
        bias_b_ptr,
        B,
        H,
        NPROG,
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
        MODE=MODE,
        BLOCK_H=BLOCK_H,
    )


@libentry()
@triton.jit
def _gru_cell_backward_rows_contig_kernel(
    grad_hy_ptr,
    workspace_ptr,
    grad_input_ptr,
    grad_hidden_ptr,
    grad_hx_ptr,
    bias_a_ptr,
    bias_b_ptr,
    B,
    H,
    NPROG,
    MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Contiguous fast path: a short argument list keeps the Python launch cost
    # low enough to stay hidden under the benchmark's cache-flush window.
    _gru_bwd_rows_body(
        grad_hy_ptr,
        workspace_ptr,
        grad_input_ptr,
        grad_hidden_ptr,
        grad_hx_ptr,
        bias_a_ptr,
        bias_b_ptr,
        B,
        H,
        NPROG,
        H,
        1,
        5 * H,
        1,
        3 * H,
        1,
        3 * H,
        1,
        H,
        1,
        MODE=MODE,
        BLOCK_H=BLOCK_H,
    )


@libentry()
@triton.jit
def _gru_bias_finalize_kernel(
    partial_ptr,
    grad_input_bias_ptr,
    grad_hidden_bias_ptr,
    NPROG,
    H,
    BLOCK_H: tl.constexpr,
):
    # 2-D grid: axis 0 = column block, axis 1 = gate (0 reset, 1 input, 2 new, 3 hidden-new).
    # Bias outputs are always contiguous here (freshly allocated or staged).
    col_block = tl.program_id(0)
    gate = tl.program_id(1)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < H
    total = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for program in range(0, NPROG):
        total += tl.load(
            partial_ptr + program * (4 * H) + gate * H + cols,
            mask=col_mask,
            other=0.0,
        )
    value = total.to(grad_input_bias_ptr.dtype.element_ty)
    if gate == 0:
        tl.store(grad_input_bias_ptr + cols, value, mask=col_mask)
        tl.store(grad_hidden_bias_ptr + cols, value, mask=col_mask)
    elif gate == 1:
        tl.store(grad_input_bias_ptr + H + cols, value, mask=col_mask)
        tl.store(grad_hidden_bias_ptr + H + cols, value, mask=col_mask)
    elif gate == 2:
        tl.store(grad_input_bias_ptr + 2 * H + cols, value, mask=col_mask)
    else:
        tl.store(grad_hidden_bias_ptr + 2 * H + cols, value, mask=col_mask)


def _launch_backward(
    grad_hy,
    workspace,
    grad_input,
    grad_hidden,
    grad_hx,
    grad_input_bias,
    grad_hidden_bias,
    has_bias,
):
    """Launch the row-streaming kernels; both overloads share this entry.

    Degenerate shapes (empty tensors), fp64 inputs and rows wider than
    ``MAX_ROW_BLOCK_H`` are routed to the two-step common launcher.
    """
    from .thnn_fused_gru_cell_backward__common import _launch_gru_cell_backward

    batch_size, hidden_size = grad_hy.shape
    numel = batch_size * hidden_size
    if numel == 0:
        if has_bias and hidden_size != 0:
            fin_block_h = min(BIAS_FIN_BLOCK_H, triton.next_power_of_2(hidden_size))
            with torch_device_fn.device(grad_hy.device):
                _gru_bias_finalize_kernel[(triton.cdiv(hidden_size, fin_block_h), 4)](
                    grad_hx,
                    grad_input_bias,
                    grad_hidden_bias,
                    0,
                    hidden_size,
                    BLOCK_H=fin_block_h,
                    num_warps=1,
                )
        return
    if grad_hy.dtype == torch.float64 or (
        triton.next_power_of_2(hidden_size) > MAX_ROW_BLOCK_H
    ):
        _launch_gru_cell_backward(
            grad_hy,
            workspace,
            has_bias,
            grad_input,
            grad_hidden,
            grad_hx,
            grad_input_bias,
            grad_hidden_bias,
        )
        return

    nprog = min(GCU_NUM_GRID, batch_size)
    block_h = triton.next_power_of_2(hidden_size)
    num_warps = (
        NUM_WARPS_LARGE if numel >= NUM_WARPS_LARGE_MIN_ELEMS else NUM_WARPS_SMALL
    )
    if not has_bias:
        mode = 0
        bias_a = grad_hx
        bias_b = grad_hx
    elif batch_size == 1:
        mode = 2
        bias_a = grad_input_bias
        bias_b = grad_hidden_bias
    else:
        mode = 1
        bias_a = torch.empty(
            (nprog, 4 * hidden_size), dtype=torch.float32, device=grad_hy.device
        )
        bias_b = grad_input_bias

    with torch_device_fn.device(grad_hy.device):
        contiguous = (
            grad_hy.is_contiguous()
            and workspace.is_contiguous()
            and grad_input.is_contiguous()
            and grad_hidden.is_contiguous()
            and grad_hx.is_contiguous()
        )
        if contiguous:
            _gru_cell_backward_rows_contig_kernel[(nprog,)](
                grad_hy,
                workspace,
                grad_input,
                grad_hidden,
                grad_hx,
                bias_a,
                bias_b,
                batch_size,
                hidden_size,
                nprog,
                MODE=mode,
                BLOCK_H=block_h,
                num_warps=num_warps,
            )
        else:
            _gru_cell_backward_rows_kernel[(nprog,)](
                grad_hy,
                workspace,
                grad_input,
                grad_hidden,
                grad_hx,
                bias_a,
                bias_b,
                batch_size,
                hidden_size,
                nprog,
                grad_hy.stride(0),
                grad_hy.stride(1),
                workspace.stride(0),
                workspace.stride(1),
                grad_input.stride(0),
                grad_input.stride(1),
                grad_hidden.stride(0),
                grad_hidden.stride(1),
                grad_hx.stride(0),
                grad_hx.stride(1),
                MODE=mode,
                BLOCK_H=block_h,
                num_warps=num_warps,
            )
        if mode == 1:
            fin_block_h = min(BIAS_FIN_BLOCK_H, block_h)
            _gru_bias_finalize_kernel[(triton.cdiv(hidden_size, fin_block_h), 4)](
                bias_a,
                grad_input_bias,
                grad_hidden_bias,
                nprog,
                hidden_size,
                BLOCK_H=fin_block_h,
                num_warps=1,
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
    logger.debug("GEMS_ENFLAME _THNN_FUSED_GRU_CELL_BACKWARD_OUT")
    if BASELINE_MODE == 1:
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
        if hidden_size != 0:
            fin_block_h = min(BIAS_FIN_BLOCK_H, triton.next_power_of_2(hidden_size))
            with torch_device_fn.device(grad_hy.device):
                _gru_bias_finalize_kernel[(triton.cdiv(hidden_size, fin_block_h), 4)](
                    out2,
                    out3,
                    out4,
                    0,
                    hidden_size,
                    BLOCK_H=fin_block_h,
                    num_warps=1,
                )
        return out0, out1, out2, out3, out4

    # Non-contiguous out tensors are written through a contiguous staging copy,
    # keeping the identity of out (same workaround as the previous release: direct
    # strided stores from the kernel miss the last column of gate blocks on v4.0).
    out_tensors = (out0, out1, out2, out3, out4)
    if all(tensor.is_contiguous() for tensor in out_tensors):
        _launch_backward(
            grad_hy, workspace, out0, out1, out2, out3, out4, has_bias=True
        )
        return out_tensors

    staging = tuple(
        torch.empty_like(tensor, memory_format=torch.contiguous_format)
        for tensor in out_tensors
    )
    _launch_backward(grad_hy, workspace, *staging, has_bias=True)
    for destination, source in zip(out_tensors, staging):
        destination.copy_(source)
    return out_tensors
