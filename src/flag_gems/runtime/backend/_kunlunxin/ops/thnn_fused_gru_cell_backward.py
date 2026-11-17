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
# Correctness (long-standing): the generic implementation chooses between two
# bias reductions: a small-batch kernel (``BLOCK_BATCH=256``) and a two-stage
# coalesced reduction for ``batch >= 512``.  Both produce silently wrong bias
# gradients on XPU3: the masked tail poisons the loop-carried ``tl.sum``
# accumulators (``wiki/patterns/xpu3-loop-carried-reduce-miscompiles.md``).  A
# block-size sweep of the small-batch kernel measured ``BLOCK_BATCH<=64`` exact
# for both accumulators at batch {1,2,8,33,513} (max diff <= 1.5e-5 fp32).
# The generic two-stage path additionally fails to compile on XPU3
# (``OutOfResources: uni_sram``).
#
# Performance (this revision): the generic elementwise kernel walks a flat
# element index and recovers ``row = off // hidden`` / ``col = off % hidden``
# with a *runtime* divisor, then issues five masked strided loads per element.
# On XPU3 that scalarises to ~19 ns/element (measured 4.9 ms vs ATen 24 us at
# ``(256, 1024)``).  When every tensor is contiguous (the benchmark path) this
# override instead launches a row-program kernel: one program per batch row,
# ~9 contiguous block-DMA chunks, no integer division, no per-lane gather:
#
#   * gate columns use ``j = arange(3H)`` with ``H`` a ``tl.constexpr`` so the
#     two gate outputs are written as single contiguous ``3H`` stores;
#   * the workspace row ``[5H]`` is read as contiguous gate blocks;
#   * ``grad_hx`` reads only its own ``H`` row (see the kernel comment).
#
# The bias reduction gets a matching row-contiguous kernel: each program owns a
# ``BLOCK_G`` slice of the ``3H`` gate axis and folds every batch row into an
# element-wise vector accumulator (a single pass, no strided per-lane loads and
# no loop-carried ``tl.sum``).
#
# Everything else (validation, ``.out`` handling, empty shapes, non-contiguous
# inputs/outputs) reuses the generic Triton kernels.  No ATen/CPU compute is
# introduced; ``torch.empty`` is allocation only.

import logging

import torch
import triton
import triton.language as tl

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
# The row kernel materialises [3H] tiles; 3*1024 fp32 lanes plus live inputs is
# the largest width that fits the XPU3 scratchpad.  Larger hidden sizes use the
# (correct) generic kernels.
_FAST_MAX_HIDDEN = 1024
# Gate-axis slice for the row-contiguous bias fold.
_BIAS_BLOCK_G = 128


@triton.jit
def _gru_cell_backward_row_kernel(
    grad_hy,
    workspace,
    grad_input_gates,
    grad_hidden_gates,
    grad_hx,
    H: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    # One program per batch row.  j walks the 3*H contiguous gate columns of the
    # outputs; ``col = j % H`` is a compile-time strength-reduced index (H is
    # constexpr) and every memory access below is a contiguous block DMA.
    row = tl.program_id(0)
    j = tl.arange(0, 3 * H)[None, :]
    col = j % H
    ghy = tl.load(grad_hy + row * H + col)
    w = workspace + row * (5 * H)
    reset_gate = tl.load(w + col)
    input_gate = tl.load(w + H + col)
    new_gate = tl.load(w + 2 * H + col)
    hx = tl.load(w + 3 * H + col)
    hidden_new = tl.load(w + 4 * H + col)
    if not IS_FP64:
        ghy = ghy.to(tl.float32)
        reset_gate = reset_gate.to(tl.float32)
        input_gate = input_gate.to(tl.float32)
        new_gate = new_gate.to(tl.float32)
        hx = hx.to(tl.float32)
        hidden_new = hidden_new.to(tl.float32)

    # Preserve the evaluation order used by ATen's CUDA kernel.
    grad_input_gate = ghy * (hx - new_gate)
    grad_input_gate *= 1.0 - input_gate
    grad_input_gate *= input_gate
    grad_new_input = ghy * (1.0 - input_gate)
    grad_new_input *= 1.0 - new_gate * new_gate
    grad_hidden_new = grad_new_input * reset_gate
    grad_reset = grad_new_input * hidden_new
    grad_reset *= 1.0 - reset_gate
    grad_reset *= reset_gate

    # v_in / v_hid are contiguous 3H runs (gate = j // H); the lane value is
    # selected with a compile-time gate index, so each store is one DMA.
    gate = j // H
    v_in = tl.where(
        gate == 0, grad_reset, tl.where(gate == 1, grad_input_gate, grad_new_input)
    )
    v_hid = tl.where(
        gate == 0, grad_reset, tl.where(gate == 1, grad_input_gate, grad_hidden_new)
    )
    ob = row * (3 * H) + j
    in_ty = grad_input_gates.dtype.element_ty
    hid_ty = grad_hidden_gates.dtype.element_ty
    if not IS_FP64:
        v_in = v_in.to(in_ty)
        v_hid = v_hid.to(hid_ty)
    tl.store(grad_input_gates + ob, v_in)
    tl.store(grad_hidden_gates + ob, v_hid)

    # grad_hx only depends on grad_hy and the input gate; recomputing it on its
    # own H-wide row keeps the store contiguous (extracting the middle gate
    # block out of the 3H tile would need a gather).
    c = tl.arange(0, H)[None, :]
    gx = tl.load(grad_hy + row * H + c)
    ig = tl.load(w + H + c)
    if not IS_FP64:
        gx = gx.to(tl.float32)
        ig = ig.to(tl.float32)
    hx_v = gx * ig
    if not IS_FP64:
        hx_v = hx_v.to(grad_hx.dtype.element_ty)
    tl.store(grad_hx + row * H + c, hx_v)


@triton.jit
def _gru_bias_backward_gate_kernel(
    grad_input_gates,
    grad_hidden_gates,
    grad_input_bias,
    grad_hidden_bias,
    batch_size,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    IS_FP64: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Each program owns one BLOCK_G slice of the 3*H gate axis and folds every
    # batch row (contiguous lane block, no per-lane stride, no tl.sum in the
    # loop).  This measured 2x faster than the generic small-batch reduction at
    # every bias shape (e.g. 0.55 ms vs 1.17 ms for (256, 1024) fp32); a wider
    # BLOCK_G loses to the extra gate-axis parallelism.
    gidx = tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)
    if EVEN:
        gmask = gidx >= 0
    else:
        gmask = gidx < G
    if IS_FP64:
        input_acc = tl.zeros((BLOCK_G,), dtype=tl.float64)
        hidden_acc = tl.zeros((BLOCK_G,), dtype=tl.float64)
    else:
        input_acc = tl.zeros((BLOCK_G,), dtype=tl.float32)
        hidden_acc = tl.zeros((BLOCK_G,), dtype=tl.float32)
    for row in range(batch_size):
        base = row * G + gidx
        input_values = tl.load(grad_input_gates + base, mask=gmask, other=0.0)
        hidden_values = tl.load(grad_hidden_gates + base, mask=gmask, other=0.0)
        if IS_FP64:
            input_acc += input_values.to(tl.float64)
            hidden_acc += hidden_values.to(tl.float64)
        else:
            input_acc += input_values.to(tl.float32)
            hidden_acc += hidden_values.to(tl.float32)
    in_ty = grad_input_bias.dtype.element_ty
    hid_ty = grad_hidden_bias.dtype.element_ty
    if not IS_FP64:
        input_acc = input_acc.to(in_ty)
        hidden_acc = hidden_acc.to(hid_ty)
    tl.store(grad_input_bias + gidx, input_acc, mask=gmask)
    tl.store(grad_hidden_bias + gidx, hidden_acc, mask=gmask)


def _contiguous_rows(t, batch_size, hidden_size):
    return (
        t.is_contiguous() and t.stride(0) == hidden_size and t.stride(1) == 1
    )


def _row_fast_ok(grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx):
    batch_size, hidden_size = grad_hy.shape
    # bf16 legalisation fails for the widest [3H] tile (ConvertTritonXPUToLLVM);
    # cap bf16 at 512 while fp16/fp32 keep the 1024 tile.
    max_hidden = 512 if grad_hy.dtype == torch.bfloat16 else _FAST_MAX_HIDDEN
    if hidden_size <= 0 or hidden_size > max_hidden:
        return False
    if hidden_size & (hidden_size - 1) != 0:
        # 3*H must be a power-of-two arange extent.
        return False
    return (
        _contiguous_rows(grad_hy, batch_size, hidden_size)
        and workspace.is_contiguous()
        and workspace.stride(0) == 5 * hidden_size
        and _contiguous_rows(grad_input_gates, batch_size, 3 * hidden_size)
        and _contiguous_rows(grad_hidden_gates, batch_size, 3 * hidden_size)
        and _contiguous_rows(grad_hx, batch_size, hidden_size)
    )


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
    gate_size = 3 * hidden_size
    fast = _row_fast_ok(
        grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx
    )
    with torch_device_fn.device(grad_hy.device):
        if n_elements != 0:
            if fast:
                _gru_cell_backward_row_kernel[(batch_size,)](
                    grad_hy,
                    workspace,
                    grad_input_gates,
                    grad_hidden_gates,
                    grad_hx,
                    H=hidden_size,
                    IS_FP64=grad_hy.dtype == torch.float64,
                )
            else:
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
        if has_bias and gate_size != 0:
            bias_fast = (
                fast
                and _contiguous_rows(grad_input_gates, batch_size, gate_size)
                and _contiguous_rows(grad_hidden_gates, batch_size, gate_size)
                and grad_input_bias.is_contiguous()
                and grad_hidden_bias.is_contiguous()
            )
            if bias_fast and batch_size > 0:
                block_g = min(triton.next_power_of_2(gate_size), _BIAS_BLOCK_G)
                even_g = gate_size % block_g == 0
                _gru_bias_backward_gate_kernel[(triton.cdiv(gate_size, block_g),)](
                    grad_input_gates,
                    grad_hidden_gates,
                    grad_input_bias,
                    grad_hidden_bias,
                    batch_size,
                    G=gate_size,
                    BLOCK_G=block_g,
                    IS_FP64=grad_hy.dtype == torch.float64,
                    EVEN=even_g,
                )
            else:
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
