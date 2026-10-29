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
#
# Kunlunxin (XPU) override of ``aten::lstm`` (inference, multi-layer,
# optionally bidirectional).
#
# The generic implementation (``flag_gems/ops/lstm.py``) folds the input and
# the recurrent gate projections into one per-step kernel that lowers through
# ``tl.dot``; the XPU backend aborts that module before code generation.  This
# overlay splits the work into two single-loop kernels that the XPU pipeline
# accepts:
#
#   * ``_lstm_input_proj_kernel``  -- ``x @ W_ih^T + b_ih + b_hh`` for every
#     timestep at once, into an fp32 scratch buffer;
#   * ``_lstm_recur_kernel``       -- ``h @ W_hh^T`` plus the scratch, the four
#     gate activations and the cell/hidden update, one launch per timestep.
#
# Gate pre-activations are accumulated in fp32 for every input dtype, and the
# flat-weight contract matches ``torch.lstm``'s ``params`` tuple.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.dropout import dropout as _dropout
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["lstm"]

_BLOCK_G_MAX = 64
_BLOCK_H_MAX = 32
_BLOCK_K_MAX = 32


@libentry()
@triton.jit
def _hx_in_kernel(
    src,
    dst,
    HIDDEN_SIZE: tl.constexpr,
    PAD_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN_SIZE
    values = tl.load(src + batch_idx * HIDDEN_SIZE + offs, mask=mask, other=0.0)
    tl.store(dst + batch_idx * PAD_K + offs, values, mask=mask)


@libentry()
@triton.jit
def _hx_out_kernel(
    src,
    dst,
    HIDDEN_SIZE: tl.constexpr,
    PAD_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN_SIZE
    values = tl.load(src + batch_idx * PAD_K + offs, mask=mask, other=0.0)
    tl.store(dst + batch_idx * HIDDEN_SIZE + offs, values, mask=mask)


@libentry()
@triton.jit
def _lstm_input_proj_kernel(
    x_ptr,
    w_ih_ptr,
    b_ih_ptr,
    b_hh_ptr,
    pre_ptr,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    w_ih_stride_r,
    w_ih_stride_c,
    b_ih_stride,
    b_hh_stride,
    INPUT_SIZE: tl.constexpr,
    GATES: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    pid_g = tl.program_id(2)
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    gmask = offs_g < GATES

    acc = tl.zeros((BLOCK_G,), dtype=tl.float32)
    # The bias is folded in before the reduction loop: the XPU backend
    # miscompiles a second load of the same index vector that appears after
    # a loop already indexed by it.
    if HAS_BIAS:
        acc += tl.load(b_ih_ptr + offs_g * b_ih_stride, mask=gmask, other=0.0).to(
            tl.float32
        )
        acc += tl.load(b_hh_ptr + offs_g * b_hh_stride, mask=gmask, other=0.0).to(
            tl.float32
        )
    for k_block in range(0, tl.cdiv(INPUT_SIZE, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        kmask = offs_k < INPUT_SIZE
        x = tl.load(
            x_ptr + seq_idx * x_stride_s + batch_idx * x_stride_b + offs_k * x_stride_f,
            mask=kmask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ih_ptr
            + offs_g[:, None] * w_ih_stride_r
            + offs_k[None, :] * w_ih_stride_c,
            mask=gmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(
        pre_ptr + (seq_idx * BATCH_SIZE + batch_idx) * GATES + offs_g,
        acc,
        mask=gmask,
    )


@libentry()
@triton.jit
def _lstm_recur_kernel(
    pre_ptr,
    h_prev_ptr,
    c_prev_ptr,
    w_hh_ptr,
    h_next_ptr,
    c_next_ptr,
    out_ptr,
    seq_idx,
    out_feature_offset,
    pre_stride_s,
    pre_stride_b,
    w_hh_stride_r,
    w_hh_stride_c,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    HIDDEN_SIZE: tl.constexpr,
    PAD_K: tl.constexpr,
    GATES: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = offs_h < HIDDEN_SIZE
    # offs_h addresses four rows of the (GATES, HIDDEN_SIZE) weight: offs_h,
    # HIDDEN_SIZE + offs_h, 2 * HIDDEN_SIZE + offs_h and 3 * HIDDEN_SIZE +
    # offs_h. Only offs_h < HIDDEN_SIZE keeps all four inside the weight.
    wmask = offs_h < HIDDEN_SIZE

    # The projected input gates seed the accumulators before the reduction
    # loop: the XPU backend miscompiles a load of the same index vector that
    # appears after a loop already indexed by it.
    pre_row = pre_ptr + seq_idx * pre_stride_s + batch_idx * pre_stride_b
    i_acc = tl.load(pre_row + offs_h, mask=hmask, other=0.0)
    f_acc = tl.load(pre_row + HIDDEN_SIZE + offs_h, mask=hmask, other=0.0)
    g_acc = tl.load(pre_row + 2 * HIDDEN_SIZE + offs_h, mask=hmask, other=0.0)
    o_acc = tl.load(pre_row + 3 * HIDDEN_SIZE + offs_h, mask=hmask, other=0.0)

    # The recurrent projection is accumulated one column at a time: each step
    # reads a single h value and the matching W_hh column of the four gates.
    # The XPU backend miscompiles the cross-lane reduction over a two
    # dimensional tile, so the gates are never reduced along an axis.
    for k in range(0, HIDDEN_SIZE):
        h_k = tl.load(h_prev_ptr + batch_idx * PAD_K + k).to(tl.float32)
        i_acc += (
            tl.load(
                w_hh_ptr + offs_h * w_hh_stride_r + k * w_hh_stride_c,
                mask=wmask,
                other=0.0,
            ).to(tl.float32)
            * h_k
        )
        f_acc += (
            tl.load(
                w_hh_ptr + (HIDDEN_SIZE + offs_h) * w_hh_stride_r + k * w_hh_stride_c,
                mask=wmask,
                other=0.0,
            ).to(tl.float32)
            * h_k
        )
        g_acc += (
            tl.load(
                w_hh_ptr
                + (2 * HIDDEN_SIZE + offs_h) * w_hh_stride_r
                + k * w_hh_stride_c,
                mask=wmask,
                other=0.0,
            ).to(tl.float32)
            * h_k
        )
        o_acc += (
            tl.load(
                w_hh_ptr
                + (3 * HIDDEN_SIZE + offs_h) * w_hh_stride_r
                + k * w_hh_stride_c,
                mask=wmask,
                other=0.0,
            ).to(tl.float32)
            * h_k
        )

    # ``tl.exp2`` is not usable here: on XPU it lowers to the natural
    # exponential, so the gate activations are written with ``tl.exp``.
    i_gate = 1.0 / (1.0 + tl.exp(-i_acc))
    f_gate = 1.0 / (1.0 + tl.exp(-f_acc))
    g_gate = 2.0 / (1.0 + tl.exp(-2.0 * g_acc)) - 1.0
    o_gate = 1.0 / (1.0 + tl.exp(-o_acc))

    c_prev = tl.load(
        c_prev_ptr + batch_idx * PAD_K + offs_h,
        mask=hmask,
        other=0.0,
    ).to(tl.float32)
    c_next = f_gate * c_prev + i_gate * g_gate
    h_next = o_gate * (2.0 / (1.0 + tl.exp(-2.0 * c_next)) - 1.0)

    state_offsets = batch_idx * PAD_K + offs_h
    out_offsets = (
        seq_idx * out_stride_s
        + batch_idx * out_stride_b
        + (out_feature_offset + offs_h) * out_stride_f
    )
    tl.store(h_next_ptr + state_offsets, h_next, mask=hmask)
    tl.store(c_next_ptr + state_offsets, c_next, mask=hmask)
    tl.store(out_ptr + out_offsets, h_next, mask=hmask)


def _ceil_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _block_size(value: int, maximum: int) -> int:
    return min(max(_ceil_power_of_2(value), 16), maximum)


def _matrix_strides(tensor: torch.Tensor, rows: int, cols: int):
    if tensor.numel() != rows * cols:
        raise RuntimeError(
            f"invalid LSTM weight with {tensor.numel()} elements, expected {rows * cols}"
        )
    if tensor.dim() == 2:
        if tensor.shape != (rows, cols):
            raise RuntimeError(
                f"invalid LSTM weight shape {tuple(tensor.shape)}, expected {(rows, cols)}"
            )
        return tensor.stride(0), tensor.stride(1)
    if tensor.dim() == 1:
        return cols, 1
    raise RuntimeError(f"LSTM weights must be 1-D or 2-D, got {tensor.dim()}-D")


def _bias_stride(tensor: torch.Tensor, size: int) -> int:
    if tensor.numel() != size:
        raise RuntimeError(
            f"invalid LSTM bias with {tensor.numel()} elements, expected {size}"
        )
    if tensor.dim() == 1:
        return tensor.stride(0)
    raise RuntimeError("LSTM bias tensors must be 1-D, got {}-D".format(tensor.dim()))


def _param_group(params, index: int, has_biases: bool):
    group_size = 4 if has_biases else 2
    base = index * group_size
    if has_biases:
        return params[base], params[base + 1], params[base + 2], params[base + 3]
    return params[base], params[base + 1], params[base], params[base]


def _hx_in(src, dst, batch_size: int, hidden_size: int, pad_k: int):
    if batch_size == 0:
        return
    block = _ceil_power_of_2(hidden_size)
    with torch_device_fn.device(dst.device):
        _hx_in_kernel[(batch_size,)](
            src, dst, HIDDEN_SIZE=hidden_size, PAD_K=pad_k, BLOCK=block
        )


def _hx_out(src, dst, batch_size: int, hidden_size: int, pad_k: int):
    if batch_size == 0:
        return
    block = _ceil_power_of_2(hidden_size)
    with torch_device_fn.device(dst.device):
        _hx_out_kernel[(batch_size,)](
            src, dst, HIDDEN_SIZE=hidden_size, PAD_K=pad_k, BLOCK=block
        )


def _run_direction(
    layer_input,
    hx,
    cx,
    layer_output,
    final_h,
    final_c,
    params,
    state_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
):
    w_ih, w_hh, b_ih, b_hh = _param_group(params, state_idx, has_biases)
    w_ih_stride_r, w_ih_stride_c = _matrix_strides(w_ih, 4 * hidden_size, input_size)
    w_hh_stride_r, w_hh_stride_c = _matrix_strides(w_hh, 4 * hidden_size, hidden_size)
    b_ih_stride = _bias_stride(b_ih, 4 * hidden_size) if has_biases else 1
    b_hh_stride = _bias_stride(b_hh, 4 * hidden_size) if has_biases else 1

    pad_k = max(_ceil_power_of_2(hidden_size), 16)
    h_a = torch.zeros(
        (batch_size, pad_k), device=layer_input.device, dtype=layer_input.dtype
    )
    c_a = torch.zeros_like(h_a)
    h_b = torch.zeros_like(h_a)
    c_b = torch.zeros_like(c_a)
    if batch_size == 0:
        return
    _hx_in(hx[state_idx], h_a, batch_size, hidden_size, pad_k)
    _hx_in(cx[state_idx], c_a, batch_size, hidden_size, pad_k)

    gates = 4 * hidden_size
    pre = torch.empty(
        (seq_len, batch_size, gates), device=layer_input.device, dtype=torch.float32
    )
    block_g = _block_size(gates, _BLOCK_G_MAX)
    block_k = _block_size(input_size, _BLOCK_K_MAX)
    proj_grid = (
        seq_len,
        batch_size,
        triton.cdiv(gates, block_g),
    )
    with torch_device_fn.device(layer_input.device):
        _lstm_input_proj_kernel[proj_grid](
            layer_input,
            w_ih,
            b_ih,
            b_hh,
            pre,
            layer_input.stride(0),
            layer_input.stride(1),
            layer_input.stride(2),
            w_ih_stride_r,
            w_ih_stride_c,
            b_ih_stride,
            b_hh_stride,
            INPUT_SIZE=input_size,
            GATES=gates,
            BATCH_SIZE=batch_size,
            HAS_BIAS=has_biases,
            BLOCK_G=block_g,
            BLOCK_K=block_k,
        )

    block_h = _block_size(hidden_size, _BLOCK_H_MAX)
    recur_grid = (
        batch_size,
        triton.cdiv(hidden_size, block_h),
    )
    h_cur, h_other = h_a, h_b
    c_cur, c_other = c_a, c_b
    with torch_device_fn.device(layer_input.device):
        for step in range(seq_len):
            seq_idx = seq_len - 1 - step if reverse else step
            _lstm_recur_kernel[recur_grid](
                pre,
                h_cur,
                c_cur,
                w_hh,
                h_other,
                c_other,
                layer_output,
                seq_idx,
                out_feature_offset,
                pre.stride(0),
                pre.stride(1),
                w_hh_stride_r,
                w_hh_stride_c,
                layer_output.stride(0),
                layer_output.stride(1),
                layer_output.stride(2),
                HIDDEN_SIZE=hidden_size,
                PAD_K=pad_k,
                GATES=gates,
                BLOCK_H=block_h,
            )
            h_cur, h_other = h_other, h_cur
            c_cur, c_other = c_other, c_cur

    _hx_out(h_cur, final_h[state_idx], batch_size, hidden_size, pad_k)
    _hx_out(c_cur, final_c[state_idx], batch_size, hidden_size, pad_k)


def _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional):
    if input.dim() != 3:
        raise RuntimeError("lstm: input must have 3 dimensions")
    if not isinstance(hx, (tuple, list)) or len(hx) != 2:
        raise RuntimeError("lstm expects two hidden states")
    if num_layers <= 0:
        raise RuntimeError("lstm: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("lstm: dropout probability must be between 0 and 1")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "FlagGems lstm supports float16, bfloat16, and float32"
        )

    hx0, cx0 = hx
    if hx0.shape != cx0.shape:
        raise RuntimeError("lstm: hidden and cell states must have the same shape")
    if hx0.dim() != 3:
        raise RuntimeError("lstm: hidden and cell states must have 3 dimensions")
    num_directions = 2 if bidirectional else 1
    expected_states = num_layers * num_directions
    if hx0.shape[0] != expected_states:
        raise RuntimeError(
            f"lstm: expected {expected_states} hidden state rows, got {hx0.shape[0]}"
        )
    expected_params = expected_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"lstm: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx0.device != input.device or cx0.device != input.device:
        raise RuntimeError(
            "lstm: input, hidden state, and cell state must share a device"
        )
    if hx0.dtype != input.dtype or cx0.dtype != input.dtype:
        raise RuntimeError(
            "lstm: input, hidden state, and cell state must share a dtype"
        )


def lstm(
    input,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    logger.debug("GEMS_KUNLUNXIN LSTM")
    _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional)

    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1)
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    if seq_len == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")

    hx0, cx0 = hx
    hx0 = hx0.contiguous()
    cx0 = cx0.contiguous()
    hidden_size = hx0.shape[2]
    num_directions = 2 if bidirectional else 1
    final_h = torch.empty(
        (num_layers * num_directions, batch_size, hidden_size),
        device=input.device,
        dtype=input.dtype,
    )
    final_c = torch.empty_like(final_h)

    layer_input = input_view
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        layer_output = torch.empty(
            (seq_len, batch_size, hidden_size * num_directions),
            device=input.device,
            dtype=input.dtype,
        )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            _run_direction(
                layer_input,
                hx0,
                cx0,
                layer_output,
                final_h,
                final_c,
                params,
                state_idx,
                direction * hidden_size,
                layer_input_size,
                hidden_size,
                batch_size,
                seq_len,
                has_biases,
                direction == 1,
            )

        layer_input = layer_output
        if train and dropout != 0.0 and layer + 1 < num_layers:
            layer_input, _ = _dropout(layer_input, dropout, True)

    output = layer_input.transpose(0, 1) if batch_first else layer_input
    return output, final_h, final_c
