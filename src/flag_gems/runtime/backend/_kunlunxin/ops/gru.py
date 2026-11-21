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

"""Kunlunxin/XPU implementation of GRU.

The generic implementation switches to persistent/GEMV recurrences whenever the
grid fits the device.  Those kernels rely on cross-program synchronization that
the current lowering does not support.  This implementation keeps the same host
pipeline (packed <-> padded conversion, input GEMM, multi-layer/bidirectional
handling) but always launches the per-step kernel.

The kernel bodies also avoid the "early return after a store" shape: inactive
packed rows are frozen with ``tl.where`` instead of returning from a block that
already has side effects.  This keeps the control flow simple for the compiler.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.dropout import dropout as _dropout
from flag_gems.ops.gru import (
    _BLOCK_B,
    _PACK_BLOCK,
    _STEP_BLOCK_H,
    _STEP_BLOCK_K,
    _STEP_NUM_STAGES,
    _STEP_NUM_WARPS,
    _batch_offsets_kernel,
    _bias_stride,
    _block_size,
    _ceil_power_of_2,
    _copy_hx_slice,
    _empty,
    _pack_output_kernel,
    _param_group,
    _store_hx_slice,
    _transpose_weight,
    _unpack_padded_kernel,
    _validate_args,
    _validate_weight,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("gru"),
    key=["input_size", "hidden_size", "batch_size"],
)
@triton.jit
def _gru_input_gemm_kernel_v(
    x_ptr,
    w_ih_ptr,
    b_ih_ptr,
    u_ptr,
    input_size,
    hidden_size,
    batch_size,
    x_stride_s,
    x_stride_b,
    x_stride_f,
    w_ih_stride_r,
    w_ih_stride_c,
    b_ih_stride,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Input-side pre-activation for all timesteps.  There is no early return and
    # no packed-specific branch here; the recurrence freezes inactive rows.
    pid_b = tl.program_id(0)
    seq_idx = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b < batch_size
    n_mask = offs_n < 3 * hidden_size

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for k_block in range(0, tl.cdiv(input_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr
            + seq_idx * x_stride_s
            + offs_b[:, None] * x_stride_b
            + offs_k[None, :] * x_stride_f,
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < input_size),
            other=0.0,
        )
        w = tl.load(
            w_ih_ptr
            + offs_k[:, None] * w_ih_stride_r
            + offs_n[None, :] * w_ih_stride_c,
            mask=(offs_k[:, None] < input_size) & (offs_n[None, :] < 3 * hidden_size),
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b = tl.load(b_ih_ptr + offs_n * b_ih_stride, mask=n_mask, other=0.0)
        acc += b[None, :]

    # Rows past batch_sizes[seq_idx] are padding; the recurrence freezes their
    # hidden state, so their pre-activation value is never consumed.

    out_offsets = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_n[None, :] * u_stride_f
    )
    tl.store(u_ptr + out_offsets, acc, mask=b_mask[:, None] & n_mask[None, :])


@libentry()
@triton.jit
def _gru_hidden_gemm_kernel_v(
    h_prev_ptr,
    w_hh_ptr,
    b_hh_ptr,
    pre_ptr,
    hidden_size,
    batch_size,
    w_hh_stride_r,
    w_hh_stride_c,
    b_hh_stride,
    pre_stride_b,
    pre_stride_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Hidden-side GEMM only: gh[b, n] = h[b,:] @ W_hh[n,:] + b_hh[n].

    This kernel contains a tl.dot but no sigmoid/tanh.  Keeping the activations
    out of this kernel lets them compile through the accurate elementwise path
    instead of the approximate activation lowering used with tl.dot.
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_mask = offs_b[:, None] < batch_size
    n_mask = offs_n[None, :] < 3 * hidden_size
    out_mask = b_mask & n_mask

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=COMPUTE_DTYPE)
    for k_block in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        h = tl.load(
            h_prev_ptr + offs_b[:, None] * hidden_size + offs_k[None, :],
            mask=(offs_b[:, None] < batch_size) & (offs_k[None, :] < hidden_size),
            other=0.0,
        )
        w = tl.load(
            w_hh_ptr
            + offs_k[:, None] * w_hh_stride_r
            + offs_n[None, :] * w_hh_stride_c,
            mask=(offs_k[:, None] < hidden_size) & (offs_n[None, :] < 3 * hidden_size),
            other=0.0,
        )
        acc += tl.dot(h, w, out_dtype=COMPUTE_DTYPE, allow_tf32=False)

    if HAS_BIAS:
        b = tl.load(
            b_hh_ptr + offs_n * b_hh_stride, mask=offs_n < 3 * hidden_size, other=0.0
        )
        acc += b[None, :]

    tl.store(
        pre_ptr + offs_b[:, None] * pre_stride_b + offs_n[None, :] * pre_stride_n,
        acc,
        mask=out_mask,
    )


@libentry()
@triton.jit
def _gru_gates_update_kernel_v(
    u_ptr,
    pre_ptr,
    h_prev_ptr,
    h_next_ptr,
    out_ptr,
    batch_sizes_ptr,
    seq_idx,
    out_feature_offset,
    hidden_size,
    batch_size,
    u_stride_s,
    u_stride_b,
    u_stride_f,
    pre_stride_b,
    pre_stride_n,
    out_stride_s,
    out_stride_b,
    out_stride_f,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PACKED: tl.constexpr,
):
    """Gate activations and the h update.

    There is deliberately no tl.dot in this kernel, so sigmoid and tanh compile
    through the accurate elementwise path instead of the approximate activation
    lowering used with tl.dot.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bh_mask = (offs_b[:, None] < batch_size) & (offs_h[None, :] < hidden_size)

    pre_base = offs_b[:, None] * pre_stride_b + offs_h[None, :] * pre_stride_n
    gh_r = tl.load(pre_ptr + pre_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    gh_z = tl.load(
        pre_ptr + pre_base + hidden_size * pre_stride_n, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    gh_n = tl.load(
        pre_ptr + pre_base + 2 * hidden_size * pre_stride_n, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)

    u_base = (
        seq_idx * u_stride_s
        + offs_b[:, None] * u_stride_b
        + offs_h[None, :] * u_stride_f
    )
    u_r = tl.load(u_ptr + u_base, mask=bh_mask, other=0.0).to(COMPUTE_DTYPE)
    u_z = tl.load(
        u_ptr + u_base + hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)
    u_n = tl.load(
        u_ptr + u_base + 2 * hidden_size * u_stride_f, mask=bh_mask, other=0.0
    ).to(COMPUTE_DTYPE)

    r_gate = tl.sigmoid(u_r + gh_r)
    z_gate = tl.sigmoid(u_z + gh_z)
    n_gate = tl_extra_shim.tanh(u_n + r_gate * gh_n)

    h_prev = tl.load(
        h_prev_ptr + offs_b[:, None] * hidden_size + offs_h[None, :],
        mask=bh_mask,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev

    if PACKED:
        # Rows whose sequence has ended must not advance.
        active = offs_b < tl.load(batch_sizes_ptr + seq_idx).to(tl.int32)
        h_next = tl.where(active[:, None], h_next, h_prev)

    state_offsets = offs_b[:, None] * hidden_size + offs_h[None, :]
    out_offsets = (
        seq_idx * out_stride_s
        + offs_b[:, None] * out_stride_b
        + (out_feature_offset + offs_h[None, :]) * out_stride_f
    )
    tl.store(h_next_ptr + state_offsets, h_next, mask=bh_mask)
    tl.store(out_ptr + out_offsets, h_next, mask=bh_mask)


def _run_direction_v(
    layer_input,
    hx,
    layer_output,
    final_h,
    params,
    state_idx: int,
    param_idx: int,
    out_feature_offset: int,
    input_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    has_biases: bool,
    reverse: bool,
    batch_sizes=None,
):
    w_ih, w_hh, b_ih, b_hh = _param_group(params, param_idx, has_biases)
    _validate_weight(w_ih, 3 * hidden_size, input_size)
    _validate_weight(w_hh, 3 * hidden_size, hidden_size)
    if w_ih.dim() == 1:
        w_ih = w_ih.view(3 * hidden_size, input_size)
    if w_hh.dim() == 1:
        w_hh = w_hh.view(3 * hidden_size, hidden_size)
    w_ih = _transpose_weight(w_ih, 3 * hidden_size, input_size)
    w_hh = _transpose_weight(w_hh, 3 * hidden_size, hidden_size)
    w_ih_stride_r, w_ih_stride_c = w_ih.stride(0), w_ih.stride(1)
    w_hh_stride_r, w_hh_stride_c = w_hh.stride(0), w_hh.stride(1)
    b_ih_stride = _bias_stride(b_ih, 3 * hidden_size) if has_biases else 1
    b_hh_stride = _bias_stride(b_hh, 3 * hidden_size) if has_biases else 1

    if batch_size == 0:
        return

    block_h_step = _block_size(hidden_size, _STEP_BLOCK_H)
    block_k_step = _block_size(hidden_size, _STEP_BLOCK_K)
    grid = (
        triton.cdiv(batch_size, _BLOCK_B),
        triton.cdiv(hidden_size, block_h_step),
    )

    if layer_input.dtype == torch.float64:
        compute_dtype = tl.float64
        gate_dtype = torch.float64
    else:
        compute_dtype = tl.float32
        gate_dtype = torch.float32

    input_gates = _empty(
        (seq_len, batch_size, 3 * hidden_size), gate_dtype, layer_input.device
    )

    def input_gemm_grid(META):
        return (
            triton.cdiv(batch_size, META["BLOCK_B"]),
            seq_len,
            triton.cdiv(3 * hidden_size, META["BLOCK_N"]),
        )

    with torch_device_fn.device(layer_input.device):
        _gru_input_gemm_kernel_v[input_gemm_grid](
            layer_input,
            w_ih,
            b_ih,
            input_gates,
            input_size,
            hidden_size,
            batch_size,
            layer_input.stride(0),
            layer_input.stride(1),
            layer_input.stride(2),
            w_ih_stride_r,
            w_ih_stride_c,
            b_ih_stride,
            input_gates.stride(0),
            input_gates.stride(1),
            input_gates.stride(2),
            HAS_BIAS=has_biases,
            COMPUTE_DTYPE=compute_dtype,
        )

        # Always use the per-step recurrence on this backend.  Each step is
        # split into:
        #   1) a hidden GEMM with tl.dot and no activations
        #   2) a gate update without tl.dot, using libdevice sigmoid/tanh
        block_n_g = _block_size(3 * hidden_size, 64)
        hidden_grid = (
            triton.cdiv(batch_size, _BLOCK_B),
            triton.cdiv(3 * hidden_size, block_n_g),
        )
        pre = _empty((batch_size, 3 * hidden_size), gate_dtype, layer_input.device)
        h_work = _empty((batch_size, hidden_size), hx.dtype, hx.device)
        _copy_hx_slice(hx, h_work, state_idx, batch_size, hidden_size)
        h_next = _empty((batch_size, hidden_size), hx.dtype, hx.device)
        for step in range(seq_len):
            seq_idx = seq_len - 1 - step if reverse else step
            _gru_hidden_gemm_kernel_v[hidden_grid](
                h_work,
                w_hh,
                b_hh,
                pre,
                hidden_size,
                batch_size,
                w_hh_stride_r,
                w_hh_stride_c,
                b_hh_stride,
                pre.stride(0),
                pre.stride(1),
                HAS_BIAS=has_biases,
                BLOCK_B=_BLOCK_B,
                BLOCK_N=block_n_g,
                BLOCK_K=block_k_step,
                COMPUTE_DTYPE=compute_dtype,
                num_warps=_STEP_NUM_WARPS,
                num_stages=_STEP_NUM_STAGES,
            )
            _gru_gates_update_kernel_v[grid](
                input_gates,
                pre,
                h_work,
                h_next,
                layer_output,
                batch_sizes if batch_sizes is not None else pre,
                seq_idx,
                out_feature_offset,
                hidden_size,
                batch_size,
                input_gates.stride(0),
                input_gates.stride(1),
                input_gates.stride(2),
                pre.stride(0),
                pre.stride(1),
                layer_output.stride(0),
                layer_output.stride(1),
                layer_output.stride(2),
                BLOCK_B=_BLOCK_B,
                BLOCK_H=block_h_step,
                COMPUTE_DTYPE=compute_dtype,
                PACKED=batch_sizes is not None,
            )
            h_work, h_next = h_next, h_work
        final_h_state = h_work

    _store_hx_slice(final_h_state, final_h, state_idx, batch_size, hidden_size)


def _gru_forward_impl_v(
    input_view,
    hx,
    params,
    output,
    final_h,
    num_layers,
    num_directions,
    hidden_size,
    input_size,
    batch_size,
    seq_len,
    has_biases,
    train,
    dropout,
    batch_sizes=None,
):
    layer_input = input_view
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        if layer == num_layers - 1:
            layer_output = output
        else:
            layer_output = _empty(
                (seq_len, batch_size, hidden_size * num_directions),
                input_view.dtype,
                input_view.device,
            )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            reverse = direction == 1
            _run_direction_v(
                layer_input,
                hx,
                layer_output,
                final_h,
                params,
                state_idx,
                state_idx,
                direction * hidden_size,
                layer_input_size,
                hidden_size,
                batch_size,
                seq_len,
                has_biases,
                reverse,
                batch_sizes,
            )

        layer_input = layer_output
        if train and dropout != 0.0 and layer + 1 < num_layers:
            layer_input, _ = _dropout(layer_input, dropout, True)

    return layer_input


def gru(
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
    logger.debug("GEMS GRU (kunlunxin vendor)")
    _validate_args(input, hx, params, has_biases, num_layers, dropout, bidirectional)

    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1)
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    if seq_len == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")

    hidden_size = hx.shape[2]
    num_directions = 2 if bidirectional else 1

    final_h = _empty(
        (num_layers * num_directions, batch_size, hidden_size),
        input.dtype,
        input.device,
    )
    output_tf = _empty(
        (seq_len, batch_size, hidden_size * num_directions),
        input.dtype,
        input.device,
    )
    _gru_forward_impl_v(
        input_view,
        hx,
        params,
        output_tf,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch_size,
        seq_len,
        has_biases,
        train,
        dropout,
    )

    output = output_tf.transpose(0, 1) if batch_first else output_tf
    return output, final_h


def gru_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
):
    logger.debug("GEMS GRU_DATA (kunlunxin vendor)")
    if data.dim() != 2:
        raise RuntimeError("gru.data: packed data must have 2 dimensions")
    if batch_sizes.dim() != 1:
        raise RuntimeError("gru.data: batch_sizes must be 1-dimensional")
    if num_layers <= 0:
        raise RuntimeError("gru.data: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru.data: dropout probability must be between 0 and 1")
    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems gru.data supports float16, bfloat16, float32, and float64"
        )
    num_directions = 2 if bidirectional else 1
    num_states = num_layers * num_directions
    if hx.dim() != 3:
        raise RuntimeError("gru.data: hidden state must have 3 dimensions")
    if hx.shape[0] != num_states:
        raise RuntimeError(
            f"gru.data: expected {num_states} hidden state rows, got {hx.shape[0]}"
        )
    expected_params = num_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError(
            f"gru.data: expected {expected_params} parameter tensors, got {len(params)}"
        )
    if hx.device != data.device:
        raise RuntimeError("gru.data: data and hidden state must share a device")
    if hx.dtype != data.dtype:
        raise RuntimeError("gru.data: data and hidden state must share a dtype")

    num_steps = batch_sizes.numel()
    input_size = data.shape[1]
    batch = hx.shape[1]
    hidden_size = hx.shape[2]

    batch_sizes = batch_sizes.to(data.device)

    offsets = _empty((num_steps,), torch.int32, data.device)
    bs32 = _empty((num_steps,), torch.int32, data.device)
    with torch_device_fn.device(data.device):
        _batch_offsets_kernel[(1,)](
            batch_sizes,
            offsets,
            bs32,
            num_steps,
            BLOCK=_ceil_power_of_2(num_steps),
        )

    x_padded = _empty((num_steps, batch, input_size), data.dtype, data.device)
    with torch_device_fn.device(data.device):
        _unpack_padded_kernel[(num_steps * batch,)](
            data,
            x_padded,
            offsets,
            bs32,
            input_size,
            batch,
            data.stride(0),
            x_padded.stride(0),
            x_padded.stride(1),
            x_padded.stride(2),
            BLOCK_F=_PACK_BLOCK,
        )

    hidden_total = hidden_size * num_directions
    final_h = _empty((num_states, batch, hidden_size), data.dtype, data.device)
    out_padded = _empty((num_steps, batch, hidden_total), data.dtype, data.device)
    _gru_forward_impl_v(
        x_padded,
        hx,
        params,
        out_padded,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch,
        num_steps,
        has_biases,
        train,
        dropout,
        batch_sizes=bs32,
    )

    out_packed = _empty((data.shape[0], hidden_total), data.dtype, data.device)
    with torch_device_fn.device(data.device):
        _pack_output_kernel[(num_steps * batch,)](
            out_padded,
            out_packed,
            offsets,
            bs32,
            hidden_total,
            batch,
            out_padded.stride(0),
            out_padded.stride(1),
            out_padded.stride(2),
            out_packed.stride(0),
            BLOCK_F=_PACK_BLOCK,
        )

    return out_packed, final_h
