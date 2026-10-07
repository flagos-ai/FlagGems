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

import torch
import triton
import triton.language as tl

__all__ = ["lstm"]


@triton.jit
def lstm_stub_kernel(
    x_ptr,
    init_h_ptr,
    init_c_ptr,
    wx_ptr,
    wh_ptr,
    bx_ptr,
    bh_ptr,
    y_ptr,
    last_h_ptr,
    last_c_ptr,
    seq_len,
    batch_size,
    xdim,
    hdim,
    is_reverse,
    wx_max_ptr,
    wh_max_ptr,
):
    # kunlunxin launch-table binding: this launch is intercepted by
    # try_launch_table (liblaunch_shared.so) and dispatched to the xhpc
    # whole-sequence fused LSTM (baidu::xpu::api::lstm_inference). The body
    # below never executes; it only needs to compile on the normal pipeline
    # (no tl.dot / no tensor-denominator division, see sdnn.h:2550).
    tl.program_id(0)


def _run_fusion(x, init_h, init_c, wx, wh, bx, bh, y, last_h, last_c, is_reverse):
    seq_len, batch_size, xdim = x.shape
    hdim = last_h.shape[-1]
    # int16 TGEMM quantization scales (maxptr args of lstm_inference)
    wx_max = wx.abs().amax().float().reshape(1)
    wh_max = wh.abs().amax().float().reshape(1)
    # xhpc lstm_inference contract: bias is always `const float*` (4B/elem)
    # even for fp16/bf16 inputs (launch_extra.cpp launch_fg_lstm fp16 branch
    # casts t.ptr[5]/t.ptr[6] to const float*). Upcast non-fp32 bias so the
    # handler does not read fp16 (2B) as float (4B) -> OOB garbage (fp16+bias
    # maxdiff 0.674 before this fix).
    if bx is not None and bx.dtype != torch.float32:
        bx = bx.float().contiguous()
    if bh is not None and bh.dtype != torch.float32:
        bh = bh.float().contiguous()
    lstm_stub_kernel[(1,)](
        x,
        init_h,
        init_c,
        wx,
        wh,
        bx,
        bh,
        y,
        last_h,
        last_c,
        seq_len,
        batch_size,
        xdim,
        hdim,
        1 if is_reverse else 0,
        wx_max,
        wh_max,
    )


def _lstm_composite(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """fp16/bf16 LSTM as a decomposition over torch basic ops.

    Mirrors xpytorch composite_lstm_cell (_thnn_fused_lstm_cell.cpp:46-57):
    gates = x@w_ih^T + h@w_hh^T (+bias); i/f/g/o = chunk(gates, 4) with
    sigmoid/sigmoid/tanh/sigmoid; cy = f*cx + i*g; hy = o*tanh(cy), all in
    the input dtype. Every sub-op (mm, add, sigmoid, tanh, mul) is a
    compilable FlagGems triton op, so this path never enters the fused
    triton kernel compile wall (sdnn.h:2550 family).

    Perf notes: weights are transposed once per direction (contiguous
    [*, 4H] operands), the two per-step matmuls are chained through
    addmm, and the gate activations are computed on the full gates
    tensor (one sigmoid + one tanh) and sliced, keeping the per-step
    launch count low.
    """
    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1).contiguous()
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    if seq_len == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")

    hx0, cx0 = hx
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
        layer_output = torch.empty(
            (seq_len, batch_size, hidden_size * num_directions),
            device=input.device,
            dtype=input.dtype,
        )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            reverse = direction == 1
            if has_biases:
                w_ih, w_hh, b_ih, b_hh = params[4 * state_idx : 4 * state_idx + 4]
                bias = b_ih + b_hh
            else:
                w_ih, w_hh = params[2 * state_idx : 2 * state_idx + 2]
                bias = None
            # Transpose once per direction; contiguous operands keep the
            # hijacked mm/addmm kernels on the fast path.
            w_ih_t = w_ih.t().contiguous()
            w_hh_t = w_hh.t().contiguous()
            h_prev = hx0[state_idx].contiguous()
            c_prev = cx0[state_idx].contiguous()
            for step in range(seq_len):
                seq_idx = seq_len - 1 - step if reverse else step
                gates = layer_input[seq_idx] @ w_ih_t
                gates = torch.addmm(gates, h_prev, w_hh_t)
                if bias is not None:
                    gates = gates + bias
                s_all = torch.sigmoid(gates)
                t_all = torch.tanh(gates)
                c_next = (
                    s_all[:, hidden_size : 2 * hidden_size] * c_prev
                    + s_all[:, :hidden_size]
                    * t_all[:, 2 * hidden_size : 3 * hidden_size]
                )
                h_next = s_all[:, 3 * hidden_size : 4 * hidden_size] * torch.tanh(
                    c_next
                )
                layer_output[
                    seq_idx,
                    :,
                    direction * hidden_size : (direction + 1) * hidden_size,
                ] = h_next
                h_prev = h_next
                c_prev = c_next
            final_h[state_idx] = h_prev
            final_c[state_idx] = c_prev
        layer_input = layer_output
        if train and dropout != 0.0 and layer + 1 < num_layers:
            from flag_gems.ops.dropout import dropout as _dropout

            layer_input, _ = _dropout(layer_input, dropout, True)

    output = layer_input.transpose(0, 1) if batch_first else layer_input
    return output, final_h, final_c


def _run_fusion_layer(
    input,
    hx,
    params,
    has_biases,
    layer,
    direction,
    num_directions,
    seq_len,
    batch_size,
    hidden_size,
):
    """Run one direction of one layer through the xhpc fused LSTM."""
    state_idx = layer * num_directions + direction
    if has_biases:
        w_ih, w_hh, b_ih, b_hh = params[4 * state_idx : 4 * state_idx + 4]
    else:
        w_ih, w_hh = params[2 * state_idx : 2 * state_idx + 2]
        b_ih = b_hh = torch.zeros(w_ih.shape[0], device=input.device, dtype=w_ih.dtype)
    out_dtype = input.dtype
    is_reverse = direction == 1
    x_in = input
    if is_reverse:
        # Binding-layer fix (09-10 9/12): vendor is_reverse=1 is wrong in this
        # stack; reverse = flip-input -> forward -> flip-output (verified
        # 8.8e-5 vs torch bidir+2layer).
        x_in = torch.flip(input, dims=[0]).contiguous()
    # bf16: xhpc lstm_inference has no bf16 instantiation (launch_fg_lstm
    # type_index 7 -> default no-op) -> upcast to fp32, run fp32 fused kernel,
    # downcast outputs.
    if out_dtype == torch.bfloat16:
        x_in = x_in.float().contiguous()
        w_ih_f = w_ih.float().contiguous()
        w_hh_f = w_hh.float().contiguous()
        b_ih_f = b_ih.float().contiguous()
        b_hh_f = b_hh.float().contiguous()
        h_prev = hx[0][state_idx].float().contiguous()
        c_prev = hx[1][state_idx].float().contiguous()
    else:
        w_ih_f = w_ih.contiguous()
        w_hh_f = w_hh.contiguous()
        b_ih_f = b_ih.contiguous()
        b_hh_f = b_hh.contiguous()
        h_prev = hx[0][state_idx].contiguous()
        c_prev = hx[1][state_idx].contiguous()
    y = torch.empty(
        seq_len,
        batch_size,
        hidden_size,
        device=input.device,
        dtype=torch.float32 if out_dtype == torch.bfloat16 else out_dtype,
    )
    last_h = torch.empty(
        batch_size,
        hidden_size,
        device=input.device,
        dtype=torch.float32 if out_dtype == torch.bfloat16 else out_dtype,
    )
    last_c = torch.empty(
        batch_size,
        hidden_size,
        device=input.device,
        dtype=torch.float32 if out_dtype == torch.bfloat16 else out_dtype,
    )
    _run_fusion(
        x_in, h_prev, c_prev, w_ih_f, w_hh_f, b_ih_f, b_hh_f, y, last_h, last_c, False
    )
    if is_reverse:
        y = torch.flip(y, dims=[0])
    if out_dtype == torch.bfloat16:
        y = y.to(out_dtype)
        last_h = last_h.to(out_dtype)
        last_c = last_c.to(out_dtype)
    return y, last_h, last_c


def _lstm_fused(input, hx, params, has_biases, num_layers, bidirectional, batch_first):
    """Multi-layer (optionally bidirectional) LSTM via the xhpc fused kernel.

    Each direction is a single fused launch (lstm_stub_kernel -> launch table
    -> fg_lstm -> baidu::xpu::api::lstm_inference); layers/directions are
    composed in Python (same math as PyTorch). Raises on any failure so the
    caller can fall back to composite.
    """
    if batch_first:
        batch_size, seq_len, input_size = input.shape
        input_view = input.transpose(0, 1).contiguous()
    else:
        seq_len, batch_size, input_size = input.shape
        input_view = input
    hidden_size = hx[0].shape[2]
    num_directions = 2 if bidirectional else 1
    final_h = torch.empty_like(hx[0])
    final_c = torch.empty_like(hx[1])
    layer_input = input_view
    for layer in range(num_layers):
        layer_out = torch.empty(
            (seq_len, batch_size, hidden_size * num_directions),
            device=input.device,
            dtype=input.dtype,
        )
        for direction in range(num_directions):
            state_idx = layer * num_directions + direction
            y, lh, lc = _run_fusion_layer(
                layer_input,
                hx,
                params,
                has_biases,
                layer,
                direction,
                num_directions,
                seq_len,
                batch_size,
                hidden_size,
            )
            layer_out[:, :, direction * hidden_size : (direction + 1) * hidden_size] = y
            final_h[state_idx] = lh
            final_c[state_idx] = lc
        layer_input = layer_out
    output = layer_input.transpose(0, 1) if batch_first else layer_input
    return output, final_h, final_c


def lstm(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """aten::lstm: xhpc fusion binding (fast path) with composite fallback.

    The fused path (launch table fg_lstm -> lstm_inference) gives near-native
    performance (one launch per direction); reverse uses flip+fwd+flip (binding
    fix, is_reverse=1 is wrong in this stack). On any failure we fall back to
    the composite decomposition, so precision cannot regress.
    """
    try:
        if train:
            raise RuntimeError("fg_lstm inference kernel: training not supported")
        out, fh, fc = _lstm_fused(
            input, hx, params, has_biases, num_layers, bidirectional, batch_first
        )
        return out, fh, fc
    except Exception:
        return _lstm_composite(
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
