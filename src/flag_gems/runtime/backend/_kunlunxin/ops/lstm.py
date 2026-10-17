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
    # Stub kernel; the fused LSTM launch is handled by the backend.
    tl.program_id(0)


def _run_fusion(x, init_h, init_c, wx, wh, bx, bh, y, last_h, last_c, is_reverse):
    seq_len, batch_size, xdim = x.shape
    hdim = last_h.shape[-1]
    # quantization scales
    wx_max = wx.abs().amax().float().reshape(1)
    wh_max = wh.abs().amax().float().reshape(1)
    # upcast bias to fp32
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
    """Run one direction of one layer."""
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
        # reverse via flip-input / flip-output
        x_in = torch.flip(input, dims=[0]).contiguous()
    # bf16: compute in fp32 and cast back
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
    """Multi-layer (optionally bidirectional) LSTM.

    Each direction is a single fused launch; layers and directions are composed
    in Python.
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
    """aten::lstm via the backend fused LSTM.

    Vendor failures are propagated rather than hidden by a native/composite
    fallback.
    """
    if train:
        raise NotImplementedError("fused LSTM inference does not support training")
    return _lstm_fused(
        input, hx, params, has_biases, num_layers, bidirectional, batch_first
    )
