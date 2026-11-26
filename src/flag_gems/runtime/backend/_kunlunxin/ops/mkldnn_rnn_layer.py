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

"""Kunlunxin XPU3 override for ``mkldnn_rnn_layer`` (single-layer LSTM, mode=2).

The generic kernel keeps one program per batch element and forms every gate
with a register-resident ``tl.sum(w * x, axis=1)`` reduction.  XPU3 rejects
that shape in ``TritonXPUCoreTiling`` (``'arith.addi' op requires the same
encoding for all operands and results``), so the generic file cannot compile
here at all.

The maths is the one ``lstm`` already implements for this backend, so this
override reuses the three ``lstm`` step kernels verbatim instead of carrying a
second copy of them: one ``tl.dot`` projection for ``x[t]``, one for ``h``, and
one elementwise kernel that adds the biases and runs the gate chain.  Keeping
the elementwise part in its own module matters here, because an elementwise op
applied to a ``tl.dot`` result in the same module makes the backend collect it
into an elementwise lookup table that aborts the compile.
"""

import logging

import torch
import triton

from flag_gems.ops.lstm import (
    _lstm_step_combine_kernel,
    _lstm_step_hh_kernel,
    _lstm_step_ih_kernel,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# One program covers BLOCK_B batch rows and BLOCK_H hidden columns.  BLOCK_B is
# pinned at 8 and BLOCK_H has a floor of 16: below that the backend inliner
# fails on the four-dot K-loop.  BLOCK_K is pinned at its maximum and the edges
# are masked instead.
_BLOCK_B = 8
_BLOCK_H_MIN = 16
_BLOCK_H_MAX = 32
_BLOCK_K = 64


def _ceil_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _mkldnn_rnn_layer_forward(
    input,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    hx,
    cx,
    reverse,
    hidden_size,
    has_biases,
):
    """Launch the Triton LSTM kernels for a single unidirectional layer."""
    seq_len, batch_size, input_size = input.shape

    input = input.contiguous()
    w_ih = w_ih.contiguous()
    w_hh = w_hh.contiguous()
    hx = hx.contiguous()
    cx = cx.contiguous()

    if has_biases:
        b_ih = b_ih.contiguous()
        b_hh = b_hh.contiguous()
    else:
        # oneDNN always applies both bias vectors; emulate no-bias by zeros.
        b_ih = torch.zeros(4 * hidden_size, dtype=input.dtype, device=input.device)
        b_hh = torch.zeros(4 * hidden_size, dtype=input.dtype, device=input.device)

    output = torch.empty(
        (seq_len, batch_size, hidden_size), dtype=input.dtype, device=input.device
    )

    # The recurrent state is double-buffered: each step reads ``h_work``/``c_work``
    # and writes ``h_next``/``c_next``, then the two are swapped.  The buffers are
    # private copies of ``hx``/``cx``: the caller's tensors are inputs and must
    # not be overwritten by the recurrence.
    h_work = hx.clone()
    c_work = cx.clone()
    h_next = torch.empty_like(h_work)
    c_next = torch.empty_like(c_work)

    block_h = min(max(_ceil_power_of_2(hidden_size), _BLOCK_H_MIN), _BLOCK_H_MAX)
    grid = (triton.cdiv(batch_size, _BLOCK_B), triton.cdiv(hidden_size, block_h))

    # Per-gate partial sums in accumulator precision, consumed by the combine
    # kernel: gi = x @ w_ih.T, gh = h_prev @ w_hh.T.
    gi = torch.empty(
        (batch_size, 4 * hidden_size), dtype=torch.float32, device=input.device
    )
    gh = torch.empty_like(gi)

    with torch_device_fn.device(input.device):
        if batch_size == 0:
            return output, hx, cx
        if input_size == 0:
            # An empty reduction contributes nothing (the bias is added by the
            # combine kernel), and a zero-trip loop is miscompiled on XPU, so
            # write the zeros directly instead of launching the projection.
            gi.zero_()
        for step in range(seq_len):
            # reverse=True consumes the sequence back-to-front.
            seq_idx = seq_len - 1 - step if reverse else step
            if input_size != 0:
                _lstm_step_ih_kernel[grid](
                    input,
                    w_ih,
                    gi,
                    seq_idx,
                    input_size,
                    hidden_size,
                    batch_size,
                    input.stride(0),
                    input.stride(1),
                    input.stride(2),
                    w_ih.stride(0),
                    w_ih.stride(1),
                    gi.stride(0),
                    gi.stride(1),
                    BLOCK_B=_BLOCK_B,
                    BLOCK_H=block_h,
                    BLOCK_K=_BLOCK_K,
                )
            _lstm_step_hh_kernel[grid](
                h_work,
                w_hh,
                gh,
                hidden_size,
                batch_size,
                h_work.stride(0),
                h_work.stride(1),
                w_hh.stride(0),
                w_hh.stride(1),
                gh.stride(0),
                gh.stride(1),
                BLOCK_B=_BLOCK_B,
                BLOCK_H=block_h,
                BLOCK_K=_BLOCK_K,
            )
            _lstm_step_combine_kernel[grid](
                gi,
                gh,
                c_work,
                b_ih,
                b_hh,
                h_next,
                c_next,
                output,
                seq_idx,
                0,
                hidden_size,
                batch_size,
                gi.stride(0),
                gi.stride(1),
                gh.stride(0),
                gh.stride(1),
                c_work.stride(0),
                c_work.stride(1),
                b_ih.stride(0),
                b_hh.stride(0),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                HAS_BIAS=has_biases,
                BLOCK_B=_BLOCK_B,
                BLOCK_H=block_h,
            )
            h_work, h_next = h_next, h_work
            c_work, c_next = c_next, c_work

    if h_work is hx:
        # seq_len == 0: nothing was written, so hand back a private copy rather
        # than the caller's own tensor.
        h_work = h_work.clone()
        c_work = c_work.clone()
    return output, h_work, c_work


class _MkldnnRnnLayerFunction(torch.autograd.Function):
    """Autograd for the single-layer unidirectional LSTM.

    Forward  → Triton kernels.  Backward → recompute the forward in native
    PyTorch and let autograd differentiate it, because the Triton kernels are
    opaque to autograd.
    """

    @staticmethod
    def forward(
        ctx, input, w_ih, w_hh, b_ih, b_hh, hx, cx, reverse, hidden_size, has_biases
    ):
        output, hy, cy = _mkldnn_rnn_layer_forward(
            input, w_ih, w_hh, b_ih, b_hh, hx, cx, reverse, hidden_size, has_biases
        )
        ctx.save_for_backward(input, w_ih, w_hh, b_ih, b_hh, hx, cx)
        ctx.reverse = reverse
        ctx.hidden_size = hidden_size
        ctx.has_biases = has_biases
        # The oneDNN workspace is opaque and only consumed by the backward; an
        # empty placeholder satisfies the 4-tensor schema.
        workspace = torch.empty(0, dtype=input.dtype, device=input.device)
        return output, hy, cy, workspace

    @staticmethod
    def backward(ctx, grad_output, grad_hy, grad_cy, grad_workspace):
        input, w_ih, w_hh, b_ih, b_hh, hx, cx = ctx.saved_tensors
        reverse = ctx.reverse
        has_biases = ctx.has_biases
        seq_len = input.shape[0]

        with torch.enable_grad():
            h = hx.clone()
            c = cx.clone()
            outputs = []
            steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            for t_idx in steps:
                xt = input[t_idx]
                gates = (
                    torch.addmm(b_ih, xt, w_ih.t())
                    if has_biases
                    else torch.mm(xt, w_ih.t())
                )
                gates = gates + (
                    torch.addmm(b_hh, h, w_hh.t())
                    if has_biases
                    else torch.mm(h, w_hh.t())
                )
                i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
                c = torch.sigmoid(f_g) * c + torch.sigmoid(i_g) * torch.tanh(g_g)
                h = torch.sigmoid(o_g) * torch.tanh(c)
                outputs.append(h)
            if reverse:
                outputs = outputs[::-1]
            output_native = torch.stack(outputs, dim=0)

            inputs = [input, w_ih, w_hh, hx, cx]
            if has_biases:
                inputs += [b_ih, b_hh]
            grads = torch.autograd.grad(
                outputs=[output_native, h, c],
                inputs=inputs,
                grad_outputs=[
                    grad_output.reshape(output_native.shape),
                    grad_hy.reshape(h.shape),
                    grad_cy.reshape(c.shape),
                ],
                retain_graph=False,
                allow_unused=True,
            )

        grad_input, grad_w_ih, grad_w_hh, grad_hx, grad_cx = grads[:5]
        if has_biases:
            grad_b_ih, grad_b_hh = grads[5], grads[6]
        else:
            grad_b_ih = grad_b_hh = None
        return (
            grad_input,
            grad_w_ih,
            grad_w_hh,
            grad_b_ih,
            grad_b_hh,
            grad_hx,
            grad_cx,
            None,  # reverse
            None,  # hidden_size
            None,  # has_biases
        )


def mkldnn_rnn_layer(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_,
    reverse,
    batch_sizes,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    train,
):
    """Single-layer unidirectional LSTM layer (oneDNN mkldnn_rnn_layer, mode=2).

    Mirrors ``torch.mkldnn_rnn_layer``: ``weight0/weight1`` are the input- and
    hidden-to-hidden weights ``(4H, input)`` / ``(4H, H)`` and ``weight2/weight3``
    the corresponding biases ``(4H,)``.  Returns ``(output, hy, cy, workspace)``;
    the oneDNN ``workspace`` is opaque and only consumed by the backward pass,
    so an empty placeholder is returned.  Multi-layer, bidirectional, packed
    (``batch_sizes``), ``batch_first`` and non-LSTM ``mode`` all raise
    ``NotImplementedError``.
    """
    logger.debug("GEMS_KUNLUNXIN MKLDNN_RNN_LAYER")

    if mode != 2:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER only supports LSTM (mode=2)"
        )
    if num_layers != 1 or bidirectional:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER only supports single-layer unidirectional"
        )
    if batch_first:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER only supports batch_first=False (T, N, *) layout"
        )
    if batch_sizes is not None and len(batch_sizes) > 0:
        raise NotImplementedError(
            "GEMS_KUNLUNXIN MKLDNN_RNN_LAYER does not support packed sequences (batch_sizes)"
        )

    # ``train`` does not change the forward result: a single-layer LSTM has no
    # dropout.
    del train

    return _MkldnnRnnLayerFunction.apply(
        input,
        weight0,
        weight1,
        weight2,
        weight3,
        hx_,
        cx_,
        reverse,
        hidden_size,
        has_biases,
    )


__all__ = ["mkldnn_rnn_layer"]
