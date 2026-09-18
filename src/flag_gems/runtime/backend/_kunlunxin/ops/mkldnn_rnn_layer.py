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

from flag_gems.ops.linear import linear as _linear

logger = logging.getLogger(__name__)


def _lstm_gates(gates, hidden_size):
    """Split the packed (batch, 4H) pre-activations into i/f/g/o gates."""
    H = hidden_size
    i_g = gates[:, 0:H]
    f_g = gates[:, H : 2 * H]
    g_g = gates[:, 2 * H : 3 * H]
    o_g = gates[:, 3 * H : 4 * H]
    return torch.sigmoid(i_g), torch.sigmoid(f_g), torch.tanh(g_g), torch.sigmoid(o_g)


def _lstm_forward_folded(input, w_ih, w_hh, b_ih, b_hh, hx, cx, reverse):
    """Folded single-layer unidirectional LSTM forward (mode=2).

    XPU cannot compile the generic fused Triton kernel (a 2D weight tile + 2D
    reduction inside the sequential time loop overflows ``uni_sram`` at
    ``TritonXPUCoreTiling``), so the recurrence is folded into a minimal
    sequence of primitive ops, mirroring the ``rnn_relu`` kunlunxin override:

    * the input projection ``W_ih @ x + b_ih`` is pre-computed once as a single
      batched ``linear`` (M = seq*batch);
    * each step fuses the hidden recurrence with ``addmm`` (``pre[t] + h @ W_hh^T``);
    * element-wise gates are ``sigmoid``/``tanh``/``mul``.

    All accumulation is done in fp32 to match the oneDNN reference (which
    computes in fp32 and casts back at the end) for fp16/bf16 inputs.
    """
    seq_len, batch_size, input_size = input.shape
    hidden_size = w_hh.shape[1]  # w_hh is packed (4H, H)
    H = hidden_size

    x = input.float().contiguous()
    w_ih_f = w_ih.float().contiguous()
    w_hh_f = w_hh.float().contiguous()
    h = hx.float().contiguous()  # (batch, H)
    c = cx.float().contiguous()  # (batch, H)

    if b_ih is not None:
        b_ih_f = b_ih.float().contiguous()
    else:
        b_ih_f = torch.zeros(4 * H, dtype=torch.float32, device=input.device)
    if b_hh is not None:
        b_hh_f = b_hh.float().contiguous()
    else:
        b_hh_f = torch.zeros(4 * H, dtype=torch.float32, device=input.device)

    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    try:
        # ---- fast path: batched input projection + fused addmm recurrence ----
        x2d = x.reshape(seq_len * batch_size, input_size)
        pre = _linear(x2d, w_ih_f, b_ih_f).reshape(seq_len, batch_size, 4 * H)
        w_hh_t = w_hh_f.t().contiguous()  # (H, 4H)
        outputs = [None] * seq_len
        for t in steps:
            gates = pre[t] + torch.addmm(b_hh_f, h, w_hh_t)
            i_g, f_g, g_g, o_g = _lstm_gates(gates, H)
            c = f_g * c + i_g * g_g
            h = o_g * torch.tanh(c)
            outputs[t] = h
    except ZeroDivisionError:
        # ---- safe path: per-step small linear (M=batch), crash-free ----
        # On very small shapes the libtuner do_bench estimate for the big-M
        # linear rounds to 0 and raises ZeroDivisionError while cold-tuning.
        # Per-step linear matmuls (M = batch) never hit that edge.
        outputs = [None] * seq_len
        for t in steps:
            ih_t = _linear(x[t], w_ih_f, b_ih_f)  # (batch, 4H)
            hh_t = _linear(h, w_hh_f, b_hh_f)  # (batch, 4H)
            gates = ih_t + hh_t
            i_g, f_g, g_g, o_g = _lstm_gates(gates, H)
            c = f_g * c + i_g * g_g
            h = o_g * torch.tanh(c)
            outputs[t] = h

    output = torch.stack(outputs, dim=0)  # (seq_len, batch, H)
    return output.to(input.dtype), h.to(input.dtype), c.to(input.dtype)


class MkldnnRnnLayerFunction(torch.autograd.Function):
    """Autograd for a single-layer unidirectional LSTM (oneDNN mkldnn_rnn_layer).

    Forward  → folded primitive ops (fast).
    Backward → recompute the forward in native PyTorch, then let autograd
               differentiate it. The folded forward is opaque to autograd, so
               the graph is reconstructed from native ops for the backward.
    """

    @staticmethod
    def forward(
        ctx,
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
        logger.debug("GEMS_KUNLUNXIN MKLDNN_RNN_LAYER FORWARD")

        output, hy, cy = _lstm_forward_folded(
            input, w_ih, w_hh, b_ih, b_hh, hx, cx, reverse
        )

        ctx.save_for_backward(input, w_ih, w_hh, b_ih, b_hh, hx, cx)
        ctx.reverse = reverse
        ctx.hidden_size = hidden_size
        ctx.has_biases = has_biases

        # workspace is an opaque oneDNN buffer only consumed by the (unsupported)
        # mkldnn_rnn_layer_backward; expose an empty placeholder to satisfy the
        # 4-tensor schema.
        workspace = torch.empty(0, dtype=input.dtype, device=input.device)
        return output, hy, cy, workspace

    @staticmethod
    def backward(ctx, grad_output, grad_hy, grad_cy, grad_workspace):
        logger.debug("GEMS_KUNLUNXIN MKLDNN_RNN_LAYER BACKWARD")

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
                i_g = torch.sigmoid(i_g)
                f_g = torch.sigmoid(f_g)
                g_g = torch.tanh(g_g)
                o_g = torch.sigmoid(o_g)
                c = f_g * c + i_g * g_g
                h = o_g * torch.tanh(c)
                outputs.append(h)

            if reverse:
                outputs = outputs[::-1]
            output_native = torch.stack(outputs, dim=0)
            hy_native = h
            cy_native = c

            inputs = [input, w_ih, w_hh, hx, cx]
            if has_biases:
                inputs += [b_ih, b_hh]

            grads = torch.autograd.grad(
                outputs=[output_native, hy_native, cy_native],
                inputs=inputs,
                grad_outputs=[
                    grad_output.reshape(output_native.shape),
                    grad_hy.reshape(hy_native.shape),
                    grad_cy.reshape(cy_native.shape),
                ],
                retain_graph=False,
                allow_unused=True,
            )

        grad_input = grads[0]
        grad_w_ih = grads[1]
        grad_w_hh = grads[2]
        grad_hx = grads[3]
        grad_cx = grads[4]
        if has_biases:
            grad_b_ih = grads[5]
            grad_b_hh = grads[6]
        else:
            grad_b_ih = None
            grad_b_hh = None

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
    the corresponding biases ``(4H,)``. Returns ``(output, hy, cy, workspace)``;
    the oneDNN ``workspace`` is opaque and only consumed by the backward pass, so
    an empty placeholder is returned. Multi-layer, bidirectional, packed
    (``batch_sizes``), ``batch_first`` and non-LSTM ``mode`` all raise
    ``NotImplementedError``.
    """
    logger.debug("GEMS_KUNLUNXIN MKLDNN_RNN_LAYER")

    if mode != 2:
        raise NotImplementedError("GEMS MKLDNN_RNN_LAYER only supports LSTM (mode=2)")
    if num_layers != 1 or bidirectional:
        raise NotImplementedError(
            "GEMS MKLDNN_RNN_LAYER only supports single-layer unidirectional"
        )
    if batch_first:
        raise NotImplementedError(
            "GEMS MKLDNN_RNN_LAYER only supports batch_first=False (T, N, *) layout"
        )
    if batch_sizes is not None and len(batch_sizes) > 0:
        raise NotImplementedError(
            "GEMS MKLDNN_RNN_LAYER does not support packed sequences (batch_sizes)"
        )

    # ``train`` is part of the 16-arg aten schema but does not change the result
    # here: a single-layer LSTM has no dropout, so the forward output is
    # train-independent, and backward is supplied by MkldnnRnnLayerFunction
    # rather than a oneDNN train-mode reserve/workspace.
    del train

    return MkldnnRnnLayerFunction.apply(
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


def _redirect_generic_entry_points():
    """Redirect the generic ``flag_gems.ops.mkldnn_rnn_layer`` entry points here.

    ``SpecOpRegistrar`` only replaces the top-level ``flag_gems.mkldnn_rnn_layer``
    (the ``torch.mkldnn_rnn_layer`` dispatch target under ``use_gems()``).  The
    generic ``flag_gems.ops.mkldnn_rnn_layer`` is a fused Triton LSTM whose 2D
    weight tile + reduction inside the sequential time loop cannot compile on XPU
    (``TritonXPUCoreTiling`` / ``uni_sram`` overflow), so the direct-wrapper and
    direct-backward tests (``from flag_gems.ops.mkldnn_rnn_layer import
    mkldnn_rnn_layer``) and the benchmark (``gems_op=flag_gems.ops.mkldnn_rnn_layer``)
    would still hit the broken kernel.  Patch both entry points to this folded
    override so they exercise the XPU implementation.
    """
    import sys

    import flag_gems.ops as _flag_gems_ops

    # (1) package attribute reached via `flag_gems.ops.mkldnn_rnn_layer`.
    setattr(_flag_gems_ops, "mkldnn_rnn_layer", mkldnn_rnn_layer)
    # (2) submodule function reached via `from flag_gems.ops.mkldnn_rnn_layer
    #     import mkldnn_rnn_layer`.
    _generic_submodule = sys.modules.get("flag_gems.ops.mkldnn_rnn_layer")
    if _generic_submodule is not None:
        setattr(_generic_submodule, "mkldnn_rnn_layer", mkldnn_rnn_layer)


_redirect_generic_entry_points()
