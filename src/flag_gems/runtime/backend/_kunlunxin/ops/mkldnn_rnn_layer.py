# Copyright 2026, The FlagOS Contributors.
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
# Kunlunxin (XPU3) specialization of ``aten::mkldnn_rnn_layer``.
#
# The operator is a single-layer, unidirectional LSTM over a sequence (the
# oneDNN CPU entry point).  The generic Triton implementation runs the whole
# recurrence inside *one* kernel (one program per batch element, a 2-D
# ``tl.sum`` mat-vec per gate) and does not compile on this backend at all:
# ``TritonXPUCoreTiling`` aborts with ``out of resource: uni_sram`` /
# ``'arith.addi' op requires the same encoding for all operands`` (measured on
# every one of the 193 accuracy cells).  On the XPU there is no oneDNN backend
# either, so the op currently reaches neither path.
#
# This module re-expresses the same recurrence with the proven ``lstm_cell``
# recipe:
#
#   1. ``_mkldnn_proj_kernel`` -- one mask-free, rounded ``x @ w^T`` projection.
#      It is reused for the input projection of *all* timesteps (one launch,
#      ``(T*B, I) @ (I, 4H)``) and for the hidden projection of a single
#      timestep (``(B, H) @ (H, 4H)``, one launch per step).  The weight is fed
#      as the transposed B operand (``w[n, k]`` contiguous over ``k``), the
#      batch index wraps with ``% M`` and the staging-buffer rows are padded to
#      the block, so the ``tl.dot`` is mask-free and in-bounds.  Both operands
#      are upcast to fp32 before the dot, so the projection matches ATen's
#      fp32 ``addmm`` regardless of the input dtype.
#   2. ``mkldnn_rnn_layer_gate_kernel`` (from ``_mkldnn_rnn_layer_gate``) -- the
#      flat, mask-free LSTM gate chain for one timestep.  It keeps the
#      recurrence state in fp32 buffers (like the generic kernel, which never
#      rounds ``h``/``c`` between steps) and rounds only the stores.  It lives
#      in a sibling module because a ``tl.sigmoid``/``tl.tanh`` kernel defined
#      next to a ``tl.dot`` kernel fails a *cold* XPU compile
#      (``KeyError: 'tt.sigmoid'``).
#
# Launch count is ``1 + 2 * seq_len``.  Shapes whose input/hidden size is not a
# multiple of 8, a single-row batch, a non-contiguous operand or an empty
# sequence fall back to the generic FlagGems implementation (correct, edge-only;
# never CPU/ATen).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._kunlunxin.ops._mkldnn_rnn_layer_gate import (
    mkldnn_rnn_layer_final_kernel as _final_kernel,
)
from flag_gems.runtime.backend._kunlunxin.ops._mkldnn_rnn_layer_gate import (
    mkldnn_rnn_layer_gate_kernel as _gate_kernel,
)
from flag_gems.runtime.backend._kunlunxin.ops._mkldnn_rnn_layer_gate import (
    mkldnn_rnn_layer_gate_kernel_full as _gate_kernel_full,
)

logger = logging.getLogger(__name__)

_ACC_DTYPE = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
}


def _lowpow2(n):
    """Largest power of two dividing ``n`` (``n > 0``)."""
    return n & (-n)


def _proj_block_k(k):
    """Largest of {64, 32, 16, 8} dividing ``k`` (0 when none does)."""
    if k % 64 == 0:
        return 64
    if k % 32 == 0:
        return 32
    if k % 16 == 0:
        return 16
    if k % 8 == 0:
        return 8
    return 0


@triton.jit
def _mkldnn_proj_kernel(
    x_ptr,
    w_ptr,
    c_ptr,
    M,
    s_xm,
    s_xk,
    s_wn,
    s_wk,
    s_cm,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Rounded fp32 ``x @ w^T`` into ``c``.

    ``w`` is stored ``(N, K)`` row-major (contiguous in K) and loaded as the
    transposed B operand.  Rows wrap with ``% M`` so the scratch rows of the
    padded staging buffers are discarded rather than masked; every reduction
    and column extent is a block multiple, so nothing is masked.  ``% 1``
    miscompiles on this backend, so ``M == 1`` never reaches this kernel (the
    caller falls back to the generic path).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + ram[:, None] * s_xm + rk[None, :] * s_xk).to(tl.float32)
        w = tl.load(w_ptr + rn[None, :] * s_wn + rk[:, None] * s_wk).to(tl.float32)
        # ``input_precision="ieee"`` keeps the fp32 projection exact like the
        # generic implementation and ATen's fp32 ``addmm`` instead of the
        # default reduced (tf32-like) precision.  The 2-arg ``acc +=`` form is
        # mandatory: the 3-arg chain miscompiles on this backend.
        acc += tl.dot(x, w, input_precision="ieee", out_dtype=tl.float32)
    tl.store(c_ptr + rm[:, None] * s_cm + rn[None, :], acc)


def _fast_ok(input, w_ih, w_hh, b_ih, b_hh, hx, cx, input_size, hidden_size):
    if input.dtype not in _ACC_DTYPE:
        return False
    seq_len, batch, in_size = input.shape
    # ``rm % M`` (the batch wrap) miscompiles at M == 1.
    if seq_len < 1 or batch < 2:
        return False
    if _proj_block_k(in_size) == 0 or _proj_block_k(hidden_size) == 0:
        return False
    if input_size < 8 or hidden_size < 8:
        return False
    if not input.is_contiguous():
        return False
    if w_ih.dim() != 2 or not w_ih.is_contiguous() or w_ih.shape != (
        4 * hidden_size,
        input_size,
    ):
        return False
    if w_hh.dim() != 2 or not w_hh.is_contiguous() or w_hh.shape != (
        4 * hidden_size,
        hidden_size,
    ):
        return False
    for t in (hx, cx):
        if not t.is_contiguous() or t.shape != (batch, hidden_size):
            return False
    for b in (b_ih, b_hh):
        if b is not None and (not b.is_contiguous() or b.numel() != 4 * hidden_size):
            return False
    return True


def _mkldnn_rnn_layer_forward(
    input, w_ih, w_hh, b_ih, b_hh, hx, cx, reverse, hidden_size, has_biases
):
    """Launch the Triton LSTM kernels for a single unidirectional layer."""
    seq_len, batch, input_size = input.shape
    device = input.device
    out_dtype = input.dtype
    gate_n = 4 * hidden_size

    block_n = min(128, _lowpow2(hidden_size))

    # Input projection for every timestep in one launch: (T*B, I) @ (I, 4H).
    x2d = input.reshape(seq_len * batch, input_size)
    m_x = seq_len * batch
    block_m_x = min(128, max(16, triton.next_power_of_2(m_x)))
    mp_x = ((m_x + block_m_x - 1) // block_m_x) * block_m_x
    gx = torch.empty((mp_x, gate_n), dtype=torch.float32, device=device)
    _mkldnn_proj_kernel[(mp_x // block_m_x, gate_n // block_n)](
        x2d,
        w_ih,
        gx,
        m_x,
        x2d.stride(0),
        1,
        w_ih.stride(0),
        w_ih.stride(1),
        gx.stride(0),
        K=input_size,
        BLOCK_M=block_m_x,
        BLOCK_N=block_n,
        BLOCK_K=_proj_block_k(input_size),
    )

    # Hidden projection buffer, one row block reused at every timestep.
    block_m_h = min(128, max(16, triton.next_power_of_2(batch)))
    mp_h = ((batch + block_m_h - 1) // block_m_h) * block_m_h
    gh = torch.empty((mp_h, gate_n), dtype=torch.float32, device=device)
    block_k_h = _proj_block_k(hidden_size)

    # oneDNN always applies both bias vectors; emulate no-bias by zeros.
    if has_biases and b_ih is not None and b_hh is not None:
        bi, bh = b_ih, b_hh
    else:
        bi = torch.zeros(gate_n, dtype=out_dtype, device=device)
        bh = torch.zeros(gate_n, dtype=out_dtype, device=device)

    # Recurrence state kept in fp32 (the generic kernel never rounds h/c
    # between steps either); only the stores are rounded.
    hseq = torch.empty((seq_len, batch, hidden_size), dtype=torch.float32, device=device)
    cbuf = torch.empty((batch, hidden_size), dtype=torch.float32, device=device)

    output = torch.empty(
        (seq_len, batch, hidden_size), dtype=out_dtype, device=device
    )
    hy = torch.empty((batch, hidden_size), dtype=out_dtype, device=device)
    cy = torch.empty((batch, hidden_size), dtype=out_dtype, device=device)

    n = batch * hidden_size
    # Sub-warp flat tiles are widened to a full warp by this backend (an
    # unmasked store then spills past small buffers), so never go below 64.
    flat_block = min(1024, max(64, _lowpow2(n)))
    grid_gate = ((n + flat_block - 1) // flat_block,)
    # The tail predicate costs ~3.5x on this backend, so use the mask-free
    # variant whenever the flat extent is an exact multiple of the block.
    if n % flat_block == 0:
        gate_fn = _gate_kernel_full
        gate_extra = ()
    else:
        gate_fn = _gate_kernel
        gate_extra = (n,)

    steps = range(0, seq_len)
    if reverse:
        steps = range(seq_len - 1, -1, -1)

    # The first step reads the caller's ``cx`` directly (the gate kernel
    # upcasts it), later steps read the fp32 recurrence buffer.
    c_src = cx
    h_src = hx
    last_step = 0 if reverse else seq_len - 1
    for t in steps:
        _mkldnn_proj_kernel[(mp_h // block_m_h, gate_n // block_n)](
            h_src,
            w_hh,
            gh,
            batch,
            h_src.stride(0),
            1,
            w_hh.stride(0),
            w_hh.stride(1),
            gh.stride(0),
            K=hidden_size,
            BLOCK_M=block_m_h,
            BLOCK_N=block_n,
            BLOCK_K=block_k_h,
        )
        gate_fn[grid_gate](
            gx[t * batch : (t + 1) * batch],
            gh[:batch],
            c_src,
            bi,
            bh,
            hseq[t],
            cbuf,
            output[t],
            *gate_extra,
            HIDDEN_SIZE=hidden_size,
            HAS_BIAS=True,
            IS_FP64=False,
            BLOCK_SIZE=flat_block,
        )
        h_src = hseq[t]
        c_src = cbuf

    # The last executed step holds the final state; publish it as hy/cy.
    _final_kernel[grid_gate](
        hseq[last_step],
        cbuf,
        hy,
        cy,
        n,
        BLOCK_SIZE=flat_block,
    )

    return output, hy, cy


class MkldnnRnnLayerFunction(torch.autograd.Function):
    """Autograd for a single-layer unidirectional LSTM (oneDNN mkldnn_rnn_layer).

    Forward  -> Triton kernels (fast).
    Backward -> recompute the forward in native PyTorch, then let autograd
                differentiate it (identical to the generic FlagGems backward,
                which is the only available backward for this opaque kernel).
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
        logger.debug("GEMS MKLDNN_RNN_LAYER_FUNCTION_FORWARD")

        output, hy, cy = _mkldnn_rnn_layer_forward(
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
        )

        ctx.save_for_backward(input, w_ih, w_hh, b_ih, b_hh, hx, cx)
        ctx.reverse = reverse
        ctx.hidden_size = hidden_size
        ctx.has_biases = has_biases

        workspace = torch.empty(0, dtype=input.dtype, device=input.device)
        return output, hy, cy, workspace

    @staticmethod
    def backward(ctx, grad_output, grad_hy, grad_cy, grad_workspace):
        logger.debug("GEMS MKLDNN_RNN_LAYER_FUNCTION_BACKWARD")

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
    the corresponding biases ``(4H,)``.  Returns ``(output, hy, cy, workspace)``;
    the oneDNN ``workspace`` is opaque and only consumed by the backward pass, so
    an empty placeholder is returned.  Multi-layer, bidirectional, packed
    (``batch_sizes``), ``batch_first`` and non-LSTM ``mode`` all raise
    ``NotImplementedError`` (identical to the generic FlagGems contract).
    """
    logger.debug("GEMS MKLDNN_RNN_LAYER")

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

    # ``train`` does not change the result here: a single-layer LSTM has no
    # dropout, and backward is supplied by MkldnnRnnLayerFunction rather than a
    # oneDNN train-mode reserve/workspace.
    del train

    input_size = input.shape[2]
    if not _fast_ok(
        input, weight0, weight1, weight2, weight3, hx_, cx_, input_size, hidden_size
    ):
        from flag_gems.ops.mkldnn_rnn_layer import mkldnn_rnn_layer as _generic

        return _generic(
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
        )

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


# The dedicated tests import ``flag_gems.ops.mkldnn_rnn_layer`` directly (not
# the aten-dispatch entry) and the benchmark passes
# ``flag_gems.ops.mkldnn_rnn_layer``, so publish the override on both the
# submodule and the ``flag_gems.ops`` package -- the same pattern the Kunlunxin
# ``_ctc_loss`` / ``conj_physical`` overrides use.
import sys as _sys  # noqa: E402

for _mod_name in ("flag_gems.ops.mkldnn_rnn_layer", "flag_gems.ops"):
    _mod = _sys.modules.get(_mod_name)
    if _mod is not None:
        setattr(_mod, "mkldnn_rnn_layer", mkldnn_rnn_layer)
