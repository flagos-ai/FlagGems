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

"""kunlunxin (XPU) backend implementation of ``mkldnn_rnn_layer``.

The GENERIC ``flag_gems/ops/mkldnn_rnn_layer.py`` is a KernelGen Triton kernel
that cannot compile on the XPU triton fork:

    OutOfResources out of resource: uni_sram
    TritonXPUCoreTiling: 'arith.addi' op requires the same encoding for all
    operands and results

(``mkldnn_rnn_layer.py:101``, the gate-offset computation). The kernel keeps
eight masked ``(BLOCK_H, BLOCK_IN/BLOCK_H)`` weight tiles live across the whole
time loop; the XPU core-tiling pass runs out of uni_sram and emits the
encoding error (probe matrix on card 7, 2026-09-05: the identical pattern
fails at 8x8 / 16x16 shapes, and removing the masks moves the failure to the
``cstate`` mul, i.e. it is the multi-tile liveness, not the mask).

Per the backend convention for recurrent kernels (see ``rnn_relu``), the
recurrence is instead evaluated with a minimal chain of native primitives on
the device (fp32 accumulation, matching the oneDNN reference semantics):

    pre    = x @ W_ih^T + b_ih + b_hh                 # batched over all steps
    gates  = pre[t] + h @ W_hh^T
    i,f,o  = sigmoid(gates_i/f/o);  g = tanh(gates_g)
    c'     = f*c + i*g ;  h' = o*tanh(c')

The chain is fully autograd-tracked (no custom backward needed) and matches
the analytical reference that the tests use. ``@`` uses plain ``torch.mm``
(never ``torch.addmm``: the kunlunxin ``addmm`` override raises under
``use_gems``); under ``use_gems`` the mm/sigmoid/tanh/mul/add/stack calls
dispatch to the (validated) kunlunxin vendor implementations.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Gate order of the packed (4H, *) oneDNN weights: i, f, g, o.
_G_OFFSETS = (0, 1, 2, 3)


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

    Same contract as the generic ``flag_gems.ops.mkldnn_rnn_layer``:
    ``weight0/weight1`` are the input- and hidden-to-hidden weights
    ``(4H, input)`` / ``(4H, H)`` and ``weight2/weight3`` the corresponding
    biases ``(4H,)``. Returns ``(output, hy, cy, workspace)`` with an empty
    workspace placeholder. Multi-layer, bidirectional, packed (``batch_sizes``),
    ``batch_first`` and non-LSTM ``mode`` raise ``NotImplementedError``.
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
    # for a single-layer LSTM (no dropout); the autograd graph is live either way.
    del train

    w_ih, w_hh, b_ih, b_hh = weight0, weight1, weight2, weight3
    seq_len, batch_size, _ = input.shape

    if not has_biases:
        # oneDNN always applies both bias vectors; emulate no-bias by zeros.
        b_ih = torch.zeros(4 * hidden_size, dtype=input.dtype, device=input.device)
        b_hh = torch.zeros(4 * hidden_size, dtype=input.dtype, device=input.device)

    # All accumulation in fp32 to match the oneDNN/analytical reference
    # regardless of the input dtype (the tests cast to float as well).
    x2d = input.reshape(seq_len * batch_size, -1).to(torch.float32)
    w_ih_f = w_ih.to(torch.float32)
    w_hh_f = w_hh.to(torch.float32)
    b_ih_f = b_ih.to(torch.float32)
    b_hh_f = b_hh.to(torch.float32)

    # Input-to-hidden gates for all time steps in one batched matmul:
    # (T*B, I) @ (I, 4H) + (b_ih + b_hh)  ->  (T*B, 4H)
    pre = x2d @ w_ih_f.t()
    pre = (pre + b_ih_f) + b_hh_f
    pre = pre.reshape(seq_len, batch_size, 4 * hidden_size)

    h = hx_.to(torch.float32)
    c = cx_.to(torch.float32)
    w_hh_t = w_hh_f.t().contiguous()

    outputs = [None] * seq_len
    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    for t in steps:
        gates = pre[t] + h @ w_hh_t  # (B, 4H), fp32
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outputs[t] = h

    output = torch.stack(outputs, dim=0).to(input.dtype)
    hy = h.to(input.dtype)
    cy = c.to(input.dtype)

    # workspace is an opaque oneDNN buffer only consumed by the (unsupported)
    # mkldnn_rnn_layer_backward; expose an empty placeholder to satisfy the
    # 4-tensor schema.
    workspace = torch.empty(0, dtype=input.dtype, device=input.device)

    return output, hy, cy, workspace


__all__ = ["mkldnn_rnn_layer"]


def _patch_generic_wrapper():
    """Route direct calls to the generic wrapper (flag_gems.ops.mkldnn_rnn_layer
    module) to this backend override.

    The direct-wrapper tests and ``flag_gems.ops`` benchmarks import the op
    through ``flag_gems.ops.mkldnn_rnn_layer`` (bypassing the top-level
    ``flag_gems`` registry that SpecOpRegistrar patches), so the generic
    KernelGen Triton kernel would still be hit on XPU (it cannot compile
    there: uni_sram / TritonXPUCoreTiling failures). Patching the module
    attribute at import time keeps the change backend-local: the generic
    module source is untouched and other vendor backends are unaffected
    (this module is only imported for the kunlunxin backend).
    """
    try:
        import sys

        _generic_module = sys.modules.get("flag_gems.ops.mkldnn_rnn_layer")
        if _generic_module is not None and hasattr(_generic_module, "mkldnn_rnn_layer"):
            _generic_module.mkldnn_rnn_layer = mkldnn_rnn_layer
        # ``from .mkldnn_rnn_layer import mkldnn_rnn_layer`` in the ops package
        # __init__ binds the *generic* function as the package attribute, so
        # ``flag_gems.ops.mkldnn_rnn_layer`` (the benchmark entry point) must be
        # re-bound to this backend implementation as well.
        import flag_gems.ops as _ops

        if hasattr(_ops, "mkldnn_rnn_layer"):
            _ops.mkldnn_rnn_layer = mkldnn_rnn_layer
    except ImportError:
        pass


_patch_generic_wrapper()