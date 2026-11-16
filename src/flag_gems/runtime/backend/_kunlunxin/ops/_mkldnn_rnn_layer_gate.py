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
# Per-timestep LSTM gate kernel for the Kunlunxin ``mkldnn_rnn_layer`` override,
# kept in its own module: defining a ``tl.sigmoid``/``tl_extra_shim.tanh``
# kernel inside the operator module that also holds ``tl.dot`` kernels makes a
# *cold* XPU compile abort with ``KeyError: 'tt.sigmoid'`` once a dot kernel has
# been built in the same process, while the identical kernel defined/imported
# from a sibling module compiles cleanly.
# See ``wiki/patterns/xpu3-cold-compile-sigmoid-lut-module.md``.
#
# This is the elementwise tail of the generic ``mkldnn_rnn_layer`` LSTM step
# (add both rounded projections + both biases, run the gate chain, store the
# new h/c) rewritten in the proven mask-free *flat* layout used by
# ``_lstm_cell_gate`` / ``_gru_cell_gate``: one lane per ``(batch, hidden)``
# element, ``offsets // HIDDEN_SIZE`` row split.  ``HAS_BIAS``/``IS_FP64``/
# ``BLOCK_SIZE`` must stay ``tl.constexpr`` and the row split must stay as-is;
# simpler 1-D/2-D rewrites trip the same cold-compile bug.
#
# Extra store vs. the plain cell: the sequence output (one row per timestep,
# written straight into ``output[t]`` in the operator dtype).  It reuses the
# flat ``offsets`` (the gate buffers and the outputs share the ``(batch,
# hidden)`` layout), so every load/store stays unmasked.  The final ``hy``/
# ``cy`` are published by ``mkldnn_rnn_layer_final_kernel`` below -- a
# conditionally-executed copy of the same value inside the gate kernel was
# measured to write only some lanes.

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

_tanh = tl_extra_shim.tanh


@triton.jit
def mkldnn_rnn_layer_gate_kernel(
    input_gates,
    hidden_gates,
    cx,
    input_bias,
    hidden_bias,
    h_out,
    c_out,
    seq_out,
    N,
    HIDDEN_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # ``N = batch * hidden``.  Sub-warp tiles (``BLOCK_SIZE < 64``) are widened
    # to a full warp by this backend, so an unmasked flat store spills past the
    # end of small buffers and corrupts neighbouring allocations (measured: the
    # sequence output was clobbered for ``batch*hidden < 64``).  Loads therefore
    # clamp the lane index (always in-bounds, no masked-load ``other``
    # dependency) and stores carry an explicit tail mask.
    raw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lane_mask = raw < N
    offsets = tl.minimum(raw, N - 1)
    rows = offsets // HIDDEN_SIZE
    cols = offsets - rows * HIDDEN_SIZE
    gate_base = rows * (4 * HIDDEN_SIZE) + cols

    # Gate order is input, forget, cell, output.
    input_i = tl.load(input_gates + gate_base)
    input_f = tl.load(input_gates + gate_base + HIDDEN_SIZE)
    input_g = tl.load(input_gates + gate_base + 2 * HIDDEN_SIZE)
    input_o = tl.load(input_gates + gate_base + 3 * HIDDEN_SIZE)
    hidden_i = tl.load(hidden_gates + gate_base)
    hidden_f = tl.load(hidden_gates + gate_base + HIDDEN_SIZE)
    hidden_g = tl.load(hidden_gates + gate_base + 2 * HIDDEN_SIZE)
    hidden_o = tl.load(hidden_gates + gate_base + 3 * HIDDEN_SIZE)
    cx_value = tl.load(cx + offsets)

    if not IS_FP64:
        input_i = input_i.to(tl.float32)
        input_f = input_f.to(tl.float32)
        input_g = input_g.to(tl.float32)
        input_o = input_o.to(tl.float32)
        hidden_i = hidden_i.to(tl.float32)
        hidden_f = hidden_f.to(tl.float32)
        hidden_g = hidden_g.to(tl.float32)
        hidden_o = hidden_o.to(tl.float32)
        cx_value = cx_value.to(tl.float32)

    if HAS_BIAS:
        bias_i = tl.load(input_bias + cols)
        bias_f = tl.load(input_bias + HIDDEN_SIZE + cols)
        bias_g = tl.load(input_bias + 2 * HIDDEN_SIZE + cols)
        bias_o = tl.load(input_bias + 3 * HIDDEN_SIZE + cols)
        hidden_bias_i = tl.load(hidden_bias + cols)
        hidden_bias_f = tl.load(hidden_bias + HIDDEN_SIZE + cols)
        hidden_bias_g = tl.load(hidden_bias + 2 * HIDDEN_SIZE + cols)
        hidden_bias_o = tl.load(hidden_bias + 3 * HIDDEN_SIZE + cols)
        if not IS_FP64:
            bias_i = bias_i.to(tl.float32)
            bias_f = bias_f.to(tl.float32)
            bias_g = bias_g.to(tl.float32)
            bias_o = bias_o.to(tl.float32)
            hidden_bias_i = hidden_bias_i.to(tl.float32)
            hidden_bias_f = hidden_bias_f.to(tl.float32)
            hidden_bias_g = hidden_bias_g.to(tl.float32)
            hidden_bias_o = hidden_bias_o.to(tl.float32)

    gate_i = input_i + hidden_i
    gate_f = input_f + hidden_f
    gate_g = input_g + hidden_g
    gate_o = input_o + hidden_o
    if HAS_BIAS:
        gate_i += bias_i
        gate_i += hidden_bias_i
        gate_f += bias_f
        gate_f += hidden_bias_f
        gate_g += bias_g
        gate_g += hidden_bias_g
        gate_o += bias_o
        gate_o += hidden_bias_o

    cy_value = tl.sigmoid(gate_f) * cx_value + tl.sigmoid(gate_i) * _tanh(gate_g)
    hy_value = tl.sigmoid(gate_o) * _tanh(cy_value)

    # Recurrence state (fp32 buffers) and the rounded sequence output.  These
    # three stores are unconditional (extra conditional "only on the last
    # step" stores of the same value were measured to write only a subset of
    # lanes on this backend, so the final ``hy``/``cy`` come from the separate
    # ``mkldnn_rnn_layer_final_kernel`` instead).
    tl.store(h_out + offsets, hy_value, mask=lane_mask)
    tl.store(c_out + offsets, cy_value, mask=lane_mask)
    tl.store(seq_out + offsets, hy_value, mask=lane_mask)


@triton.jit
def mkldnn_rnn_layer_gate_kernel_full(
    input_gates,
    hidden_gates,
    cx,
    input_bias,
    hidden_bias,
    h_out,
    c_out,
    seq_out,
    HIDDEN_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Mask-free fast path: ``batch * hidden`` is a multiple of ``BLOCK_SIZE``.

    Identical math to ``mkldnn_rnn_layer_gate_kernel`` with the tail
    clamp/predicate removed.  The mask costs ~3.5x on this backend (measured
    60.5 us vs 17 us per launch for ``batch*hidden = 2048``), which is why the
    branchless variant exists at all.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // HIDDEN_SIZE
    cols = offsets - rows * HIDDEN_SIZE
    gate_base = rows * (4 * HIDDEN_SIZE) + cols

    # Gate order is input, forget, cell, output.
    input_i = tl.load(input_gates + gate_base)
    input_f = tl.load(input_gates + gate_base + HIDDEN_SIZE)
    input_g = tl.load(input_gates + gate_base + 2 * HIDDEN_SIZE)
    input_o = tl.load(input_gates + gate_base + 3 * HIDDEN_SIZE)
    hidden_i = tl.load(hidden_gates + gate_base)
    hidden_f = tl.load(hidden_gates + gate_base + HIDDEN_SIZE)
    hidden_g = tl.load(hidden_gates + gate_base + 2 * HIDDEN_SIZE)
    hidden_o = tl.load(hidden_gates + gate_base + 3 * HIDDEN_SIZE)
    cx_value = tl.load(cx + offsets)

    if not IS_FP64:
        input_i = input_i.to(tl.float32)
        input_f = input_f.to(tl.float32)
        input_g = input_g.to(tl.float32)
        input_o = input_o.to(tl.float32)
        hidden_i = hidden_i.to(tl.float32)
        hidden_f = hidden_f.to(tl.float32)
        hidden_g = hidden_g.to(tl.float32)
        hidden_o = hidden_o.to(tl.float32)
        cx_value = cx_value.to(tl.float32)

    if HAS_BIAS:
        bias_i = tl.load(input_bias + cols)
        bias_f = tl.load(input_bias + HIDDEN_SIZE + cols)
        bias_g = tl.load(input_bias + 2 * HIDDEN_SIZE + cols)
        bias_o = tl.load(input_bias + 3 * HIDDEN_SIZE + cols)
        hidden_bias_i = tl.load(hidden_bias + cols)
        hidden_bias_f = tl.load(hidden_bias + HIDDEN_SIZE + cols)
        hidden_bias_g = tl.load(hidden_bias + 2 * HIDDEN_SIZE + cols)
        hidden_bias_o = tl.load(hidden_bias + 3 * HIDDEN_SIZE + cols)
        if not IS_FP64:
            bias_i = bias_i.to(tl.float32)
            bias_f = bias_f.to(tl.float32)
            bias_g = bias_g.to(tl.float32)
            bias_o = bias_o.to(tl.float32)
            hidden_bias_i = hidden_bias_i.to(tl.float32)
            hidden_bias_f = hidden_bias_f.to(tl.float32)
            hidden_bias_g = hidden_bias_g.to(tl.float32)
            hidden_bias_o = hidden_bias_o.to(tl.float32)

    gate_i = input_i + hidden_i
    gate_f = input_f + hidden_f
    gate_g = input_g + hidden_g
    gate_o = input_o + hidden_o
    if HAS_BIAS:
        gate_i += bias_i
        gate_i += hidden_bias_i
        gate_f += bias_f
        gate_f += hidden_bias_f
        gate_g += bias_g
        gate_g += hidden_bias_g
        gate_o += bias_o
        gate_o += hidden_bias_o

    cy_value = tl.sigmoid(gate_f) * cx_value + tl.sigmoid(gate_i) * _tanh(gate_g)
    hy_value = tl.sigmoid(gate_o) * _tanh(cy_value)

    tl.store(h_out + offsets, hy_value)
    tl.store(c_out + offsets, cy_value)
    tl.store(seq_out + offsets, hy_value)


@triton.jit
def mkldnn_rnn_layer_final_kernel(
    h_src,
    c_src,
    hy,
    cy,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Publish the final fp32 recurrence state as ``hy``/``cy``.

    Kept out of the gate kernel: a conditional in-kernel store of the same
    value miscompiled (partial lane writes), and this extra launch is cheap.
    Same clamp-load / masked-store treatment as the gate kernel, so sub-warp
    ``N`` cannot spill past ``hy``/``cy``.
    """
    raw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lane_mask = raw < N
    offsets = tl.minimum(raw, N - 1)
    tl.store(hy + offsets, tl.load(h_src + offsets), mask=lane_mask)
    tl.store(cy + offsets, tl.load(c_src + offsets), mask=lane_mask)
