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
# GRU gate kernel for the Kunlunxin ``gru_cell`` override, kept in its own
# module: defining it inside ``gru_cell.py`` makes a cold XPU compile abort with
# ``KeyError: 'tt.sigmoid'`` once a ``tl.dot`` kernel has been built in the same
# process, while the identical kernel defined/imported from another module (the
# backend ``_thnn_fused_gru_cell`` module already does this) compiles cleanly.
#
# This is the body of ``_thnn_fused_gru_cell_flat_kernel_nomask`` with the
# backward workspace stores removed.  ``HAS_BIAS`` / ``IS_FP64`` must stay
# ``tl.constexpr`` parameters and the ``offsets // HIDDEN_SIZE`` split must stay
# as-is: simpler 1-D/2-D variants trip the same cold-compile bug.

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

_tanh = tl_extra_shim.tanh


@triton.jit
def gru_cell_gate_kernel(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    HIDDEN_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // HIDDEN_SIZE
    cols = offsets - rows * HIDDEN_SIZE
    gate_base = rows * (3 * HIDDEN_SIZE) + cols

    input_reset = tl.load(input_gates + gate_base)
    input_gate = tl.load(input_gates + gate_base + HIDDEN_SIZE)
    input_new = tl.load(input_gates + gate_base + 2 * HIDDEN_SIZE)
    hidden_reset = tl.load(hidden_gates + gate_base)
    hidden_gate = tl.load(hidden_gates + gate_base + HIDDEN_SIZE)
    hidden_new = tl.load(hidden_gates + gate_base + 2 * HIDDEN_SIZE)
    hx_value = tl.load(hx + offsets)
    if not IS_FP64:
        input_reset = input_reset.to(tl.float32)
        input_gate = input_gate.to(tl.float32)
        input_new = input_new.to(tl.float32)
        hidden_reset = hidden_reset.to(tl.float32)
        hidden_gate = hidden_gate.to(tl.float32)
        hidden_new = hidden_new.to(tl.float32)
        hx_value = hx_value.to(tl.float32)

    if HAS_BIAS:
        bias_reset = tl.load(input_bias + cols)
        bias_update = tl.load(input_bias + HIDDEN_SIZE + cols)
        bias_new = tl.load(input_bias + 2 * HIDDEN_SIZE + cols)
        hidden_bias_reset = tl.load(hidden_bias + cols)
        hidden_bias_update = tl.load(hidden_bias + HIDDEN_SIZE + cols)
        hidden_bias_new = tl.load(hidden_bias + 2 * HIDDEN_SIZE + cols)
        if not IS_FP64:
            bias_reset = bias_reset.to(tl.float32)
            bias_update = bias_update.to(tl.float32)
            bias_new = bias_new.to(tl.float32)
            hidden_bias_reset = hidden_bias_reset.to(tl.float32)
            hidden_bias_update = hidden_bias_update.to(tl.float32)
            hidden_bias_new = hidden_bias_new.to(tl.float32)

    reset_gate = input_reset + hidden_reset
    update_gate = input_gate + hidden_gate
    if HAS_BIAS:
        reset_gate += bias_reset
        reset_gate += hidden_bias_reset
        update_gate += bias_update
        update_gate += hidden_bias_update
        input_new += bias_new
        hidden_new += hidden_bias_new

    reset_gate = tl.sigmoid(reset_gate)
    update_gate = tl.sigmoid(update_gate)
    new_gate = input_new + reset_gate * hidden_new
    new_gate = _tanh(new_gate)
    hy_value = new_gate + update_gate * (hx_value - new_gate)
    tl.store(hy + offsets, hy_value)
