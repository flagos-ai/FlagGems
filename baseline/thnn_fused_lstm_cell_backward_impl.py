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

"""NVIDIA native baseline for `thnn_fused_lstm_cell_backward_impl`.

Mirrors benchmark/test_thnn_fused_lstm_cell_backward_impl.py
(LSTMCellBackwardBenchmark, a base.Benchmark subclass):
    op_name  = "thnn_fused_lstm_cell_backward_impl"
    torch_op = torch.ops.aten._thnn_fused_lstm_cell_backward_impl
    dtypes   = [torch.float32]

The class overrides set_shapes with LSTM_SHAPES -- the full
(batch, hidden) cross of batch in [1, 2, 4, 8, 16] and hidden in
[4, 8, 16, 32, 64] -- so core_shapes.yaml is not consulted.

cy and the workspace come from a real torch.ops.aten._thnn_fused_lstm_cell
forward, which build_inputs performs outside the timed region. Note the
input_bias is zeros while the hidden_bias is randn, as in the benchmark.
has_bias is True.

float32 only: the underlying ATen operator supports neither Half nor BFloat16.

This calls the ATen backward kernel directly; it is not an autograd benchmark.
"""

import torch

OP_NAME = "thnn_fused_lstm_cell_backward_impl"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._thnn_fused_lstm_cell_backward_impl is unavailable"
)

# test_thnn_fused_lstm_cell_backward_impl.py::LSTM_SHAPES -- (batch, hidden)
_SHAPES = [
    (batch, hidden) for batch in (1, 2, 4, 8, 16) for hidden in (4, 8, 16, 32, 64)
]

_HAS_BIAS = True


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def native():
    op = _resolve("torch.ops.aten._thnn_fused_lstm_cell_backward_impl")
    return op if callable(op) else None


def grid():
    return [{"batch": batch, "hidden": hidden} for batch, hidden in _SHAPES]


def build_inputs(binding, dtype, device):
    batch_size, hidden_size = binding["batch"], binding["hidden"]
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    input_bias = torch.zeros(4 * hidden_size, dtype=dtype, device=device)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=device)
    hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )
    grad_hy = torch.randn_like(hx)
    grad_cy = torch.randn_like(cy)
    return (grad_hy, grad_cy, cx, cy, workspace, _HAS_BIAS), {}


def key_shape(binding):
    return [binding["batch"], binding["hidden"]]


def config(binding, dtype):
    batch, hidden = binding["batch"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [batch, hidden], "dtype": dt, "note": "grad_hy"},
                {"shape": [batch, hidden], "dtype": dt, "note": "grad_cy"},
                {"shape": [batch, hidden], "dtype": dt, "note": "cx"},
                {"shape": [batch, hidden], "dtype": dt, "note": "cy"},
                {
                    "shape": [batch, 4 * hidden],
                    "dtype": dt,
                    "note": "workspace from the forward",
                },
                {"scalar": _HAS_BIAS},
            ],
            "kwargs": {},
        },
        "outputs": {
            "grad_gates": {"shape": [batch, 4 * hidden], "dtype": dt},
            "grad_cx": {"shape": [batch, hidden], "dtype": dt},
            "grad_bias": {"shape": [4 * hidden], "dtype": dt},
        },
        "dims": {"batch": batch, "hidden": hidden},
    }
