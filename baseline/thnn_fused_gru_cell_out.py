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

"""NVIDIA native baseline for `thnn_fused_gru_cell_out`.

Mirrors benchmark/test_thnn_fused_gru_cell.py (FusedGRUCellOutBenchmark, a
base.Benchmark subclass):
    op_name  = "thnn_fused_gru_cell_out"
    torch_op = _fused_gru_cell_out, which calls
               torch.ops.aten._thnn_fused_gru_cell.out
    dtypes   = consts.FLOAT_DTYPES

The class overrides set_shapes with the (batch, hidden) pairs of CORE_SHAPES
and only adds COMPREHENSIVE_SHAPES at --level comprehensive, so
core_shapes.yaml is not consulted and the three large shapes are not in this
grid. The third element of each CORE_SHAPES tuple is unused by set_shapes.

Gate tensors are [batch, 3 * hidden], hx is [batch, hidden], the biases are
[3 * hidden], and the two outputs are [batch, hidden] and [batch, 5 * hidden].
out0/out1 must be keywords for the .out overload.

float64 is in the dtype list because the benchmark appends it when
flag_gems.runtime.device.support_fp64 is true, and the NVIDIA backend
reports fp64_enabled=True.
"""

import torch

OP_NAME = "thnn_fused_gru_cell_out"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
IS_INPLACE = False
MUTATED_INPUTS = (5, 6)
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._thnn_fused_gru_cell.out is unavailable on this torch build"
)

# test_thnn_fused_gru_cell.py::CORE_SHAPES, read as (batch, hidden)
_SHAPES = [
    (1, 32),
    (8, 64),
    (32, 128),
    (64, 256),
    (128, 512),
    (256, 1024),
]


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


def _fused_gru_cell_out(
    input_gates, hidden_gates, hx, input_bias, hidden_bias, out0, out1
):
    """benchmark::_fused_gru_cell_out -- out0/out1 forwarded as keywords."""
    return torch.ops.aten._thnn_fused_gru_cell.out(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        out0=out0,
        out1=out1,
    )


def native():
    if _resolve("torch.ops.aten._thnn_fused_gru_cell.out") is None:
        return None
    return _fused_gru_cell_out


def grid():
    return [{"batch": batch, "hidden": hidden} for batch, hidden in _SHAPES]


def build_inputs(binding, dtype, device):
    batch_size, hidden_size = binding["batch"], binding["hidden"]
    input_gates = torch.randn(batch_size, 3 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    input_bias = torch.randn(3 * hidden_size, dtype=dtype, device=device)
    hidden_bias = torch.randn_like(input_bias)
    out0 = torch.empty_like(hx)
    out1 = torch.empty(batch_size, 5 * hidden_size, dtype=dtype, device=device)
    return (
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        out0,
        out1,
    ), {}


def key_shape(binding):
    return [binding["batch"], binding["hidden"]]


def config(binding, dtype):
    batch, hidden = binding["batch"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {"shape": [batch, 3 * hidden], "dtype": dt},
                {"shape": [batch, 3 * hidden], "dtype": dt},
                {"shape": [batch, hidden], "dtype": dt},
                {"shape": [3 * hidden], "dtype": dt},
                {"shape": [3 * hidden], "dtype": dt},
                {"shape": [batch, hidden], "dtype": dt, "note": "out0"},
                {"shape": [batch, 5 * hidden], "dtype": dt, "note": "out1"},
            ],
            "kwargs": {},
        },
        "outputs": {
            "out0": {"shape": [batch, hidden], "dtype": dt},
            "out1": {"shape": [batch, 5 * hidden], "dtype": dt},
        },
        "dims": {"batch": batch, "hidden": hidden},
    }
